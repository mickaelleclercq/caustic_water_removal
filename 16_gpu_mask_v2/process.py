"""
Approche E v2 — GPU — Masque multi-échelle + Décomposition LF/HF

Améliorations par rapport à E v1 (14_gpu_mask_inpaint) :
  1. Masque top-hat multi-échelle (kernels 11, 21, 33 px) → détecte toutes les tailles
  2. Seuil plus agressif (σ=0.8 au lieu de 1.5) + dilatation plus forte (3 iters, 9px)
  3. N=9 au lieu de N=7 → médiane plus stable
  4. ★ Décomposition basse/haute fréquence (σ_lf=20px)
     Au lieu de remplacer entièrement les pixels caustiques par la médiane (→ flou),
     on ne remplace QUE la composante LF (illumination) par celle de la médiane,
     en conservant la HF (texture) de la frame originale.
     Formule: résultat = ref + masque × (med_LF − ref_LF)
     → perte de netteté ≈ 0 % en dehors des caustiques

GPU assignment : --gpu 1 (par défaut)

Usage :
  python 16_gpu_mask_v2/process.py [--gpu 1] [--test-only]
"""
import cv2
import numpy as np
import time
import os
import argparse
import pickle
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE_PATH  = os.path.join(os.path.dirname(__file__), '..', 'homography_cache_half4.pkl')
VIDEO_PATH  = os.path.join(os.path.dirname(__file__), '..', 'GX010236_synced_enhanced.MP4')
OUTPUT_DIR  = os.path.dirname(__file__)

SIFT_SCALE     = 0.25
N              = 9                 # fenêtre temporelle (v1 = 7)
TOPHAT_KERNELS = [11, 21, 33]     # multi-échelle (v1 = [15])
TOPHAT_SIG     = 0.8              # seuil détection (v1 = 1.5)
DILATE_K       = 9                # kernel dilatation (v1 = 7)
DILATE_ITER    = 3                # itérations dilatation (v1 = 1)
BLUR_SMOOTH    = (21, 21)         # lissage bords masque (v1 = (11,11))
BLUR_SIG_MASK  = 5.0              # sigma lissage masque (v1 = 3.0)
SIGMA_LF       = 20.0             # sigma décomposition LF/HF (nouveau)
TEST_INDICES   = [15, 75, 130]


# ── Chargement ──────────────────────────────────────────────────────────────

def load_frames(video_path, scale):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if abs(scale - 1.0) < 1e-4:
            frames.append(frame)
        else:
            frames.append(cv2.resize(frame, (0, 0), fx=scale, fy=scale))
    cap.release()
    return frames, fps


# ── Masque caustiques multi-échelle (CPU) ────────────────────────────────────

def detect_caustics_mask_multiscale(frame_bgr):
    """
    Top-Hat morphologique à plusieurs échelles (max sur les 3 kernels),
    seuil agressif, deux étapes de dilatation → masque float32 [0,1].
    0 = pixel intact, 1 = caustique à remplacer.
    """
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v    = hsv[:, :, 2].astype(np.float32)

    # Combine top-hat de plusieurs tailles → sensible à toutes les échelles
    tophat = np.zeros_like(v)
    for ks in TOPHAT_KERNELS:
        kern   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        tophat = np.maximum(tophat, cv2.morphologyEx(v, cv2.MORPH_TOPHAT, kern))

    thresh  = tophat.mean() + TOPHAT_SIG * tophat.std()
    mask_u8 = ((tophat > thresh).astype(np.uint8) * 255)

    dil_kn  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_K, DILATE_K))
    mask_u8 = cv2.dilate(mask_u8, dil_kn, iterations=DILATE_ITER)
    mask_f  = cv2.GaussianBlur(
        mask_u8.astype(np.float32), BLUR_SMOOTH, BLUR_SIG_MASK) / 255.0
    return mask_f  # (H, W)


# ── SIFT + Homographie (CPU) ─────────────────────────────────────────────────

def compute_sift_homography(ref_small, neighbor_small):
    sift = cv2.SIFT_create(nfeatures=2000)
    g1 = cv2.cvtColor(ref_small,      cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(neighbor_small, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None
    flann   = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good    = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good) < 10:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
    return H


def scale_homography(H_small, sift_scale=SIFT_SCALE, out_scale=1.0):
    if H_small is None:
        return None
    S_sift    = np.diag([sift_scale,      sift_scale,      1.0])
    S_out_inv = np.diag([1.0 / out_scale, 1.0 / out_scale, 1.0])
    return S_out_inv @ H_small @ S_sift


def precompute_homographies(small_frames, half):
    cache_path = os.path.normpath(CACHE_PATH)
    if os.path.exists(cache_path):
        print(f"  Chargement cache partagé : {cache_path}")
        t0 = time.time()
        with open(cache_path, 'rb') as f:
            H_cache = pickle.load(f)
        print(f"  → {len(H_cache)} homographies chargées en {time.time()-t0:.1f}s")
        return H_cache
    n     = len(small_frames)
    tasks = [(i, j)
             for i in range(n)
             for j in range(max(0, i - half), min(n, i + half + 1))
             if j != i]
    print(f"  Précomputation de {len(tasks)} homographies…")
    t0      = time.time()
    H_cache = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(compute_sift_homography, small_frames[i], small_frames[j]): (i, j)
                for i, j in tasks}
        done = 0
        for fut in as_completed(futs):
            i, j = futs[fut]
            H_cache[(i, j)] = fut.result()
            done += 1
            if done % 1000 == 0:
                print(f"    {done}/{len(tasks)} ({time.time()-t0:.0f}s)")
    print(f"  → {len(tasks)} homographies en {time.time()-t0:.1f}s")
    return H_cache


# ── GPU : warp ────────────────────────────────────────────────────────────────

def warp_gpu(frame_t, H_np, device):
    """Warp frame_t (C,H,W) float32 via homographie H_np (dst→src)."""
    C, fH, fW = frame_t.shape
    ys = torch.arange(fH, dtype=torch.float32, device=device)
    xs = torch.arange(fW, dtype=torch.float32, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    ones   = torch.ones(fH, fW, dtype=torch.float32, device=device)
    coords = torch.stack([gx.reshape(-1), gy.reshape(-1), ones.reshape(-1)], 0)

    H_t   = torch.from_numpy(H_np).float().to(device)
    src   = H_t @ coords
    denom = src[2:3].clamp(min=1e-6)
    sx    = src[0:1] / denom
    sy    = src[1:2] / denom

    nx   = sx / (fW - 1) * 2.0 - 1.0
    ny   = sy / (fH - 1) * 2.0 - 1.0
    grid = torch.stack([nx, ny], dim=-1).reshape(1, fH, fW, 2)

    warped = F.grid_sample(frame_t.unsqueeze(0), grid,
                           mode='bilinear', padding_mode='reflection',
                           align_corners=True)
    return warped.squeeze(0)


# ── GPU : flou gaussien séparable (LF/HF) ────────────────────────────────────

def gaussian_blur_gpu(t, sigma, device):
    """
    Flou gaussien séparable sur tenseur (C,H,W) float32.
    Approximation efficace : passage aux 4K via avg_pool downsample,
    filtre au quart de résolution, upsample → sigma effectif = sigma.
    """
    C, H, W = t.shape
    # Downsample ×4 pour réduire la taille du noyau
    ds  = 4
    sig = max(sigma / ds, 1.5)
    k   = int(sig * 6 + 1) | 1   # odd

    x      = torch.arange(k, dtype=torch.float32, device=device) - k // 2
    kern1d = torch.exp(-0.5 * (x / sig) ** 2)
    kern1d = kern1d / kern1d.sum()

    # Flou appliqué à résolution réduite
    batch  = t.unsqueeze(0)                                  # (1, C, H, W)
    small  = F.avg_pool2d(batch, kernel_size=ds, stride=ds)  # (1, C, H/4, W/4)

    # Convolution séparable
    p = k // 2
    kern_h = kern1d.view(1, 1, 1, k).expand(C, 1, 1, k)
    kern_v = kern1d.view(1, 1, k, 1).expand(C, 1, k, 1)

    blr = F.conv2d(F.pad(small, [p, p, 0, 0], mode='reflect'), kern_h, groups=C)
    blr = F.conv2d(F.pad(blr,   [0, 0, p, p], mode='reflect'), kern_v, groups=C)

    # Remonter à la résolution d'origine
    out = F.interpolate(blr, size=(H, W), mode='bilinear', align_corners=False)
    return out.squeeze(0)   # (C, H, W)


# ── Traitement d'une frame ────────────────────────────────────────────────────

def process_frame_gpu(frames_full, H_cache, center_idx, half, device):
    ref_np = frames_full[center_idx]
    ref_t  = torch.from_numpy(ref_np).float().permute(2, 0, 1).to(device)  # (C,H,W)

    # Masque caustiques multi-échelle (CPU)
    mask_np = detect_caustics_mask_multiscale(ref_np)                    # (H, W) float32
    mask_t  = torch.from_numpy(mask_np).float().to(device).unsqueeze(0)  # (1, H, W)

    # Aligner voisins sur GPU
    start   = max(0, center_idx - half)
    end     = min(len(frames_full), center_idx + half + 1)
    aligned = [ref_t]
    for i in range(start, end):
        if i == center_idx:
            continue
        H_small = H_cache.get((center_idx, i))
        H_full  = scale_homography(H_small)
        if H_full is None:
            aligned.append(ref_t)
        else:
            neigh_t = torch.from_numpy(frames_full[i]).float().permute(2, 0, 1).to(device)
            aligned.append(warp_gpu(neigh_t, H_full, device))

    stack = torch.stack(aligned, dim=0)                            # (K, C, H, W)
    med_t = torch.sort(stack, dim=0).values[len(aligned) // 2]    # (C, H, W)

    # ── Décomposition LF / HF ────────────────────────────────────────────────
    # ref_lf  : illumination basse fréquence de la frame originale
    # med_lf  : illumination basse fréquence de la médiane (stable → sans caustiques)
    # ref_hf  : texture haute fréquence (non contaminée par les caustiques sur les
    #           détails fins ; on la conserve pour éviter tout flou)
    #
    # Formule : résultat = ref + masque × (med_lf − ref_lf)
    #   → en dehors du masque : frame originale intacte (0 % de perte de netteté)
    #   → dans le masque : on swap juste l'illumination (LF) sans toucher la texture (HF)
    ref_lf = gaussian_blur_gpu(ref_t, SIGMA_LF, device)
    med_lf = gaussian_blur_gpu(med_t, SIGMA_LF, device)

    result = ref_t + mask_t * (med_lf - ref_lf)

    return result.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()


# ── Métriques ────────────────────────────────────────────────────────────────

def sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',       type=int,  default=1)
    parser.add_argument('--test-only', action='store_true')
    args   = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}')

    half = N // 2
    print(f"=== Méthode E v2 GPU — Masque multi-échelle + LF/HF N={N} σ_lf={SIGMA_LF} ===")
    print(f"  GPU : {device} ({torch.cuda.get_device_name(args.gpu)})")
    print(f"  Top-Hat kernels : {TOPHAT_KERNELS}, seuil σ={TOPHAT_SIG}")
    print(f"  Dilatation : {DILATE_ITER}×{DILATE_K}px, lissage masque σ={BLUR_SIG_MASK}")

    # 1. Chargement
    print("Chargement frames small (0.25×)…")
    t0 = time.time()
    small_frames, fps = load_frames(VIDEO_PATH, SIFT_SCALE)
    print(f"  {len(small_frames)} frames en {time.time()-t0:.1f}s")

    print("Chargement frames 4K…")
    t0 = time.time()
    full_frames, _ = load_frames(VIDEO_PATH, 1.0)
    h, w = full_frames[0].shape[:2]
    print(f"  {len(full_frames)} frames ({w}×{h}) en {time.time()-t0:.1f}s")

    # 2. Cache homographies (half=4 couvre N≤9)
    H_cache = precompute_homographies(small_frames, half)

    # 3. Test frames
    print(f"\n{'Frame':>6}  {'Original':>10}  {'Result':>10}  {'Δ%':>7}  {'Time':>6}")
    print("-" * 48)
    rows = []
    for idx in TEST_INDICES:
        t1     = time.time()
        result = process_frame_gpu(full_frames, H_cache, idx, half, device)
        dt     = time.time() - t1
        so     = sharpness(full_frames[idx])
        sr     = sharpness(result)
        print(f"{idx:>6}  {so:>10.0f}  {sr:>10.0f}  {(sr-so)/so*100:>+6.1f}%  {dt:>5.2f}s")
        orig   = full_frames[idx].copy()
        res_cp = result.copy()
        cv2.putText(orig,   f'Original {idx}',          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(res_cp, f'E v2 LF/HF N={N}',        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        ds = min(1.0, 1920 / w)
        rows.append(np.hstack([cv2.resize(orig,   (0, 0), fx=ds, fy=ds),
                                cv2.resize(res_cp, (0, 0), fx=ds, fy=ds)]))
    comp_path = os.path.join(OUTPUT_DIR, 'comparaison_E_v2.jpg')
    cv2.imwrite(comp_path, np.vstack(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    print(f"\nComparaison → {comp_path}")

    if args.test_only:
        print("Mode test-only : terminé.")
        return

    # 4. Vidéo complète
    vid_path = os.path.join(OUTPUT_DIR, f'result_E_v2_gpu_4k_N{N}.mp4')
    print(f"\nTraitement {len(full_frames)} frames → {vid_path}")
    t0 = time.time()
    writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for idx in range(len(full_frames)):
        writer.write(process_frame_gpu(full_frames, H_cache, idx, half, device))
        if idx % 50 == 0:
            el  = time.time() - t0
            eta = el / (idx + 1) * (len(full_frames) - idx - 1)
            print(f"  Frame {idx:>5}/{len(full_frames)}  ETA {eta:.0f}s")
    writer.release()
    total = time.time() - t0
    print(f"Vidéo → {vid_path}  ({total:.0f}s, {total/len(full_frames):.2f}s/frame)")


if __name__ == '__main__':
    main()
