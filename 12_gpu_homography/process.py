"""
Approche A — GPU — Homographie RANSAC + Médiane glissante (N=5)

• SIFT + FLANN + Homographie RANSAC calculés sur CPU à 0.25× (rapide, ~50ms/paire)
  → précomputation parallèle (ThreadPool) de toutes les paires en amont
• warpPerspective + médiane temporelle exécutés sur GPU (PyTorch)
• Sortie : 4K (3840×2160), vidéo complète GX010236_synced_enhanced.MP4

GPU assignment : --gpu 0  (méthode A + D partagent GPU0, E + J partage GPU1)

Usage :
  python 12_gpu_homography/process.py [--gpu 0] [--test-only]
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

CACHE_PATH = os.path.join(os.path.dirname(__file__), '..', 'homography_cache_half4.pkl')

VIDEO_PATH  = os.path.join(os.path.dirname(__file__), '..', 'GX010236_synced_enhanced.MP4')
OUTPUT_DIR  = os.path.dirname(__file__)
SIFT_SCALE  = 0.25
N           = 5
TEST_INDICES = [15, 75, 130]


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


# ── SIFT + Homographie (CPU) ─────────────────────────────────────────────────

def compute_sift_homography(ref_small, neighbor_small):
    """Calcule H alignant neighbor sur ref à résolution small. Retourne None si échec."""
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


def precompute_homographies(small_frames, half, n_workers=16):
    # Charger le cache partagé si disponible
    cache_path = os.path.normpath(CACHE_PATH)
    if os.path.exists(cache_path):
        print(f"  Chargement cache partagé : {cache_path}")
        t0 = time.time()
        with open(cache_path, 'rb') as f:
            H_cache = pickle.load(f)
        print(f"  → {len(H_cache)} homographies chargées en {time.time()-t0:.1f}s")
        return H_cache
    # Sinon : calcul local
    n     = len(small_frames)
    tasks = [(i, j)
             for i in range(n)
             for j in range(max(0, i - half), min(n, i + half + 1))
             if j != i]
    print(f"  Précomputation de {len(tasks)} homographies ({n_workers} workers)…")
    t0      = time.time()
    H_cache = {}
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
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


# ── GPU : warp + médiane ─────────────────────────────────────────────────────

def warp_gpu(frame_t, H_np, device):
    """
    Warp frame_t (C,H,W) float32 sur device via homographie H_np (dst→src).
    Retourne (C,H,W) float32.
    """
    C, fH, fW = frame_t.shape
    ys = torch.arange(fH, dtype=torch.float32, device=device)
    xs = torch.arange(fW, dtype=torch.float32, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')          # (fH, fW)
    ones     = torch.ones(fH, fW, dtype=torch.float32, device=device)
    coords   = torch.stack([gx.reshape(-1), gy.reshape(-1), ones.reshape(-1)], 0)  # (3, fH*fW)

    H_t   = torch.from_numpy(H_np).float().to(device)
    src   = H_t @ coords                                     # (3, fH*fW)
    denom = src[2:3].clamp(min=1e-6)
    sx    = src[0:1] / denom                                 # (1, fH*fW) pixel x source
    sy    = src[1:2] / denom                                 # (1, fH*fW) pixel y source

    # Normalise [-1, 1]
    nx = sx / (fW - 1) * 2.0 - 1.0
    ny = sy / (fH - 1) * 2.0 - 1.0
    grid = torch.stack([nx, ny], dim=-1).reshape(1, fH, fW, 2)

    warped = F.grid_sample(frame_t.unsqueeze(0), grid,
                           mode='bilinear', padding_mode='reflection',
                           align_corners=True)
    return warped.squeeze(0)


# ── Traitement d'une frame ────────────────────────────────────────────────────

def process_frame_gpu(frames_full, H_cache, center_idx, half, device):
    start = max(0, center_idx - half)
    end   = min(len(frames_full), center_idx + half + 1)
    ref_f = frames_full[center_idx]

    # Référence sur GPU
    ref_t = torch.from_numpy(ref_f).float().permute(2, 0, 1).to(device)  # (C,H,W)

    aligned = [ref_t]
    for i in range(start, end):
        if i == center_idx:
            continue
        H_small = H_cache.get((center_idx, i))
        H_full  = scale_homography(H_small, SIFT_SCALE, 1.0)
        if H_full is None:
            aligned.append(ref_t)
        else:
            neigh_t = torch.from_numpy(frames_full[i]).float().permute(2, 0, 1).to(device)
            aligned.append(warp_gpu(neigh_t, H_full, device))

    stack = torch.stack(aligned, dim=0)                          # (K, C, H, W)
    med   = torch.sort(stack, dim=0).values[len(aligned) // 2]  # (C, H, W)

    result = med.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return result


# ── Métriques ────────────────────────────────────────────────────────────────

def sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',       type=int,  default=0)
    parser.add_argument('--test-only', action='store_true')
    args   = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}')

    half = N // 2
    print(f"=== Méthode A GPU — Homographie RANSAC + Médiane N={N} ===")
    print(f"  GPU : {device} ({torch.cuda.get_device_name(args.gpu)})")

    # 1. Chargement
    print("Chargement frames small (0.25×)…")
    t0 = time.time()
    small_frames, fps = load_frames(VIDEO_PATH, SIFT_SCALE)
    print(f"  {len(small_frames)} frames small en {time.time()-t0:.1f}s")

    print("Chargement frames 4K…")
    t0 = time.time()
    full_frames, _  = load_frames(VIDEO_PATH, 1.0)
    h, w = full_frames[0].shape[:2]
    print(f"  {len(full_frames)} frames 4K ({w}×{h}) en {time.time()-t0:.1f}s")

    # 2. Précomputation homographies
    t0 = time.time()
    H_cache = precompute_homographies(small_frames, half)

    # 3. Test frames
    print(f"\n{'Frame':>6}  {'Original':>10}  {'Result':>10}  {'Δ%':>7}  {'Time':>6}")
    print("-" * 48)
    rows = []
    for idx in TEST_INDICES:
        t1 = time.time()
        result = process_frame_gpu(full_frames, H_cache, idx, half, device)
        dt = time.time() - t1
        so = sharpness(full_frames[idx])
        sr = sharpness(result)
        print(f"{idx:>6}  {so:>10.0f}  {sr:>10.0f}  {(sr-so)/so*100:>+6.1f}%  {dt:>5.2f}s")
        orig   = full_frames[idx].copy()
        result = result.copy()
        cv2.putText(orig,   f'Original {idx}',    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        cv2.putText(result, f'A GPU N={N}',        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
        ds = min(1.0, 1920 / w)
        rows.append(np.hstack([cv2.resize(orig, (0,0), fx=ds, fy=ds),
                                cv2.resize(result, (0,0), fx=ds, fy=ds)]))
    comp_path = os.path.join(OUTPUT_DIR, 'comparaison_A_gpu.jpg')
    cv2.imwrite(comp_path, np.vstack(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    print(f"\nComparaison → {comp_path}")

    if args.test_only:
        print("Mode test-only : terminé.")
        return

    # 4. Vidéo complète
    vid_path = os.path.join(OUTPUT_DIR, f'result_A_gpu_4k_N{N}.mp4')
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
