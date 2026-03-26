"""
Approche J — GPU — Pyramide Laplacienne + suppression sélective excès fin (N=9)

• SIFT + Homographie précomputés sur CPU (ThreadPool, 0.25×)
• warpPerspective sur GPU (PyTorch grid_sample)
• Pyramide Laplacienne L=4 construite et décomposée sur GPU (avg_pool2d + interpolate)
• Médiane temporelle par niveau sur GPU (torch.sort)
• Niveau 0 : suppression sélective de l'excès positif au percentile 98 (caustiques fines)
• Sortie : 4K (3840×2160), vidéo complète GX010236_synced_enhanced.MP4

GPU assignment : --gpu 1

Usage :
  python 15_gpu_pyramid_J/process.py [--gpu 1] [--test-only]
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
N           = 9
L           = 4   # niveaux pyramide
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


# ── GPU : pyramide Laplacienne ────────────────────────────────────────────────

def build_laplacian_pyramid_gpu(img_t, levels):
    """
    img_t : (C, H, W) float32
    Retourne une liste de (C, H', W') tensors (detail levels 0..levels-1) + résidu gaussien.
    """
    img_4d = img_t.unsqueeze(0)   # (1, C, H, W)
    gauss  = [img_4d]
    for _ in range(levels):
        gauss.append(F.avg_pool2d(gauss[-1], kernel_size=2, stride=2))

    pyramid = []
    for k in range(levels):
        h, w = gauss[k].shape[2:]
        up   = F.interpolate(gauss[k + 1], size=(h, w), mode='bilinear', align_corners=False)
        pyramid.append((gauss[k] - up).squeeze(0))   # (C, H', W')
    pyramid.append(gauss[-1].squeeze(0))              # résidu (C, H_small, W_small)
    return pyramid


def reconstruct_pyramid_gpu(pyramid):
    """Reconstruit l'image depuis la pyramide. Retourne (C, H, W)."""
    result = pyramid[-1].unsqueeze(0)   # (1, C, H_small, W_small)
    for k in range(len(pyramid) - 2, -1, -1):
        h, w   = pyramid[k].shape[1:]
        result = F.interpolate(result, size=(h, w), mode='bilinear', align_corners=False)
        result = result + pyramid[k].unsqueeze(0)
    return result.squeeze(0)            # (C, H, W)


# ── Traitement d'une frame ────────────────────────────────────────────────────

def process_frame_gpu(frames_full, H_cache, center_idx, half, levels, device):
    start  = max(0, center_idx - half)
    end    = min(len(frames_full), center_idx + half + 1)
    ref_t  = torch.from_numpy(frames_full[center_idx]).float().permute(2, 0, 1).to(device)

    # Aligner voisins sur GPU
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

    K = len(aligned)

    # Pyramide Laplacienne pour chaque frame alignée
    pyramids = [build_laplacian_pyramid_gpu(a, levels) for a in aligned]
    ref_pyr  = pyramids[0]

    result_pyramid = []
    for lvl in range(levels + 1):
        stack = torch.stack([p[lvl] for p in pyramids], dim=0)          # (K, C, H', W')
        med   = torch.sort(stack, dim=0).values[K // 2]                 # (C, H', W')

        if lvl == 0:
            # Niveau fin : suppression sélective de l'excès positif
            fine_orig = ref_pyr[0]                                       # (C, H, W)
            fine_med  = med
            excess    = (fine_orig - fine_med).clamp(min=0.0)           # (C, H, W)

            # Seuil = percentile 98 de l'excès positif (sur tous les canaux)
            pos_vals = excess[excess > 0]
            if pos_vals.numel() > 100:
                thresh = float(torch.quantile(pos_vals, 0.98))
            else:
                thresh = 50.0

            progress    = ((excess - thresh) / (thresh + 1e-6)).clamp(0, 1)
            fine_result = fine_orig - progress * excess
            result_pyramid.append(fine_result)
        else:
            result_pyramid.append(med)

    recon = reconstruct_pyramid_gpu(result_pyramid)
    return recon.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()


# ── Métriques ────────────────────────────────────────────────────────────────

def sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',       type=int, default=1)
    parser.add_argument('--test-only', action='store_true')
    args   = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}')

    half = N // 2
    print(f"=== Méthode J GPU — Pyramide Laplacienne L={L} N={N} ===")
    print(f"  GPU : {device} ({torch.cuda.get_device_name(args.gpu)})")

    # 1. Chargement
    print("Chargement frames small (0.25×)…")
    t0 = time.time()
    small_frames, fps = load_frames(VIDEO_PATH, SIFT_SCALE)
    print(f"  {len(small_frames)} frames en {time.time()-t0:.1f}s")

    print("Chargement frames 4K…")
    t0 = time.time()
    full_frames, _  = load_frames(VIDEO_PATH, 1.0)
    h, w = full_frames[0].shape[:2]
    print(f"  {len(full_frames)} frames ({w}×{h}) en {time.time()-t0:.1f}s")

    # 2. Précomputation homographies
    H_cache = precompute_homographies(small_frames, half)

    # Warm-up GPU
    _ = process_frame_gpu(full_frames, H_cache, TEST_INDICES[0], half, L, device)
    torch.cuda.synchronize(device)

    # 3. Test frames
    print(f"\n{'Frame':>6}  {'Original':>10}  {'Result':>10}  {'Δ%':>7}  {'Time':>6}")
    print("-" * 48)
    rows = []
    for idx in TEST_INDICES:
        torch.cuda.synchronize(device)
        t1 = time.time()
        result = process_frame_gpu(full_frames, H_cache, idx, half, L, device)
        torch.cuda.synchronize(device)
        dt = time.time() - t1
        so = sharpness(full_frames[idx])
        sr = sharpness(result)
        print(f"{idx:>6}  {so:>10.0f}  {sr:>10.0f}  {(sr-so)/so*100:>+6.1f}%  {dt:>5.2f}s")
        orig   = full_frames[idx].copy()
        result = result.copy()
        cv2.putText(orig,   f'Original {idx}',            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        cv2.putText(result, f'J GPU Pyramid L={L} N={N}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
        ds = min(1.0, 1920 / w)
        rows.append(np.hstack([cv2.resize(orig, (0,0), fx=ds, fy=ds),
                                cv2.resize(result, (0,0), fx=ds, fy=ds)]))
    comp_path = os.path.join(OUTPUT_DIR, 'comparaison_J_gpu.jpg')
    cv2.imwrite(comp_path, np.vstack(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    print(f"\nComparaison → {comp_path}")

    if args.test_only:
        print("Mode test-only : terminé.")
        return

    # 4. Vidéo complète
    vid_path = os.path.join(OUTPUT_DIR, f'result_J_gpu_4k_N{N}.mp4')
    print(f"\nTraitement {len(full_frames)} frames → {vid_path}")
    t0 = time.time()
    writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for idx in range(len(full_frames)):
        writer.write(process_frame_gpu(full_frames, H_cache, idx, half, L, device))
        if idx % 50 == 0:
            el  = time.time() - t0
            eta = el / (idx + 1) * (len(full_frames) - idx - 1)
            print(f"  Frame {idx:>5}/{len(full_frames)}  ETA {eta:.0f}s")
    writer.release()
    total = time.time() - t0
    print(f"Vidéo → {vid_path}  ({total:.0f}s, {total/len(full_frames):.2f}s/frame)")


if __name__ == '__main__':
    main()
