"""
Approche J — Pleine résolution 4K / 1080p

Stratégie pour éviter le goulot d'étranglement SIFT à 4K :
  - Charger les frames à 0.25 scale pour le calcul SIFT + homographie
  - Extrapoler la matrice H à la résolution cible : H_full = S⁻¹ · H_small · S
  - Appliquer warpPerspective + pyramide Laplace à la résolution cible

Benchmark estimé à 4K :
  SIFT 0.25 × 9 voisins ≈ 450ms
  Homographie × 8       ≈ 400ms
  Pyramid + warp 4K     ≈ 400ms
  Total / frame         ≈ 1.5s  →  150 frames ≈ 4 min

Usage :
  python process_J_4k.py [--scale 1.0|0.5|0.25] [--N 9] [--test-only]

  --scale 1.0  → 3840×2160 (4K)
  --scale 0.5  → 1920×1080 (FHD)
  --scale 0.25 → 960×540   (quarter, pour vérification rapide)
  --test-only  → traiter seulement les 3 frames témoins, pas de vidéo complète
"""
import cv2
import numpy as np
import time
import os
import argparse

VIDEO_PATH    = os.path.join(os.path.dirname(__file__), '..', '01_extraction', 'subclip_5s.mp4')
OUTPUT_DIR    = os.path.dirname(__file__)

SIFT_SCALE    = 0.25   # résolution interne SIFT (toujours 0.25, rapide)
L             = 4      # niveaux pyramide
N             = 9      # fenêtre temporelle
TEST_INDICES  = [15, 75, 130]


# ── Chargement ──────────────────────────────────────────────────────────────

def load_frames_dual(video_path, sift_scale, out_scale):
    """
    Charge toutes les frames en deux résolutions :
      small : résolution pour SIFT (sift_scale)
      full  : résolution de sortie (out_scale)
    Retourne (frames_small, frames_full, fps)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    small, full = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small.append(cv2.resize(frame, (0,0), fx=sift_scale, fy=sift_scale))
        if abs(out_scale - sift_scale) < 1e-4:
            full.append(small[-1])
        else:
            full.append(cv2.resize(frame, (0,0), fx=out_scale, fy=out_scale))
    cap.release()
    return small, full, fps


# ── Alignement ──────────────────────────────────────────────────────────────

def compute_homography_small(ref_small, neigh_small):
    """Calcule H en coordonnées small. Retourne H ou None."""
    sift = cv2.SIFT_create(nfeatures=2000)
    g1 = cv2.cvtColor(ref_small,   cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(neigh_small, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None
    flann   = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good    = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good) < 10:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
    return H


def scale_homography(H_small, sift_scale, out_scale):
    """
    Convertit une homographie calculée en coordonnées `sift_scale`
    vers des coordonnées `out_scale`.
    H_full = S_out_inv · H_small · S_sift
    où S = diag(scale, scale, 1)
    """
    if H_small is None:
        return None
    ratio = sift_scale / out_scale   # = sift_scale / out_scale
    S_sift = np.diag([sift_scale, sift_scale, 1.0])
    S_out_inv = np.diag([1.0/out_scale, 1.0/out_scale, 1.0])
    H_full = S_out_inv @ H_small @ S_sift
    return H_full


def warp_full(frame_full, H_full):
    """Applique H_full à frame_full, retourne la frame warpée."""
    if H_full is None:
        return frame_full
    h, w = frame_full.shape[:2]
    return cv2.warpPerspective(frame_full, H_full, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)


# ── Pyramide Laplace ─────────────────────────────────────────────────────────

def build_laplacian_pyramid(img_f, levels):
    gauss = [img_f.copy()]
    for _ in range(levels):
        gauss.append(cv2.pyrDown(gauss[-1]))
    pyramid = []
    for k in range(levels):
        h, w = gauss[k].shape[:2]
        up = cv2.pyrUp(gauss[k+1], dstsize=(w, h))
        pyramid.append(gauss[k] - up)
    pyramid.append(gauss[-1])
    return pyramid


def reconstruct_from_pyramid(pyramid):
    result = pyramid[-1].copy()
    for k in range(len(pyramid)-2, -1, -1):
        h, w = pyramid[k].shape[:2]
        result = cv2.pyrUp(result, dstsize=(w, h)) + pyramid[k]
    return result


# ── Traitement d'une frame ────────────────────────────────────────────────────

def process_frame_J_full(frames_small, frames_full, center_idx, half,
                         sift_scale, out_scale, levels=L):
    start = max(0, center_idx - half)
    end   = min(len(frames_small), center_idx + half + 1)
    ref_s = frames_small[center_idx]
    ref_f = frames_full[center_idx]

    # Aligner les voisins (homographie calculée à sift_scale, appliquée à out_scale)
    aligned_full = [ref_f]
    for i in range(start, end):
        if i == center_idx:
            continue
        H_small = compute_homography_small(ref_s, frames_small[i])
        H_full  = scale_homography(H_small, sift_scale, out_scale)
        aligned_full.append(warp_full(frames_full[i], H_full))

    # Pyramide + médiane temporelle
    aligned_f = [f.astype(np.float32) for f in aligned_full]
    pyramids  = [build_laplacian_pyramid(f, levels) for f in aligned_f]
    ref_pyr   = pyramids[0]

    result_pyramid = []
    for lvl in range(levels + 1):
        stack = np.stack([p[lvl] for p in pyramids], axis=0)
        med   = np.median(stack, axis=0)

        if lvl == 0:
            # Suppression sélective de l'excès positif dans le niveau fin
            fine_orig = ref_pyr[0].astype(np.float32)
            fine_med  = med.astype(np.float32)
            excess    = np.maximum(fine_orig - fine_med, 0)

            exc_flat = excess.ravel()
            pos_only = exc_flat[exc_flat > 0]
            thresh   = float(np.percentile(pos_only, 98)) if len(pos_only) > 100 else 50.0
            progress = np.clip((excess - thresh) / (thresh + 1e-6), 0, 1)
            result_pyramid.append(fine_orig - progress * excess)
        else:
            result_pyramid.append(med)

    return np.clip(reconstruct_from_pyramid(result_pyramid), 0, 255).astype(np.uint8)


# ── Métriques ───────────────────────────────────────────────────────────────

def laplacian_variance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale',     type=float, default=1.0,
                        help='Résolution de sortie (1.0=4K, 0.5=FHD, 0.25=quarter)')
    parser.add_argument('--N',         type=int,   default=9,
                        help='Taille de la fenêtre temporelle')
    parser.add_argument('--test-only', action='store_true',
                        help='Traiter seulement les 3 frames témoins')
    args = parser.parse_args()

    out_scale  = args.scale
    n_frames_t = args.N
    half       = n_frames_t // 2

    # Résolution de sortie
    cap = cv2.VideoCapture(VIDEO_PATH)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    out_w = int(src_w * out_scale)
    out_h = int(src_h * out_scale)

    print(f"=== Approche J 4K — scale={out_scale} ({out_w}×{out_h}), N={n_frames_t}, L={L} ===")
    print(f"  SIFT internal scale : {SIFT_SCALE} ({int(src_w*SIFT_SCALE)}×{int(src_h*SIFT_SCALE)})")
    print(f"  Mode: {'test-only' if args.test_only else 'vidéo complète'}")

    t0 = time.time()
    print("Chargement…")
    frames_small, frames_full, fps = load_frames_dual(VIDEO_PATH, SIFT_SCALE, out_scale)
    print(f"  {len(frames_full)} frames en {time.time()-t0:.1f}s")

    # ── Test frames ──────────────────────────────────────────────────────────
    scale_tag = f"{out_w}x{out_h}"
    rows = []
    print(f"\n{'Frame':>6}  {'Original':>10}  {'J_result':>10}  {'Delta%':>8}  {'Time':>6}")
    print("-" * 50)
    for idx in TEST_INDICES:
        t1 = time.time()
        result = process_frame_J_full(frames_small, frames_full, idx, half,
                                      SIFT_SCALE, out_scale)
        dt = time.time() - t1

        so = laplacian_variance(frames_full[idx])
        sr = laplacian_variance(result)
        delta = (sr - so) / so * 100
        print(f"{idx:>6}  {so:>10.0f}  {sr:>10.0f}  {delta:>+7.1f}%  {dt:>5.1f}s")

        orig = frames_full[idx].copy()
        cv2.putText(orig,   f'Avant {idx} ({scale_tag})',
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)
        cv2.putText(result, f'J sélectif ({scale_tag})',
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        # Downscale for comparison image to keep file size manageable
        ds = min(1.0, 1920 / out_w)
        rows.append(np.hstack([
            cv2.resize(orig,   (0,0), fx=ds, fy=ds),
            cv2.resize(result, (0,0), fx=ds, fy=ds),
        ]))

    comp_path = os.path.join(OUTPUT_DIR, f'comparaison_J_{scale_tag}.jpg')
    cv2.imwrite(comp_path, np.vstack(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    print(f"\nComparaison sauvegardée : {comp_path}")

    if args.test_only:
        print("Mode test-only : terminé.")
        return

    # ── Vidéo complète ───────────────────────────────────────────────────────
    vid_path = os.path.join(OUTPUT_DIR, f'result_J_{scale_tag}_N{n_frames_t}.mp4')
    print(f"\nTraitement vidéo complète ({len(frames_full)} frames → {vid_path})…")
    t0 = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(vid_path, fourcc, fps, (out_w, out_h))

    for idx in range(len(frames_full)):
        result = process_frame_J_full(frames_small, frames_full, idx, half,
                                      SIFT_SCALE, out_scale)
        writer.write(result)
        if idx % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(frames_full) - idx - 1)
            print(f"  Frame {idx:>4}/{len(frames_full)}  ETA: {eta:.0f}s")

    writer.release()
    total = time.time() - t0
    print(f"Vidéo sauvegardée : {vid_path}  ({total:.0f}s, {total/len(frames_full):.1f}s/frame)")


if __name__ == '__main__':
    main()
