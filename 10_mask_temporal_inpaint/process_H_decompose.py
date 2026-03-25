"""
Approche H — Décomposition base/détail + médiane temporelle sélective

Idée clé : les caustiques sont un phénomène d'ILLUMINATION (basse fréquence
spatiale). La texture du fond (coraux, sable) est haute fréquence spatiale.

1. Décomposer chaque frame : base = GaussianBlur(image) → illumination + caustiques
                              detail = image - base → texture
2. Aligner la couche BASE des voisins (homography) → médiane temporelle sur BASE seulement
3. Garder la couche DETAIL de la frame originale (toute la netteté)
4. Recombiner : base_filtrée + detail_original

Les erreurs d'alignement sont invisibles sur la base (elle est lisse).
La texture reste parfaitement nette car jamais touchée.
"""
import cv2
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor

VIDEO_PATH = os.path.join(os.path.dirname(__file__), '..', '01_extraction', 'subclip_5s.mp4')
OUTPUT_DIR = os.path.dirname(__file__)
SCALE = 0.25
N = 9           # window size
BASE_SIGMA = 25  # sigma du flou gaussien pour séparer base/detail
TEST_INDICES = [15, 75, 130]


def load_frames(video_path, scale):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (0, 0), fx=scale, fy=scale))
    cap.release()
    return frames, fps


def decompose(frame, sigma=BASE_SIGMA):
    """Decompose frame into base (illumination) and detail (texture)."""
    frame_f = frame.astype(np.float32)
    base = cv2.GaussianBlur(frame_f, (0, 0), sigma)
    detail = frame_f - base
    return base, detail


def align_homography(ref, neighbor):
    """Align neighbor to ref using SIFT + homography RANSAC."""
    sift = cv2.SIFT_create(nfeatures=2000)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    neigh_gray = cv2.cvtColor(neighbor, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(ref_gray, None)
    kp2, des2 = sift.detectAndCompute(neigh_gray, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return neighbor
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good) < 10:
        return neighbor
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        return neighbor
    h, w = ref.shape[:2]
    return cv2.warpPerspective(neighbor, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def process_frame_decompose(frames, center_idx, half, sigma=BASE_SIGMA):
    """
    Process one frame using base/detail decomposition + temporal median on base only.
    """
    start = max(0, center_idx - half)
    end = min(len(frames), center_idx + half + 1)
    ref = frames[center_idx]

    # Decompose center frame
    base_ref, detail_ref = decompose(ref, sigma)

    # Align neighbors and extract their base layers
    bases_aligned = [base_ref]
    for i in range(start, end):
        if i == center_idx:
            continue
        aligned = align_homography(ref, frames[i])
        base_neigh, _ = decompose(aligned, sigma)
        bases_aligned.append(base_neigh)

    # Temporal median on base layers only
    base_stack = np.stack(bases_aligned, axis=0)
    base_median = np.median(base_stack, axis=0)

    # Recombine: filtered base + original detail
    result = base_median + detail_ref
    return np.clip(result, 0, 255).astype(np.uint8)


def laplacian_variance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    print(f"=== Approach H: Base/Detail Decomposition + Temporal Median ===")
    print(f"Parameters: N={N}, base_sigma={BASE_SIGMA}, scale={SCALE}")
    t0 = time.time()
    frames, fps = load_frames(VIDEO_PATH, SCALE)
    print(f"Loaded {len(frames)} frames in {time.time()-t0:.1f}s")

    half = N // 2

    # Test frames
    rows = []
    print(f"\n{'Frame':>8} {'Sharp_orig':>12} {'Sharp_result':>14} {'Delta%':>10}")
    for idx in TEST_INDICES:
        t1 = time.time()
        result = process_frame_decompose(frames, idx, half)
        dt = time.time() - t1

        sharp_orig = laplacian_variance(frames[idx])
        sharp_result = laplacian_variance(result)
        delta = (sharp_result - sharp_orig) / sharp_orig * 100
        print(f"{idx:>8} {sharp_orig:>12.1f} {sharp_result:>14.1f} {delta:>+9.1f}%  ({dt:.1f}s)")

        orig = frames[idx].copy()
        cv2.putText(orig, f'Avant {idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(result, f'Apres H (s={BASE_SIGMA})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        rows.append(np.hstack([orig, result]))

    comparison = np.vstack(rows)
    out_path = os.path.join(OUTPUT_DIR, 'comparaison_H_decompose.jpg')
    cv2.imwrite(out_path, comparison)
    print(f"\nSaved {out_path}")

    # Full video
    print(f"\nProcessing full video ({len(frames)} frames)...")
    t0 = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frames[0].shape[:2]
    vid_path = os.path.join(OUTPUT_DIR, 'result_H_decompose.mp4')
    out = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))

    for idx in range(len(frames)):
        result = process_frame_decompose(frames, idx, half)
        out.write(result)
        if idx % 10 == 0:
            print(f"  Frame {idx}/{len(frames)}")

    out.release()
    print(f"Saved {vid_path} in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
