"""
Sweep paramétrique rapide — Approche I (Pyramide Laplacienne)
Teste les combinaisons (keep_fine, levels, N) sur 3 frames
et génère une grille de comparaison visuelle.
"""
import cv2
import numpy as np
import time
import os

VIDEO_PATH = os.path.join(os.path.dirname(__file__), '..', '01_extraction', 'subclip_5s.mp4')
OUTPUT_DIR = os.path.dirname(__file__)
SCALE = 0.25
TEST_INDICES = [15, 75, 130]

# Combinaisons à tester : (keep_fine, levels, N)
CONFIGS = [
    (1, 4,  9,  'I_base'),
    (0, 4,  9,  'I_keep0'),
    (1, 5,  9,  'I_L5'),
    (0, 5,  9,  'I_L5_k0'),
    (1, 4, 13,  'I_N13'),
]


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


def build_laplacian_pyramid(img_f, levels):
    gauss = [img_f.copy()]
    for _ in range(levels):
        gauss.append(cv2.pyrDown(gauss[-1]))
    pyramid = []
    for k in range(levels):
        h, w = gauss[k].shape[:2]
        up = cv2.pyrUp(gauss[k + 1], dstsize=(w, h))
        pyramid.append(gauss[k] - up)
    pyramid.append(gauss[-1])
    return pyramid


def reconstruct_from_pyramid(pyramid):
    result = pyramid[-1].copy()
    for k in range(len(pyramid) - 2, -1, -1):
        h, w = pyramid[k].shape[:2]
        result = cv2.pyrUp(result, dstsize=(w, h)) + pyramid[k]
    return result


def align_homography(ref, neighbor):
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
    return cv2.warpPerspective(neighbor, H, (w, h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def process_frame_pyramid(frames, center_idx, half, levels, keep_fine):
    start = max(0, center_idx - half)
    end = min(len(frames), center_idx + half + 1)
    ref = frames[center_idx]

    aligned = [ref]
    for i in range(start, end):
        if i == center_idx:
            continue
        aligned.append(align_homography(ref, frames[i]))

    aligned_f = [f.astype(np.float32) for f in aligned]
    pyramids = [build_laplacian_pyramid(f, levels) for f in aligned_f]

    ref_pyramid = pyramids[0]
    result_pyramid = []
    for lvl in range(levels + 1):
        if lvl < keep_fine:
            result_pyramid.append(ref_pyramid[lvl])
        else:
            stack = np.stack([p[lvl] for p in pyramids], axis=0)
            result_pyramid.append(np.median(stack, axis=0))

    return np.clip(reconstruct_from_pyramid(result_pyramid), 0, 255).astype(np.uint8)


def laplacian_variance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    print("Chargement des frames…")
    frames, _ = load_frames(VIDEO_PATH, SCALE)
    print(f"{len(frames)} frames.")

    # Pré-calculer les alignements (réutilisés pour tous les configs)
    results = {}  # (config_label, frame_idx) → img
    LABEL_H = 36

    print(f"\n{'Config':<12}" + "".join(f"{fi:>18}" for fi in TEST_INDICES))
    print("-" * (12 + 18 * len(TEST_INDICES)))

    orig_sharps = {fi: laplacian_variance(frames[fi]) for fi in TEST_INDICES}

    for keep_fine, levels, N, label in CONFIGS:
        half = N // 2
        row_str = f"{label:<12}"
        for fi in TEST_INDICES:
            t0 = time.time()
            img = process_frame_pyramid(frames, fi, half, levels, keep_fine)
            dt = time.time() - t0
            s = laplacian_variance(img)
            d = (s - orig_sharps[fi]) / orig_sharps[fi] * 100
            row_str += f"  {s:>7.0f}({d:+.1f}%)  "
            results[(label, fi)] = img
        print(row_str)

    # ── Grille visuelle ────────────────────────────────────────────────────
    h, w = frames[TEST_INDICES[0]].shape[:2]

    def add_bar(img, text, color=(0, 255, 0)):
        bar = np.zeros((LABEL_H, w, 3), dtype=np.uint8)
        cv2.putText(bar, text, (6, LABEL_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        return np.vstack([bar, img])

    rows = []
    # Ligne d'en-tête : originaux
    row_orig = []
    for fi in TEST_INDICES:
        cell = frames[fi].copy()
        s = orig_sharps[fi]
        cell = add_bar(cell, f"Original fr{fi}  shp:{s:.0f}", color=(100, 200, 255))
        row_orig.append(cell)
    rows.append(np.hstack(row_orig))

    for keep_fine, levels, N, label in CONFIGS:
        row = []
        for fi in TEST_INDICES:
            img = results[(label, fi)]
            s = laplacian_variance(img)
            d = (s - orig_sharps[fi]) / orig_sharps[fi] * 100
            img = add_bar(img, f"{label} fr{fi}  shp:{d:+.1f}%")
            row.append(img)
        rows.append(np.hstack(row))

    grid = np.vstack(rows)
    out_path = os.path.join(OUTPUT_DIR, 'sweep_pyramid.jpg')
    cv2.imwrite(out_path, grid, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    print(f"\nGrille sauvegardée : {out_path}  ({grid.shape[1]}×{grid.shape[0]})")


if __name__ == '__main__':
    main()
