"""
Approche D — Filtre passe-bas temporel après alignement par homographie
Les caustiques scintillent (haute fréquence temporelle).
Le fond est stable (basse fréquence temporelle).
On stabilise d'abord par homographie, puis on applique un filtre gaussien
temporel par pixel. Les caustiques sont lissées, le fond reste.
"""
import cv2
import numpy as np
import time
import os

VIDEO_PATH = os.path.join(os.path.dirname(__file__), '..', '01_extraction', 'subclip_5s.mp4')
OUTPUT_DIR = os.path.dirname(__file__)
SCALE = 0.25
SIGMA_T = 2.0   # sigma du filtre temporel gaussien (en frames)
WINDOW = 9       # taille de la fenêtre temporelle (en frames)
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
    return cv2.warpPerspective(neighbor, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def gaussian_weights(size, sigma):
    """Compute 1D Gaussian kernel."""
    x = np.arange(size) - size // 2
    w = np.exp(-0.5 * (x / sigma) ** 2)
    return w / w.sum()


def process_window_lowpass(frames, center_idx, half, weights_full):
    start = max(0, center_idx - half)
    end = min(len(frames), center_idx + half + 1)
    ref = frames[center_idx]

    aligned = []
    local_weights = []
    for i in range(start, end):
        offset = i - center_idx + half  # index into weights_full
        if i == center_idx:
            aligned.append(ref.astype(np.float32))
        else:
            aligned.append(align_homography(ref, frames[i]).astype(np.float32))
        local_weights.append(weights_full[offset])

    # Weighted average (Gaussian temporal lowpass)
    w = np.array(local_weights, dtype=np.float32)
    w /= w.sum()

    result = np.zeros_like(aligned[0])
    for frame, weight in zip(aligned, w):
        result += frame * weight

    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    print(f"=== Approach D: Temporal Lowpass (Gaussian sigma={SIGMA_T}, window={WINDOW}) ===")
    print(f"Loading video at scale {SCALE}...")
    t0 = time.time()
    frames, fps = load_frames(VIDEO_PATH, SCALE)
    print(f"Loaded {len(frames)} frames in {time.time()-t0:.1f}s")

    half = WINDOW // 2
    weights = gaussian_weights(WINDOW, SIGMA_T)
    print(f"Temporal Gaussian weights: {np.round(weights, 3)}")

    # Test frames
    rows = []
    for idx in TEST_INDICES:
        print(f"  Processing frame {idx}...")
        t1 = time.time()
        result = process_window_lowpass(frames, idx, half, weights)
        dt = time.time() - t1
        print(f"    Done in {dt:.1f}s")

        orig = frames[idx].copy()
        cv2.putText(orig, f'Avant {idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(result, f'Apres Lowpass s={SIGMA_T}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        rows.append(np.hstack([orig, result]))

    comparison = np.vstack(rows)
    out_path = os.path.join(OUTPUT_DIR, 'comparaison_lowpass.jpg')
    cv2.imwrite(out_path, comparison)
    print(f"Saved {out_path}")

    # Full video
    print(f"\nProcessing full video ({len(frames)} frames)...")
    t0 = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frames[0].shape[:2]
    vid_path = os.path.join(OUTPUT_DIR, 'result_lowpass.mp4')
    out = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))

    for idx in range(len(frames)):
        result = process_window_lowpass(frames, idx, half, weights)
        out.write(result)
        if idx % 10 == 0:
            print(f"  Frame {idx}/{len(frames)}")

    out.release()
    print(f"Saved {vid_path} in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
