"""
Approche A — Homographie RANSAC + Médiane glissante
Aligne les frames voisines sur la frame centrale via SIFT + homographie RANSAC,
puis prend la médiane. L'homographie (8 DoF) est meilleure que l'ECC euclidien
pour compenser un mouvement de caméra qui avance (parallaxe).
"""
import cv2
import numpy as np
import time
import sys
import os

VIDEO_PATH = os.path.join(os.path.dirname(__file__), '..', '01_extraction', 'subclip_5s.mp4')
OUTPUT_DIR = os.path.dirname(__file__)
SCALE = 0.25  # 4K -> 960x540
N = 5         # window size
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
    """Align neighbor to ref using SIFT + homography RANSAC."""
    sift = cv2.SIFT_create(nfeatures=2000)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    neigh_gray = cv2.cvtColor(neighbor, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(ref_gray, None)
    kp2, des2 = sift.detectAndCompute(neigh_gray, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return neighbor  # fallback

    # FLANN matcher
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good) < 10:
        return neighbor  # fallback

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        return neighbor

    h, w = ref.shape[:2]
    warped = cv2.warpPerspective(neighbor, H, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
    return warped


def process_window(frames, center_idx, half):
    """Process one window: align neighbors and take median."""
    start = max(0, center_idx - half)
    end = min(len(frames), center_idx + half + 1)
    ref = frames[center_idx]

    aligned = []
    for i in range(start, end):
        if i == center_idx:
            aligned.append(ref)
        else:
            aligned.append(align_homography(ref, frames[i]))

    stack = np.stack(aligned, axis=0).astype(np.float32)
    return np.median(stack, axis=0).astype(np.uint8)


def main():
    print("=== Approach A: Homography RANSAC + Sliding Median ===")
    print(f"Loading video at scale {SCALE}...")
    t0 = time.time()
    frames, fps = load_frames(VIDEO_PATH, SCALE)
    print(f"Loaded {len(frames)} frames ({frames[0].shape[1]}x{frames[0].shape[0]}) in {time.time()-t0:.1f}s")

    half = N // 2

    # Test on specific frames
    rows = []
    for idx in TEST_INDICES:
        print(f"  Processing frame {idx}...")
        t1 = time.time()
        result = process_window(frames, idx, half)
        dt = time.time() - t1
        print(f"    Done in {dt:.1f}s")

        orig = frames[idx].copy()
        cv2.putText(orig, f'Avant {idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(result, f'Apres Homography N={N}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        rows.append(np.hstack([orig, result]))

    comparison = np.vstack(rows)
    out_path = os.path.join(OUTPUT_DIR, 'comparaison_homography.jpg')
    cv2.imwrite(out_path, comparison)
    print(f"Saved {out_path}")

    # Full video processing
    print(f"\nProcessing full video ({len(frames)} frames)...")
    t0 = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frames[0].shape[:2]
    vid_path = os.path.join(OUTPUT_DIR, 'result_homography_N5.mp4')
    out = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))

    for idx in range(len(frames)):
        result = process_window(frames, idx, half)
        out.write(result)
        if idx % 10 == 0:
            print(f"  Frame {idx}/{len(frames)}")

    out.release()
    print(f"Saved {vid_path} in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
