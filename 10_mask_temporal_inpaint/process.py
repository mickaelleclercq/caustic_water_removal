"""
Approche E — Masque caustiques + Remplacement temporel sélectif
Idée clé : ne PAS toucher les pixels du fond (→ pas de flou) et ne remplacer
QUE les pixels identifiés comme caustiques par la médiane des frames voisines.

Étapes :
1. Détection du masque de caustiques (Top-Hat sur canal V + seuillage adaptatif)
2. Alignement des N frames voisines sur la centrale (homographie RANSAC)
3. Pour les pixels masqués : prendre la MÉDIANE des frames alignées
4. Pour les pixels non-masqués : garder l'original tel quel
→ Netteté maximale du fond, caustiques remplacées par un consensus temporel.
"""
import cv2
import numpy as np
import time
import os

VIDEO_PATH = os.path.join(os.path.dirname(__file__), '..', '01_extraction', 'subclip_5s.mp4')
OUTPUT_DIR = os.path.dirname(__file__)
SCALE = 0.25
N = 7          # plus de frames pour avoir un meilleur consensus sur les caustiques
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


def detect_caustics_mask(frame, kernel_size=15, threshold_factor=1.5):
    """
    Detect caustic pixels using morphological top-hat on value channel.
    Returns a binary mask (255 = caustic, 0 = normal).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)

    # Top-hat: isolate bright spots smaller than kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, kernel)

    # Adaptive threshold: caustic if tophat > mean + threshold_factor * std
    mean_val = tophat.mean()
    std_val = tophat.std()
    thresh = mean_val + threshold_factor * std_val
    mask = (tophat > thresh).astype(np.uint8) * 255

    # Dilate slightly to include edge pixels of caustics
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    return mask


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


def process_frame_selective(frames, center_idx, half):
    """
    Process one frame: detect caustics, align neighbors, replace only masked pixels.
    """
    start = max(0, center_idx - half)
    end = min(len(frames), center_idx + half + 1)
    ref = frames[center_idx]

    # Detect caustics on the center frame
    mask = detect_caustics_mask(ref)

    # If almost no caustics detected, return original
    caustic_ratio = mask.sum() / (mask.size * 255)
    if caustic_ratio < 0.005:
        return ref, mask

    # Align neighbors
    aligned = []
    for i in range(start, end):
        if i == center_idx:
            aligned.append(ref)
        else:
            aligned.append(align_homography(ref, frames[i]))

    # Compute temporal median of aligned frames
    stack = np.stack(aligned, axis=0).astype(np.float32)
    median = np.median(stack, axis=0).astype(np.uint8)

    # Blend: use median only where caustics are detected, keep original elsewhere
    mask_3ch = mask[:, :, None].astype(np.float32) / 255.0

    # Soft mask: Gaussian blur for smooth transition
    mask_soft = cv2.GaussianBlur(mask_3ch, (11, 11), 3)
    if mask_soft.ndim == 2:
        mask_soft = mask_soft[:, :, None]

    result = (ref.astype(np.float32) * (1 - mask_soft) + median.astype(np.float32) * mask_soft)
    return np.clip(result, 0, 255).astype(np.uint8), mask


def main():
    print("=== Approach E: Caustic Mask + Selective Temporal Replace ===")
    print(f"Loading video at scale {SCALE}...")
    t0 = time.time()
    frames, fps = load_frames(VIDEO_PATH, SCALE)
    print(f"Loaded {len(frames)} frames in {time.time()-t0:.1f}s")

    half = N // 2

    # Test frames
    rows = []
    for idx in TEST_INDICES:
        print(f"  Processing frame {idx}...")
        t1 = time.time()
        result, mask = process_frame_selective(frames, idx, half)
        dt = time.time() - t1
        caustic_pct = mask.sum() / (mask.size * 255) * 100
        print(f"    Done in {dt:.1f}s — caustic coverage: {caustic_pct:.1f}%")

        orig = frames[idx].copy()
        # Create mask visualization
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_vis[:, :, 2] = mask  # red channel for caustics
        mask_vis[:, :, 0] = 0
        mask_overlay = cv2.addWeighted(orig, 0.7, mask_vis, 0.3, 0)

        cv2.putText(orig, f'Avant {idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(mask_overlay, f'Masque caustiques', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(result, f'Apres MaskInpaint N={N}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 3-column layout: original | mask | result
        row = np.hstack([orig, mask_overlay, result])
        rows.append(row)

    comparison = np.vstack(rows)
    out_path = os.path.join(OUTPUT_DIR, 'comparaison_mask_inpaint.jpg')
    cv2.imwrite(out_path, comparison)
    print(f"Saved {out_path}")

    # Full video
    print(f"\nProcessing full video ({len(frames)} frames)...")
    t0 = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frames[0].shape[:2]
    vid_path = os.path.join(OUTPUT_DIR, 'result_mask_inpaint_N7.mp4')
    out = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))

    for idx in range(len(frames)):
        result, _ = process_frame_selective(frames, idx, half)
        out.write(result)
        if idx % 10 == 0:
            print(f"  Frame {idx}/{len(frames)}")

    out.release()
    print(f"Saved {vid_path} in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
