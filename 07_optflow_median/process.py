"""
Approche B — Optical Flow Farnebäck dense + Médiane glissante
Utilise le flux optique dense de Farnebäck pour warper chaque voisin pixel-par-pixel
sur la frame centrale, puis prend la médiane.
Le flow dense compense les déformations locales (parallaxe 3D).
"""
import cv2
import numpy as np
import time
import os

VIDEO_PATH = os.path.join(os.path.dirname(__file__), '..', '01_extraction', 'subclip_5s.mp4')
OUTPUT_DIR = os.path.dirname(__file__)
SCALE = 0.25
N = 5
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


def warp_flow(img, flow):
    """Warp image using dense optical flow field."""
    h, w = flow.shape[:2]
    map_x = np.arange(w, dtype=np.float32)[None, :] + flow[:, :, 0]
    map_y = np.arange(h, dtype=np.float32)[:, None] + flow[:, :, 1]
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def align_optflow(ref, neighbor):
    """Align neighbor to ref using Farnebäck dense optical flow."""
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    neigh_gray = cv2.cvtColor(neighbor, cv2.COLOR_BGR2GRAY)
    # Flow from neighbor to ref: where should each pixel in neighbor go to match ref
    flow = cv2.calcOpticalFlowFarneback(
        neigh_gray, ref_gray,
        None,
        pyr_scale=0.5, levels=5, winsize=15,
        iterations=5, poly_n=7, poly_sigma=1.5,
        flags=0
    )
    return warp_flow(neighbor, flow)


def process_window(frames, center_idx, half):
    start = max(0, center_idx - half)
    end = min(len(frames), center_idx + half + 1)
    ref = frames[center_idx]

    aligned = []
    for i in range(start, end):
        if i == center_idx:
            aligned.append(ref)
        else:
            aligned.append(align_optflow(ref, frames[i]))

    stack = np.stack(aligned, axis=0).astype(np.float32)
    return np.median(stack, axis=0).astype(np.uint8)


def main():
    print("=== Approach B: Farneback Optical Flow + Sliding Median ===")
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
        cv2.putText(result, f'Apres OptFlow N={N}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        rows.append(np.hstack([orig, result]))

    comparison = np.vstack(rows)
    out_path = os.path.join(OUTPUT_DIR, 'comparaison_optflow.jpg')
    cv2.imwrite(out_path, comparison)
    print(f"Saved {out_path}")

    # Full video
    print(f"\nProcessing full video ({len(frames)} frames)...")
    t0 = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frames[0].shape[:2]
    vid_path = os.path.join(OUTPUT_DIR, 'result_optflow_N5.mp4')
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
