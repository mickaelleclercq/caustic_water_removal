"""
Approche G — Version finale optimisée : RAFT + Masque élargi + N=15
- Masque plus inclusif pour capturer toute la zone caustique
- N=15 frames (0.5s) pour plus de variation temporelle des caustiques
- Flow calculé sur images pré-filtrées
- Remplacement sélectif avec masque doux
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as Fnn
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.transforms.functional import to_tensor
import time
import os

VIDEO_PATH = os.path.join(os.path.dirname(__file__), '..', '01_extraction', 'subclip_5s.mp4')
OUTPUT_DIR = os.path.dirname(__file__)
SCALE = 0.25
N = 15
TEST_INDICES = [15, 75, 130]
DEVICE = 'cuda:0'


def load_frames_cv(video_path, scale):
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


def load_raft(device):
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights).to(device).eval()
    transforms = weights.transforms()
    return model, transforms


def cv_to_tensor(img, device):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return to_tensor(rgb).unsqueeze(0).to(device)


def pad_to_8(t):
    _, _, h, w = t.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        t = Fnn.pad(t, (0, pad_w, 0, pad_h), mode='reflect')
    return t, h, w


def compute_flow_raft(model, transforms, img1_cv, img2_cv, device):
    # Pre-filter for robust flow (reduce caustic influence)
    img1_f = cv2.GaussianBlur(img1_cv, (9, 9), 3.0)
    img2_f = cv2.GaussianBlur(img2_cv, (9, 9), 3.0)
    img1_t = cv_to_tensor(img1_f, device)
    img2_t = cv_to_tensor(img2_f, device)
    img1_t, img2_t = transforms(img1_t, img2_t)
    img1_p, oh, ow = pad_to_8(img1_t)
    img2_p, _, _ = pad_to_8(img2_t)
    with torch.no_grad():
        flows = model(img1_p, img2_p)
    return flows[-1][:, :, :oh, :ow]


def warp_with_flow(img_tensor, flow):
    _, _, h, w = img_tensor.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=torch.float32),
        torch.arange(w, device=flow.device, dtype=torch.float32),
        indexing='ij'
    )
    map_x = (grid_x + flow[0, 0]) / (w - 1) * 2 - 1
    map_y = (grid_y + flow[0, 1]) / (h - 1) * 2 - 1
    grid = torch.stack([map_x, map_y], dim=-1).unsqueeze(0)
    return Fnn.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='reflection', align_corners=True)


def tensor_to_cv(t):
    img = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def detect_caustics_mask(frame, kernel_size=21, threshold_factor=1.2):
    """
    Detect caustics with an inclusive mask:
    - Top-hat on V channel
    - Generous threshold to capture full caustic area
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)

    # Multi-scale top-hat for different caustic sizes
    mask_combined = np.zeros_like(v)
    for ks in [11, 21, 35]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, kernel)
        mean_t = tophat.mean()
        std_t = tophat.std()
        mask_combined = np.maximum(mask_combined, tophat)

    # Adaptive threshold
    mean_val = mask_combined.mean()
    std_val = mask_combined.std()
    thresh = mean_val + threshold_factor * std_val
    mask = (mask_combined > thresh).astype(np.uint8) * 255

    # Close small gaps and dilate
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, dilate_k, iterations=1)

    return mask


def process_frame(frames_cv, center_idx, half, model, transforms, device):
    start = max(0, center_idx - half)
    end = min(len(frames_cv), center_idx + half + 1)
    ref = frames_cv[center_idx]

    mask = detect_caustics_mask(ref)
    caustic_ratio = mask.sum() / (mask.size * 255)

    if caustic_ratio < 0.003:
        return ref, mask

    # Align neighbors using RAFT — skip frames for speed (every 2nd frame)
    ref_t = cv_to_tensor(ref, device)
    aligned_cv = [ref]

    indices = list(range(start, end))
    # For large windows, subsample to keep speed manageable
    if len(indices) > 9:
        # Keep every 2nd frame but always include center
        step = 2
        selected = [i for i in indices if (i - center_idx) % step == 0 or i == center_idx]
    else:
        selected = indices

    for i in selected:
        if i == center_idx:
            continue
        neigh = frames_cv[i]
        flow = compute_flow_raft(model, transforms, neigh, ref, device)
        neigh_t = cv_to_tensor(neigh, device)
        warped_t = warp_with_flow(neigh_t, flow)
        aligned_cv.append(tensor_to_cv(warped_t))

    # Temporal median of aligned frames
    stack = np.stack(aligned_cv, axis=0).astype(np.float32)
    median = np.median(stack, axis=0).astype(np.uint8)

    # Soft selective replacement
    mask_soft = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 6.0)
    mask_soft = mask_soft[:, :, None] / 255.0

    result = ref.astype(np.float32) * (1 - mask_soft) + median.astype(np.float32) * mask_soft
    return np.clip(result, 0, 255).astype(np.uint8), mask


def laplacian_variance(img):
    """Measure image sharpness via Laplacian variance."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    print("=== Approach G: RAFT + Inclusive Mask + N=15 (Final) ===")
    print(f"Device: {DEVICE}")
    t0 = time.time()
    frames_cv, fps = load_frames_cv(VIDEO_PATH, SCALE)
    print(f"Loaded {len(frames_cv)} frames in {time.time()-t0:.1f}s")

    model, transforms = load_raft(DEVICE)
    print("RAFT ready.")

    half = N // 2

    # Test frames + sharpness measurement
    rows = []
    print("\nSharpness comparison (Laplacian variance, higher=sharper):")
    print(f"{'Frame':>8} {'Original':>12} {'Result':>12} {'Delta%':>10} {'Caustic%':>10}")
    for idx in TEST_INDICES:
        t1 = time.time()
        result, mask = process_frame(frames_cv, idx, half, model, transforms, DEVICE)
        dt = time.time() - t1
        caustic_pct = mask.sum() / (mask.size * 255) * 100

        sharp_orig = laplacian_variance(frames_cv[idx])
        sharp_result = laplacian_variance(result)
        delta = (sharp_result - sharp_orig) / sharp_orig * 100
        print(f"{idx:>8} {sharp_orig:>12.1f} {sharp_result:>12.1f} {delta:>+9.1f}% {caustic_pct:>9.1f}%")

        orig = frames_cv[idx].copy()
        mask_vis = np.zeros_like(orig)
        mask_vis[:, :, 2] = mask
        mask_overlay = cv2.addWeighted(orig, 0.7, mask_vis, 0.3, 0)

        cv2.putText(orig, f'Avant {idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(mask_overlay, f'Masque ({caustic_pct:.0f}%)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(result, f'Apres G (N={N})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        rows.append(np.hstack([orig, mask_overlay, result]))

    comparison = np.vstack(rows)
    out_path = os.path.join(OUTPUT_DIR, 'comparaison_G_final.jpg')
    cv2.imwrite(out_path, comparison)
    print(f"\nSaved {out_path}")

    # Full video
    print(f"\nProcessing full video ({len(frames_cv)} frames)...")
    t0 = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frames_cv[0].shape[:2]
    vid_path = os.path.join(OUTPUT_DIR, 'result_G_final_N15.mp4')
    out = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))

    for idx in range(len(frames_cv)):
        result, _ = process_frame(frames_cv, idx, half, model, transforms, DEVICE)
        out.write(result)
        if idx % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Frame {idx}/{len(frames_cv)} ({elapsed:.0f}s)")

    out.release()
    print(f"Saved {vid_path} in {time.time()-t0:.1f}s")
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
