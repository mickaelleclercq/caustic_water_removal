"""
Approche F — RAFT + Masque sélectif + Remplacement temporel (version optimisée)

Améliorations par rapport aux approches précédentes :
1. Pré-filtrage avant calcul du flow (blur les caustiques pour ne pas biaiser le flow)
2. RAFT optical flow sur GPU pour l'alignement le plus précis
3. Masque de caustiques plus précis (multi-critère : top-hat + gradient + seuil adaptatif)
4. Remplacement sélectif : seuls les pixels caustiques sont remplacés par la médiane
5. Le fond non-caustique est conservé parfaitement intact
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.transforms.functional import to_tensor
import time
import os

VIDEO_PATH = os.path.join(os.path.dirname(__file__), '..', '01_extraction', 'subclip_5s.mp4')
OUTPUT_DIR = os.path.dirname(__file__)
SCALE = 0.25
N = 7
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
    t = to_tensor(rgb).unsqueeze(0).to(device)
    return t


def pad_to_8(t):
    _, _, h, w = t.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode='reflect')
    return t, h, w


def prefilter_for_flow(img, ksize=7):
    """Pre-blur image to reduce caustic influence on optical flow estimation."""
    return cv2.GaussianBlur(img, (ksize, ksize), 2.0)


def compute_flow_raft(model, transforms, img1_cv, img2_cv, device):
    """Compute flow from img1 to img2, using pre-filtered images for flow estimation."""
    # Pre-filter to reduce caustic bias in flow
    img1_f = prefilter_for_flow(img1_cv)
    img2_f = prefilter_for_flow(img2_cv)

    img1_t = cv_to_tensor(img1_f, device)
    img2_t = cv_to_tensor(img2_f, device)

    img1_t, img2_t = transforms(img1_t, img2_t)
    img1_p, oh, ow = pad_to_8(img1_t)
    img2_p, _, _ = pad_to_8(img2_t)

    with torch.no_grad():
        flows = model(img1_p, img2_p)
    flow = flows[-1][:, :, :oh, :ow]
    return flow


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
    warped = F.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
    return warped


def tensor_to_cv(t):
    img = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def detect_caustics_mask(frame, kernel_size=21, threshold_factor=2.0):
    """
    Multi-criteria caustic detection:
    1. Top-hat on V channel (bright spots)
    2. High local variance (caustics create strong local contrast)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)

    # Top-hat: bright spots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, kernel)

    # Threshold: adaptive based on statistics
    mean_v = tophat.mean()
    std_v = tophat.std()
    thresh = mean_v + threshold_factor * std_v
    mask_tophat = (tophat > thresh).astype(np.uint8) * 255

    # Additional: also detect very bright pixels (absolute threshold)
    bright_mask = (v > np.percentile(v, 92)).astype(np.uint8) * 255

    # Combine: pixel is caustic if both top-hat AND bright
    mask = cv2.bitwise_and(mask_tophat, bright_mask)

    # Light morphological cleanup
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    # Small dilation to grab edges
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    return mask


def process_frame(frames_cv, center_idx, half, model, transforms, device):
    start = max(0, center_idx - half)
    end = min(len(frames_cv), center_idx + half + 1)
    ref = frames_cv[center_idx]

    # Detect caustics
    mask = detect_caustics_mask(ref)
    caustic_ratio = mask.sum() / (mask.size * 255)

    if caustic_ratio < 0.003:
        return ref, mask

    # Align neighbors using RAFT
    ref_t = cv_to_tensor(ref, device)
    aligned_cv = [ref]

    for i in range(start, end):
        if i == center_idx:
            continue
        neigh = frames_cv[i]
        flow = compute_flow_raft(model, transforms, neigh, ref, device)
        neigh_t = cv_to_tensor(neigh, device)
        warped_t = warp_with_flow(neigh_t, flow)
        aligned_cv.append(tensor_to_cv(warped_t))

    # Temporal median
    stack = np.stack(aligned_cv, axis=0).astype(np.float32)
    median = np.median(stack, axis=0).astype(np.uint8)

    # Selective replacement with soft mask
    mask_soft = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 4.0)
    mask_soft = mask_soft[:, :, None] / 255.0

    result = ref.astype(np.float32) * (1 - mask_soft) + median.astype(np.float32) * mask_soft
    return np.clip(result, 0, 255).astype(np.uint8), mask


def main():
    print("=== Approach F: RAFT + Selective Caustic Replacement ===")
    print(f"Device: {DEVICE}")
    t0 = time.time()
    frames_cv, fps = load_frames_cv(VIDEO_PATH, SCALE)
    print(f"Loaded {len(frames_cv)} frames in {time.time()-t0:.1f}s")

    model, transforms = load_raft(DEVICE)
    print("RAFT ready.")

    half = N // 2

    # Test frames
    rows = []
    for idx in TEST_INDICES:
        print(f"  Processing frame {idx}...")
        t1 = time.time()
        result, mask = process_frame(frames_cv, idx, half, model, transforms, DEVICE)
        dt = time.time() - t1
        caustic_pct = mask.sum() / (mask.size * 255) * 100
        print(f"    Done in {dt:.1f}s — caustic coverage: {caustic_pct:.1f}%")

        orig = frames_cv[idx].copy()

        # Mask visualization
        mask_vis = np.zeros_like(orig)
        mask_vis[:, :, 2] = mask  # Red channel
        mask_overlay = cv2.addWeighted(orig, 0.7, mask_vis, 0.3, 0)

        cv2.putText(orig, f'Avant {idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(mask_overlay, f'Masque ({caustic_pct:.0f}%)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(result, f'Apres RAFT+Mask', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        rows.append(np.hstack([orig, mask_overlay, result]))

    comparison = np.vstack(rows)
    out_path = os.path.join(OUTPUT_DIR, 'comparaison_raft_mask.jpg')
    cv2.imwrite(out_path, comparison)
    print(f"Saved {out_path}")

    # Full video
    print(f"\nProcessing full video ({len(frames_cv)} frames)...")
    t0 = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frames_cv[0].shape[:2]
    vid_path = os.path.join(OUTPUT_DIR, 'result_raft_mask_N7.mp4')
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
