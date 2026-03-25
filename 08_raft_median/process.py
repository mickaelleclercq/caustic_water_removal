"""
Approche C — RAFT Optical Flow (GPU) + Médiane glissante
RAFT est un réseau deep learning pour l'optical flow, bien plus précis que Farnebäck.
Utilise les GPUs A100 pour le calcul du flow + warp.
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
N = 5
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
    """Convert BGR uint8 HWC to RGB float32 NCHW tensor on device."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = to_tensor(rgb).unsqueeze(0).to(device)
    return t


def pad_to_8(t):
    """Pad tensor so H and W are divisible by 8."""
    _, _, h, w = t.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode='reflect')
    return t, h, w


def compute_flow_raft(model, transforms, img1_t, img2_t):
    """Compute optical flow from img1 to img2 using RAFT."""
    img1_t, img2_t = transforms(img1_t, img2_t)
    img1_p, oh, ow = pad_to_8(img1_t)
    img2_p, _, _ = pad_to_8(img2_t)
    with torch.no_grad():
        flows = model(img1_p, img2_p)
    flow = flows[-1][:, :, :oh, :ow]  # crop back to original
    return flow


def warp_with_flow(img_tensor, flow):
    """Warp image tensor using flow field (GPU). flow: [1,2,H,W]"""
    _, _, h, w = img_tensor.shape
    # Create base grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=torch.float32),
        torch.arange(w, device=flow.device, dtype=torch.float32),
        indexing='ij'
    )
    # Add flow to grid and normalize to [-1, 1]
    map_x = (grid_x + flow[0, 0]) / (w - 1) * 2 - 1
    map_y = (grid_y + flow[0, 1]) / (h - 1) * 2 - 1
    grid = torch.stack([map_x, map_y], dim=-1).unsqueeze(0)
    warped = F.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
    return warped


def tensor_to_cv(t):
    """Convert NCHW RGB float tensor to BGR uint8 HWC."""
    img = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def process_window_raft(frames_cv, center_idx, half, model, transforms, device):
    start = max(0, center_idx - half)
    end = min(len(frames_cv), center_idx + half + 1)

    ref_t = cv_to_tensor(frames_cv[center_idx], device)
    aligned_tensors = []

    for i in range(start, end):
        if i == center_idx:
            aligned_tensors.append(ref_t)
        else:
            neigh_t = cv_to_tensor(frames_cv[i], device)
            # Flow from neighbor to ref
            flow = compute_flow_raft(model, transforms, neigh_t, ref_t)
            warped = warp_with_flow(neigh_t, flow)
            aligned_tensors.append(warped)

    # Stack and median on GPU
    stack = torch.cat(aligned_tensors, dim=0)  # [N, 3, H, W]
    median = stack.median(dim=0).values.unsqueeze(0)  # [1, 3, H, W]
    return tensor_to_cv(median)


def main():
    print("=== Approach C: RAFT Optical Flow (GPU) + Sliding Median ===")
    print(f"Device: {DEVICE}")
    print(f"Loading video at scale {SCALE}...")

    t0 = time.time()
    frames_cv, fps = load_frames_cv(VIDEO_PATH, SCALE)
    print(f"Loaded {len(frames_cv)} frames in {time.time()-t0:.1f}s")

    print("Loading RAFT model...")
    model, transforms = load_raft(DEVICE)
    print("RAFT ready.")

    half = N // 2

    # Test on specific frames
    rows = []
    for idx in TEST_INDICES:
        print(f"  Processing frame {idx}...")
        t1 = time.time()
        result = process_window_raft(frames_cv, idx, half, model, transforms, DEVICE)
        dt = time.time() - t1
        print(f"    Done in {dt:.1f}s")

        orig = frames_cv[idx].copy()
        cv2.putText(orig, f'Avant {idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(result, f'Apres RAFT N={N}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        rows.append(np.hstack([orig, result]))

    comparison = np.vstack(rows)
    out_path = os.path.join(OUTPUT_DIR, 'comparaison_raft.jpg')
    cv2.imwrite(out_path, comparison)
    print(f"Saved {out_path}")

    # Full video
    print(f"\nProcessing full video ({len(frames_cv)} frames)...")
    t0 = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frames_cv[0].shape[:2]
    vid_path = os.path.join(OUTPUT_DIR, 'result_raft_N5.mp4')
    out = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))

    for idx in range(len(frames_cv)):
        result = process_window_raft(frames_cv, idx, half, model, transforms, DEVICE)
        out.write(result)
        if idx % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Frame {idx}/{len(frames_cv)} ({elapsed:.0f}s)")

    out.release()
    print(f"Saved {vid_path} in {time.time()-t0:.1f}s")
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
