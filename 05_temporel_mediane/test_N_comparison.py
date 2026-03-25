"""
Test avec N=9 frames et N=13 frames pour comparer la réduction des caustiques.
"""
import cv2
import numpy as np

def align_small_group(frames, ref_idx):
    ref = frames[ref_idx]
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-3)
    aligned = []
    for i, f in enumerate(frames):
        if i == ref_idx:
            aligned.append(f)
            continue
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(ref_gray, gray, warp, cv2.MOTION_EUCLIDEAN, criteria)
            warped = cv2.warpAffine(f, warp, (ref.shape[1], ref.shape[0]),
                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_REFLECT)
            aligned.append(warped)
        except cv2.error:
            aligned.append(f)
    return aligned

video_path = "subclip_5s.mp4"
cap = cv2.VideoCapture(video_path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

scale = 0.25
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frames = []
while True:
    ret, frame = cap.read()
    if not ret: break
    frames.append(cv2.resize(frame, (0,0), fx=scale, fy=scale))
cap.release()

# Test sur frame 75 avec N=5, N=9 et N=13
center_idx = 75
orig = frames[center_idx].copy()

cols = [orig.copy()]
for N in [5, 9, 13]:
    half = N // 2
    start = max(0, center_idx - half)
    end = min(len(frames), center_idx + half + 1)
    group = frames[start:end]
    ref_in_group = center_idx - start
    
    aligned = align_small_group(group, ref_in_group)
    stack = np.stack(aligned, axis=0).astype(np.float32)
    median = np.median(stack, axis=0).astype(np.uint8)
    
    cv2.putText(median, f"N={N}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    cols.append(median)
    print(f"N={N} done")

cv2.putText(cols[0], "Avant", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)
comparison = np.hstack(cols)
cv2.imwrite("result_N_comparison.jpg", comparison)
print("Saved result_N_comparison.jpg")
