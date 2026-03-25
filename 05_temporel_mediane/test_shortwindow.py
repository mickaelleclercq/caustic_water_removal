"""
Approche : fenêtre glissante courte (N=5 frames consécutives).
Dans 0.15 sec, la caméra bouge peu mais les caustiques changent.
On aligne les N frames sur la frame centrale avec ECC (MOTION_EUCLIDEAN = translation + rotation)
puis médiane. On évalue le résultat sur quelques frames de test.
"""
import cv2
import numpy as np

def align_small_group(frames, ref_idx):
    """Aligne un petit groupe de frames sur la frame de référence via ECC."""
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
            aligned.append(f)  # fallback
    return aligned

video_path = "subclip_5s.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Downscale pour test rapide
scale = 0.25
print(f"Testing short-window temporal median (N=5 frames) at scale {scale}...")

# Charger toutes les frames en basse résolution
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    small = cv2.resize(frame, (0,0), fx=scale, fy=scale)
    frames.append(small)
cap.release()
print(f"Loaded {len(frames)} frames")

N = 5  # fenêtre glissante de N frames
half = N // 2

# On traite quelques frames de test (3 zones de la vidéo)
test_indices = [15, 75, 130]
results = []

for center_idx in test_indices:
    start = max(0, center_idx - half)
    end = min(len(frames), center_idx + half + 1)
    group = frames[start:end]
    ref_in_group = center_idx - start
    
    aligned = align_small_group(group, ref_in_group)
    stack = np.stack(aligned, axis=0).astype(np.float32)
    median = np.median(stack, axis=0).astype(np.uint8)
    results.append((frames[center_idx], median))
    print(f"  Frame {center_idx} done.")

# Sauvegarder comparaison
rows = []
for orig, res in results:
    cv2.putText(orig, "Avant", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv2.putText(res, "Apres ECC+Med(N=5)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    row = np.hstack((orig, res))
    rows.append(row)

final = np.vstack(rows)
cv2.imwrite("result_shortwindow.jpg", final)
print("Saved result_shortwindow.jpg")
