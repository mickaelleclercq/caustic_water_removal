"""
Temporal Median avec stabilisation ECC (Enhanced Correlation Coefficient).
ECC s'applique sur des niveaux de gris et aligne les frames avec une transformation 
de translation (MOTION_TRANSLATION) pour compenser le déplacement de caméra.
Ensuite, la médiane sur les frames alignées élimine les caustiques.
"""
import cv2
import numpy as np

video_path = "subclip_5s.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {total} frames, {w}x{h}, {fps:.1f}fps")

# Collecte des frames (downscale x2 pour vitesse et ECC)
scale = 0.25  # 4K -> 1/4 pour visu test
print(f"Working at {int(w*scale)}x{int(h*scale)} (scale={scale})")

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

# Stabilisation ECC autour de la frame centrale
ref_idx = len(frames) // 2
ref_frame = frames[ref_idx]
ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

aligned = [None] * len(frames)
aligned[ref_idx] = ref_frame

warp_mode = cv2.MOTION_TRANSLATION  # simple translation pour caméra qui avance
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 0.01)

print(f"Aligning frames using ECC (translation model)...")
fail_count = 0
for i, frame in enumerate(frames):
    if i == ref_idx:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp_matrix = cv2.findTransformECC(ref_gray, gray, warp_matrix, warp_mode, criteria)
        warped = cv2.warpAffine(frame, warp_matrix, (ref_frame.shape[1], ref_frame.shape[0]),
                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REFLECT)
        aligned[i] = warped
    except cv2.error:
        fail_count += 1
        aligned[i] = frame  # fallback

print(f"ECC alignment done. {fail_count} frames fallback (ECC failed).")

# Calcul de la médiane temporelle sur les frames alignées
print("Computing temporal median...")
stack = np.stack(aligned, axis=0).astype(np.float32)
median_result = np.median(stack, axis=0).astype(np.uint8)

# Sauvegarder comparaison: frame centrale brute vs résultat médiane
orig_frame = frames[ref_idx]
scale_out = 0.5
orig_s = cv2.resize(orig_frame, (0,0), fx=scale_out, fy=scale_out)
result_s = cv2.resize(median_result, (0,0), fx=scale_out, fy=scale_out)
cv2.putText(orig_s, "Avant", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
cv2.putText(result_s, "Apres ECC+Médiane", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
comparison = np.hstack((orig_s, result_s))
cv2.imwrite("result_ecc_median.jpg", comparison)
print("Done! Saved result_ecc_median.jpg")
