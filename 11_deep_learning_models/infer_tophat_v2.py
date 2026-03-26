#!/usr/bin/env python3
"""
Inférence Seafloor v2 — Masque Top-Hat amélioré + inpainting temporel.

Le UNet était mal entraîné (pseudo-labels trop larges).
Cette version utilise directement un Top-Hat avec des paramètres soignés :
  - Grand kernel relatif à la résolution (7% de la largeur)
  - Seuil adaptatif (percentile)
  - Uniquement les pics lumineux (pas les zones sombres)
"""
import os, cv2
import numpy as np
from tqdm import tqdm

VIDEO_PATH   = "/home/mickael/damien/01_extraction/subclip_5s.mp4"
OUTPUT_DIR   = "/home/mickael/damien/11_deep_learning_models/seafloor_invariant_run"
RESULT_VIDEO = os.path.join(OUTPUT_DIR, "result_tophat_v2.mp4")
COMPARISON   = os.path.join(OUTPUT_DIR, "comparison_tophat_v2.mp4")
GRID_PATH    = os.path.join(OUTPUT_DIR, "grid_tophat_v2.jpg")

N_WINDOW     = 9    # Nombre de frames voisines pour la médiane
SCALE        = 0.5  # Travailler en demi-résolution (plus rapide)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Charger vidéo ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)
W, H  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

all_frames = []
for _ in range(total):
    ret, f = cap.read()
    if ret: all_frames.append(f)
cap.release()
print(f"Chargé: {len(all_frames)} frames {W}×{H}")

WW, HH = int(W*SCALE), int(H*SCALE)

# ── Masque Top-Hat amélioré ───────────────────────────────────────────────────
def make_caustic_mask(bgr_full):
    """
    Détecte les caustiques avec Top-Hat + seuil percentile.
    Ne prend que les ZONES PLUS CLAIRES que leur voisinage local.
    """
    bgr = cv2.resize(bgr_full, (WW, HH))
    
    # Canal de luminance (mieux que V seul)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L   = lab[:,:,0].astype(np.float32)
    
    # Top-Hat avec grand kernel (~5% de la largeur)
    k_size = max(21, WW // 22)
    if k_size % 2 == 0: k_size += 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    tophat = cv2.morphologyEx(L.astype(np.uint8), cv2.MORPH_TOPHAT, k).astype(np.float32)
    
    # Seuil adaptatif : prendre les pixels > mean + 2*std du tophat
    # (seulement les vraies saillances locales)
    nonzero = tophat[tophat > 0]
    if len(nonzero) > 100:
        thr = np.mean(nonzero) + 1.5 * np.std(nonzero)
        thr = np.clip(thr, 15, 80)
    else:
        thr = 25
    
    mask = (tophat > thr).astype(np.uint8) * 255
    
    # Fermeture morphologique pour remplir les trous dans les caustiques
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    
    # Suppression des petits artefacts (composantes < 50 pixels)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    mask_clean = np.zeros_like(mask)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= 30:
            mask_clean[labels == lbl] = 255
    
    # Redimensionner au format pleine résolution
    mask_full = cv2.resize(mask_clean, (W, H), interpolation=cv2.INTER_NEAREST)
    return mask_full

# ── Pré-calculer tous les masques ────────────────────────────────────────────
print("\nCal masques Top-Hat sur toutes les frames...")
masks = []
coverages = []
for frame in tqdm(all_frames, desc="Masques"):
    m = make_caustic_mask(frame)
    masks.append(m)
    coverages.append((m > 0).mean() * 100)

print(f"Couverture masque — min:{min(coverages):.1f}%  mean:{np.mean(coverages):.1f}%  max:{max(coverages):.1f}%")

# ── Inpainting temporel guidé par masque ─────────────────────────────────────
print("\nInpainting temporel guidé par masque...")

fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
out_vid  = cv2.VideoWriter(RESULT_VIDEO, fourcc, fps, (W, H))
out_comp = cv2.VideoWriter(COMPARISON,   fourcc, fps, (W*2, H))

results = []
for i in tqdm(range(total), desc="Inpainting"):
    frame = all_frames[i]
    mask  = masks[i]
    
    if (mask > 0).mean() < 0.01:
        # Presque aucune caustique → frame inchangée
        results.append(frame.copy())
        continue
    
    # Fenêtre temporelle
    lo = max(0, i - N_WINDOW // 2)
    hi = min(total, i + N_WINDOW // 2 + 1)
    
    # Médiane temporelle des frames voisines (sans frame courante)
    nbrs = [all_frames[j] for j in range(lo, hi) if j != i]
    if not nbrs:
        results.append(frame.copy())
        continue
    med = np.median(np.stack(nbrs, axis=0), axis=0).astype(np.uint8)
    
    # Blending doux par distanceTransform
    mask_u8 = mask.astype(np.uint8)
    dist     = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    if dist.max() > 0:
        alpha = np.clip(dist / (dist.max() * 0.5), 0, 1)
    else:
        alpha = np.zeros_like(dist)
    alpha3 = alpha[:, :, None].astype(np.float32)
    
    result = (frame.astype(np.float32) * (1 - alpha3) +
              med.astype(np.float32)   *  alpha3).astype(np.uint8)
    results.append(result)

# ── Écrire vidéos ────────────────────────────────────────────────────────────
print("\nAssemblage vidéos...")
for i, (orig, res) in enumerate(tqdm(zip(all_frames, results), total=total)):
    out_vid.write(res)
    comp = np.hstack([orig, res])
    out_comp.write(comp)

out_vid.release()
out_comp.release()
print(f"✅ Résultat   → {RESULT_VIDEO}")
print(f"✅ Comparaison → {COMPARISON}")

# ── Grille de comparaison ─────────────────────────────────────────────────────
print("\nGrille de comparaison...")
idxs = [0, 25, 50, 75, 100, 125]
rows = []
for idx in idxs:
    if idx < len(all_frames):
        o = cv2.resize(all_frames[idx], (640, 360))
        m = cv2.cvtColor(cv2.resize(masks[idx], (640, 360)), cv2.COLOR_GRAY2BGR)
        r = cv2.resize(results[idx], (640, 360))
        rows.append(np.hstack([o, m, r]))
grid = np.vstack(rows)
cv2.imwrite(GRID_PATH, grid)
print(f"✅ Grille → {GRID_PATH}")

print("\n✅ Pipeline TopHat v2 terminé !")
