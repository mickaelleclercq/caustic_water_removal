#!/usr/bin/env python3
"""
Seafloor-Invariant Caustics Detection - Entraînement + Inférence

Stratégie :
  1. Extraire 150 frames de la vidéo
  2. Générer des pseudo-masques via Top-Hat morphologique (même méthode que dans les approches précédentes)
  3. Entraîner le UNet (architecture du paper) sur ces paires (frame, masque)
  4. Appliquer le UNet entraîné sur toutes les frames pour obtenir des masques de qualité DL
  5. Utiliser les masques pour inpainting temporel

Pourquoi ça a du sens :
  - Le DL va apprendre à généraliser le Top-Hat (moins sensible aux seuils)
  - Le masque DL sera plus propre et plus précis que le Top-Hat brut
  - Peut détecter des caustiques faibles que le Top-Hat rate
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import sys

# ─── Paramètres ──────────────────────────────────────────────────────────────
VIDEO_PATH    = "/home/mickael/damien/01_extraction/subclip_5s.mp4"
OUTPUT_DIR    = "/home/mickael/damien/11_deep_learning_models/seafloor_invariant_run"
MODEL_PATH    = os.path.join(OUTPUT_DIR, "unet_caustics.pth")
MASK_DIR      = os.path.join(OUTPUT_DIR, "pred_masks")
RESULT_VIDEO  = os.path.join(OUTPUT_DIR, "result_seafloor_invariant.mp4")
COMPARISON    = os.path.join(OUTPUT_DIR, "comparison_seafloor.mp4")

PATCH_SIZE    = 256      # Taille d'entraînement
N_EPOCHS      = 10       # Rapide mais suffisant (A100)
BATCH_SIZE    = 32
LR            = 1e-3
MORPH_KERNEL  = 15       # Kernel Top-Hat pour pseudo-labels
CAUSTIC_THRESH= 12       # Seuil pour le masque (0-255)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MASK_DIR,   exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ─── Architecture UNet (du paper Seafloor-Invariant) ─────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c,  out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        s = self.conv(x)
        return s, self.pool(s)

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = ConvBlock(out_c * 2, out_c)
    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.e1 = EncoderBlock(in_ch,  64)
        self.e2 = EncoderBlock(64,    128)
        self.e3 = EncoderBlock(128,   256)
        self.e4 = EncoderBlock(256,   512)
        self.b  = ConvBlock(512, 1024)
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512,  256)
        self.d3 = DecoderBlock(256,  128)
        self.d4 = DecoderBlock(128,   64)
        self.out = nn.Conv2d(64, out_ch, 1)
    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        b      = self.b(p4)
        d      = self.d1(b,  s4)
        d      = self.d2(d,  s3)
        d      = self.d3(d,  s2)
        d      = self.d4(d,  s1)
        return self.out(d)   # [B, 1, H, W] logits

# ─── Pseudo-masque Top-Hat ────────────────────────────────────────────────────
def make_tophat_mask(bgr_frame, kernel_size=MORPH_KERNEL, thresh=CAUSTIC_THRESH):
    """Génère un masque binaire des caustiques par Top-Hat sur canal V."""
    hsv  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    v    = hsv[:, :, 2]
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    th   = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, k)
    _,  mask = cv2.threshold(th, thresh, 255, cv2.THRESH_BINARY)
    # Légère dilatation pour couvrir les bords
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return mask  # uint8, 0 ou 255

# ─── Dataset patches ──────────────────────────────────────────────────────────
class CausticsDataset(Dataset):
    def __init__(self, frames, masks, patch_size=PATCH_SIZE, samples_per_frame=50):
        self.patches_img  = []
        self.patches_mask = []
        h, w = frames[0].shape[:2]
        for img, msk in zip(frames, masks):
            for _ in range(samples_per_frame):
                x = np.random.randint(0, h - patch_size)
                y = np.random.randint(0, w - patch_size)
                p_img = img[x:x+patch_size, y:y+patch_size]   # BGR uint8
                p_msk = msk[x:x+patch_size, y:y+patch_size]   # 0/255
                # Augmentation : flip H/V aléatoire
                if np.random.rand() > 0.5: p_img, p_msk = p_img[::-1], p_msk[::-1]
                if np.random.rand() > 0.5: p_img, p_msk = p_img[:,::-1], p_msk[:,::-1]
                self.patches_img.append(p_img.copy())
                self.patches_mask.append(p_msk.copy())
        print(f"Dataset: {len(self.patches_img)} patches")

    def __len__(self): return len(self.patches_img)

    def __getitem__(self, i):
        img  = self.patches_img[i].astype(np.float32) / 255.0
        img  = torch.from_numpy(img.transpose(2, 0, 1))  # CHW
        mask = self.patches_mask[i].astype(np.float32) / 255.0
        mask = torch.from_numpy(mask[None])               # 1HW
        return img, mask

# ════════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Charger toutes les frames + générer pseudo-labels
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ÉTAPE 1 — Chargement vidéo + pseudo-labels Top-Hat")
print("="*60)

cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)
W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Vidéo: {total} frames, {W}x{H}, {fps:.2f} fps")

# Travailler en demi-résolution pour économiser la mémoire GPU
SCALE = 0.5
WW, HH = int(W * SCALE), int(H * SCALE)

all_frames = []
all_masks  = []
for i in tqdm(range(total), desc="Chargement + pseudo-labels"):
    ret, frame = cap.read()
    if not ret: break
    frame_s = cv2.resize(frame, (WW, HH))
    mask    = make_tophat_mask(frame_s)
    all_frames.append(frame_s)
    all_masks.append(mask)
cap.release()
print(f"Chargé: {len(all_frames)} frames en {WW}x{HH}")

# ════════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Entraîner le UNet
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ÉTAPE 2 — Entraînement UNet")
print("="*60)

dataset    = CausticsDataset(all_frames, all_masks, PATCH_SIZE, samples_per_frame=60)
loader     = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

model      = UNet().to(device)
optimizer  = optim.Adam(model.parameters(), lr=LR)
scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
criterion  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]).to(device))  # poids plus élevé sur caustiques

for epoch in range(N_EPOCHS):
    model.train()
    losses = []
    for imgs, masks in tqdm(loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        pred = model(imgs)
        loss = criterion(pred, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    scheduler.step()
    print(f"  Epoch {epoch+1}/{N_EPOCHS}  loss={np.mean(losses):.4f}  lr={scheduler.get_last_lr()[0]:.6f}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModèle sauvegardé: {MODEL_PATH}")

# ════════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Inférence sur toutes les frames
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ÉTAPE 3 — Inférence UNet sur toutes les frames")
print("="*60)

model.eval()

def predict_mask(bgr_frame_small):
    """Prédit un masque de caustiques pour une frame (demi-résolution)."""
    img = bgr_frame_small.astype(np.float32) / 255.0
    inp = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(inp)
        prob  = torch.sigmoid(logit).squeeze().cpu().numpy()
    mask = (prob > 0.4).astype(np.uint8) * 255
    return mask

pred_masks_small = []
for frame_s in tqdm(all_frames, desc="Prédiction masques"):
    m = predict_mask(frame_s)
    pred_masks_small.append(m)

print(f"Masques prédits: {len(pred_masks_small)}")

# ════════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Inpainting temporel guidé par masque DL
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ÉTAPE 4 — Inpainting temporel guidé par masque DL")
print("="*60)

N_WINDOW = 7  # Nombre de frames voisines

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(RESULT_VIDEO, fourcc, fps, (W, H))

# Charger les frames originales (pleine résolution) en mémoire — 4K est lourd,
# on lit frame par frame depuis la vidéo et on garde un buffer
cap = cv2.VideoCapture(VIDEO_PATH)
orig_frames = []
for _ in range(total):
    ret, f = cap.read()
    if ret: orig_frames.append(f)
cap.release()

print(f"Buffer de {len(orig_frames)} frames 4K en RAM")

for i in tqdm(range(len(orig_frames)), desc="Inpainting temporel"):
    frame   = orig_frames[i].copy()
    # Masque upscalé à la pleine résolution
    mask_s  = pred_masks_small[i]
    mask_hd = cv2.resize(mask_s, (W, H), interpolation=cv2.INTER_NEAREST)

    if mask_hd.max() == 0:
        # Pas de caustiques → garder frame originale
        out.write(frame)
        continue

    # Médiane des frames voisines sur les pixels masqués
    half = N_WINDOW // 2
    idxs = range(max(0, i - half), min(len(orig_frames), i + half + 1))
    neighbors = [orig_frames[j] for j in idxs if j != i]

    if len(neighbors) == 0:
        out.write(frame)
        continue

    stack     = np.stack(neighbors, axis=0)   # [N, H, W, 3]
    median_frame = np.median(stack, axis=0).astype(np.uint8)

    # Remplacer uniquement les pixels masqués
    mask_bool = mask_hd > 0
    result    = frame.copy()
    result[mask_bool] = median_frame[mask_bool]

    # Léger flou sur les bords du masque pour fondre la transition
    result_blur = cv2.GaussianBlur(result, (5, 5), 0)
    # Zone de transition : légère pondération
    dist_mask = cv2.distanceTransform(mask_hd, cv2.DIST_L2, 5)
    dist_mask = np.clip(dist_mask / 5.0, 0, 1)
    for c in range(3):
        result[:, :, c] = (result_blur[:, :, c] * dist_mask +
                           result[:, :, c] * (1 - dist_mask)).astype(np.uint8)

    out.write(result)

out.release()
print(f"\nVidéo résultat: {RESULT_VIDEO}")

# ════════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — Vidéo de comparaison
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ÉTAPE 5 — Vidéo de comparaison")
print("="*60)

cap1 = cv2.VideoCapture(VIDEO_PATH)
cap2 = cv2.VideoCapture(RESULT_VIDEO)
out_c = cv2.VideoWriter(COMPARISON, fourcc, fps, (W * 2, H))

for i in range(len(orig_frames)):
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not ret1 or not ret2: break
    cv2.putText(f1, "Original",                (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,255), 5)
    cv2.putText(f2, "Seafloor-Invariant UNet", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,255), 5)
    out_c.write(np.hstack([f1, f2]))

cap1.release(); cap2.release(); out_c.release()
print(f"Comparaison: {COMPARISON}")

# Grille d'images fixes
print("\nCréation grille de comparaison...")
grid_rows = []
for idx in [0, 30, 60, 90, 120]:
    if idx >= len(orig_frames): continue
    f_orig   = cv2.resize(orig_frames[idx], (960, 540))
    m_small  = pred_masks_small[idx]
    m_hd     = cv2.resize(m_small, (960, 540), interpolation=cv2.INTER_NEAREST)
    m_color  = cv2.cvtColor(m_hd, cv2.COLOR_GRAY2BGR)

    cap_r = cv2.VideoCapture(RESULT_VIDEO)
    cap_r.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f_res = cap_r.read()
    cap_r.release()
    f_res = cv2.resize(f_res, (960, 540)) if ret else np.zeros_like(f_orig)

    cv2.putText(f_orig,  f"Orig #{idx}",   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(m_color, "DL Mask",         (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),     2)
    cv2.putText(f_res,   "Inpainted",        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),   2)
    row = np.hstack([f_orig, m_color, f_res])
    grid_rows.append(row)

grid = np.vstack(grid_rows)
grid_path = os.path.join(OUTPUT_DIR, "grid_seafloor.jpg")
cv2.imwrite(grid_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"Grille: {grid_path}")

print("\n✅ Seafloor-Invariant pipeline terminé !")
print(f"   Masque DL  → {MASK_DIR}")
print(f"   Résultat   → {RESULT_VIDEO}")
print(f"   Comparaison→ {COMPARISON}")
print(f"   Grille     → {grid_path}")
