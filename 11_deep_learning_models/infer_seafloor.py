#!/usr/bin/env python3
"""
Seafloor-Invariant — Inférence uniquement depuis checkpoint existant.
Corrige le bug de dimension UNet (padding pour tailles non-multiples de 16).
"""
import os, sys, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
from tqdm import tqdm

VIDEO_PATH   = "/home/mickael/damien/01_extraction/subclip_5s.mp4"
MODEL_PATH   = "/home/mickael/damien/11_deep_learning_models/seafloor_invariant_run/unet_caustics.pth"
OUTPUT_DIR   = "/home/mickael/damien/11_deep_learning_models/seafloor_invariant_run"
RESULT_VIDEO = os.path.join(OUTPUT_DIR, "result_seafloor_invariant.mp4")
COMPARISON   = os.path.join(OUTPUT_DIR, "comparison_seafloor.mp4")
GRID_PATH    = os.path.join(OUTPUT_DIR, "grid_seafloor.jpg")
THRESH       = 0.35
N_WINDOW     = 7
SCALE        = 0.5
MORPH_K      = 15

device = torch.device("cuda")
print(f"Device: {device}")

# ─── UNet avec padding auto ───────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c,  out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True),
        )
    def forward(self, x): return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        s = self.conv(x); return s, self.pool(s)

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = ConvBlock(out_c * 2, out_c)
    def forward(self, x, skip):
        x = self.up(x)
        # Fix taille si skip légèrement différent (padding résiduel)
        if x.shape != skip.shape:
            x = TF.pad(x, [0, skip.shape[3]-x.shape[3], 0, skip.shape[2]-x.shape[2]])
        return self.conv(torch.cat([x, skip], dim=1))

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.e1 = EncoderBlock(in_ch, 64);  self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256);   self.e4 = EncoderBlock(256, 512)
        self.b  = ConvBlock(512, 1024)
        self.d1 = DecoderBlock(1024, 512);  self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128);   self.d4 = DecoderBlock(128, 64)
        self.out = nn.Conv2d(64, out_ch, 1)
    def forward(self, x):
        s1,p1=self.e1(x); s2,p2=self.e2(p1); s3,p3=self.e3(p2); s4,p4=self.e4(p3)
        b=self.b(p4)
        d=self.d1(b,s4); d=self.d2(d,s3); d=self.d3(d,s2); d=self.d4(d,s1)
        return self.out(d)

# ─── Charger modèle ──────────────────────────────────────────────────────────
model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print(f"✅ Modèle chargé depuis {MODEL_PATH}")

# ─── Charger vidéo ───────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)
W, H  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
WW, HH = int(W*SCALE), int(H*SCALE)

all_frames = []
for _ in range(total):
    ret, f = cap.read()
    if ret: all_frames.append(f)
cap.release()
print(f"Chargé: {len(all_frames)} frames 4K")

# ─── Inférence ───────────────────────────────────────────────────────────────
print("\nInférence UNet sur toutes les frames...")

def predict_mask(bgr_full):
    bgr_s = cv2.resize(bgr_full, (WW, HH))
    img   = torch.from_numpy(bgr_s.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(img)).squeeze().cpu().numpy()
    mask_s = (prob > THRESH).astype(np.uint8) * 255
    return cv2.resize(mask_s, (W, H), interpolation=cv2.INTER_NEAREST)

pred_masks_hd = []
for f in tqdm(all_frames, desc="Prédiction"):
    pred_masks_hd.append(predict_mask(f))

# Stats masques
coverages = [m.mean()/255*100 for m in pred_masks_hd]
print(f"Couverture masque — min:{min(coverages):.1f}%  mean:{np.mean(coverages):.1f}%  max:{max(coverages):.1f}%")

# ─── Inpainting temporel ─────────────────────────────────────────────────────
print("\nInpainting temporel guidé par masque DL...")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_vid = cv2.VideoWriter(RESULT_VIDEO, fourcc, fps, (W, H))
half = N_WINDOW // 2

for i, frame in enumerate(tqdm(all_frames, desc="Inpainting")):
    mask = pred_masks_hd[i]
    if mask.max() == 0:
        out_vid.write(frame); continue

    idxs     = [j for j in range(max(0,i-half), min(len(all_frames),i+half+1)) if j != i]
    stack    = np.stack([all_frames[j] for j in idxs], axis=0)
    med      = np.median(stack, axis=0).astype(np.uint8)

    result   = frame.copy()
    m_bool   = mask > 0
    result[m_bool] = med[m_bool]

    # Transition douce sur les bords du masque
    dist     = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    alpha    = np.clip(dist / 8.0, 0, 1)[..., None].astype(np.float32)
    result   = (frame.astype(np.float32) * (1-alpha) + result.astype(np.float32) * alpha).astype(np.uint8)
    out_vid.write(result)

out_vid.release()
print(f"✅ Vidéo résultat: {RESULT_VIDEO}")

# ─── Comparaison ─────────────────────────────────────────────────────────────
print("\nCréation vidéo de comparaison...")
cap1 = cv2.VideoCapture(VIDEO_PATH)
cap2 = cv2.VideoCapture(RESULT_VIDEO)
out_c = cv2.VideoWriter(COMPARISON, fourcc, fps, (W*2, H))
for _ in range(len(all_frames)):
    r1, f1 = cap1.read(); r2, f2 = cap2.read()
    if not r1 or not r2: break
    cv2.putText(f1, "Original",                  (30,80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,255), 5)
    cv2.putText(f2, "Seafloor-Invariant UNet",   (30,80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,255,255),   5)
    out_c.write(np.hstack([f1, f2]))
cap1.release(); cap2.release(); out_c.release()

# ─── Grille ──────────────────────────────────────────────────────────────────
print("\nGrille de comparaison...")
rows = []
cap_r = cv2.VideoCapture(RESULT_VIDEO)
for idx in [0, 30, 60, 90, 120]:
    if idx >= len(all_frames): continue
    f_o = cv2.resize(all_frames[idx], (960,540))
    m   = cv2.resize(pred_masks_hd[idx], (960,540), interpolation=cv2.INTER_NEAREST)
    m_c = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    # overlay masque en rouge semi-transparent
    overlay = f_o.copy()
    overlay[m > 0] = (0, 0, 200)
    cv2.addWeighted(overlay, 0.4, f_o, 0.6, 0, f_o)

    cap_r.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f_r = cap_r.read()
    f_r = cv2.resize(f_r, (960,540)) if ret else np.zeros_like(f_o)

    cv2.putText(f_o,  f"Original+mask #{idx}", ( 8,35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(m_c,  "DL Mask",               ( 8,35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),     2)
    cv2.putText(f_r,  "Résultat",              ( 8,35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255),   2)
    rows.append(np.hstack([f_o, m_c, f_r]))
cap_r.release()

cv2.imwrite(GRID_PATH, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"✅ Grille: {GRID_PATH}")
print("\n✅ Pipeline Seafloor-Invariant terminé !")
print(f"   Résultat    → {RESULT_VIDEO}")
print(f"   Comparaison → {COMPARISON}")
print(f"   Grille      → {GRID_PATH}")
