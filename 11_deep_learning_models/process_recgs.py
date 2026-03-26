#!/usr/bin/env python3
"""
Pipeline RecGS complet pour la suppression de caustiques.

Étapes:
  1. Extraire frames de la vidéo (réduites pour COLMAP rapide)
  2. Reconstruire la scène 3D (poses caméra + nuage de points) via pycolmap
  3. Entraîner le 3D Gaussian Splatting vanilla (train.py)
  4. Entraîner RecGS (train_recgs.py) depuis le checkpoint vanilla
  5. Render et créer vidéo de comparaison
"""
import os, sys, shutil, subprocess, cv2, json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ─── Chemins ─────────────────────────────────────────────────────────────────
VIDEO_PATH   = "/home/mickael/damien/01_extraction/subclip_5s.mp4"
RECGS_DIR    = "/home/mickael/damien/11_deep_learning_models/recgs"
DATA_DIR     = "/home/mickael/damien/11_deep_learning_models/recgs_data"
OUTPUT_DIR   = "/home/mickael/damien/11_deep_learning_models/recgs_output"
RESULT_VIDEO = "/home/mickael/damien/11_deep_learning_models/recgs_output/result_recgs.mp4"
LOG_3DGS     = "/home/mickael/damien/11_deep_learning_models/recgs_3dgs.log"
LOG_RECGS    = "/home/mickael/damien/11_deep_learning_models/recgs_train.log"

PYTHON       = "/home/mickael/damien/myenv/bin/python3"
IMAGES_DIR   = os.path.join(DATA_DIR, "images")

# Résolution pour COLMAP (pas besoin de 4K)
COLMAP_W, COLMAP_H = 1280, 720
# Sous-échantillonnage temporel : 1 frame sur N
FRAME_SKIP = 2  # → ~75 frames sur 150 (bon équilibre vitesse/qualité 3DGS)

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("PIPELINE RecGS — Removing Water Caustics")
print("="*60)

# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Extraire les frames
# ════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Extraction des frames...")

cap   = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)
W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_paths = []
idx = 0
saved = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    if idx % FRAME_SKIP == 0:
        frame_r = cv2.resize(frame, (COLMAP_W, COLMAP_H))
        path = os.path.join(IMAGES_DIR, f"frame_{idx:05d}.jpg")
        cv2.imwrite(path, frame_r, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_paths.append(path)
        saved += 1
    idx += 1
cap.release()
print(f"  {saved} frames extraites → {IMAGES_DIR}")

# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Reconstruction COLMAP via pycolmap
# ════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Reconstruction COLMAP (SfM)...")

import pycolmap
from pycolmap import logging as colmap_logging

db_path     = os.path.join(DATA_DIR, "database.db")
sparse_dir  = os.path.join(DATA_DIR, "sparse")
os.makedirs(sparse_dir, exist_ok=True)

# Supprimer ancienne DB si elle existe
if os.path.exists(db_path):
    os.remove(db_path)

# Extraction des features
print("  Extraction des features SIFT...")
reader_opts = pycolmap.ImageReaderOptions(camera_model='SIMPLE_PINHOLE')
pycolmap.extract_features(
    database_path  = db_path,
    image_path     = IMAGES_DIR,
    reader_options = reader_opts,
)

# Matching séquentiel (vidéo = frames consécutives)
print("  Matching séquentiel...")
seq_opts = pycolmap.SequentialPairingOptions()
seq_opts.overlap = 10
pycolmap.match_sequential(
    database_path   = db_path,
    pairing_options = seq_opts,
)

# Reconstruction incrémentale
print("  Reconstruction incrémentale...")
maps = pycolmap.incremental_mapping(
    database_path    = db_path,
    image_path       = IMAGES_DIR,
    output_path      = sparse_dir,
)

if not maps:
    print("  ❌ COLMAP n'a pas trouvé de reconstruction. Essai avec matching exhaustif...")
    # Re-essai avec matching exhaustif
    if os.path.exists(db_path): os.remove(db_path)
    pycolmap.extract_features(database_path=db_path, image_path=IMAGES_DIR, reader_options=reader_opts)
    pycolmap.match_exhaustive(database_path=db_path)
    maps = pycolmap.incremental_mapping(database_path=db_path, image_path=IMAGES_DIR, output_path=sparse_dir)

if not maps:
    print("  ❌ COLMAP a échoué. Interruption.")
    sys.exit(1)

# Prendre la reconstruction la plus grande
best_key = max(maps, key=lambda k: len(maps[k].images))
reconstruction = maps[best_key]
n_imgs = len(reconstruction.images)
n_pts  = len(reconstruction.points3D)
print(f"  ✅ Reconstruction: {n_imgs}/{saved} images, {n_pts} points 3D")

# Exporter au format texte pour 3DGS
sparse_txt = os.path.join(sparse_dir, "0")
os.makedirs(sparse_txt, exist_ok=True)
reconstruction.write_text(sparse_txt)
print(f"  Exporté: {sparse_txt}")

# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Entraîner le 3DGS vanilla
# ════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Entraînement 3D Gaussian Splatting vanilla...")
print(f"  Log: {LOG_3DGS}")

# Paramètres réduits pour aller vite (résolution et itérations)
cmd_3dgs = [
    PYTHON, os.path.join(RECGS_DIR, "train.py"),
    "-s", DATA_DIR,
    "-m", OUTPUT_DIR,
    "--iterations", "7000",       # Paper: 30000, mais 7000 suffisent pour RecGS
    "--test_iterations", "7000",
    "--save_iterations", "7000",
    "--resolution", "2",           # Downscale ×2 pour aller 4× plus vite
    "--densification_interval", "200",
    "--opacity_reset_interval", "3000",
    "--densify_until_iter", "5000",
    "--quiet",
]
print(f"  Commande: {' '.join(cmd_3dgs[-10:])}")

with open(LOG_3DGS, "w") as logf:
    proc = subprocess.Popen(cmd_3dgs, stdout=logf, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONPATH": RECGS_DIR})
    proc.wait()

if proc.returncode != 0:
    print(f"  ❌ 3DGS a échoué (code {proc.returncode}). Voir {LOG_3DGS}")
    # Afficher les 20 dernières lignes du log
    with open(LOG_3DGS) as f:
        lines = f.readlines()
    print("".join(lines[-20:]))
    sys.exit(1)
print("  ✅ 3DGS vanilla entraîné")

# Trouver le checkpoint
chkpt_dir = os.path.join(OUTPUT_DIR, "point_cloud", "iteration_7000")
print(f"  Checkpoint: {chkpt_dir}")

# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Entraîner RecGS
# ════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Entraînement RecGS...")
print(f"  Log: {LOG_RECGS}")

# Chercher le checkpoint .pth
chkpt_path = None
for f in Path(OUTPUT_DIR).glob("chkpnt*.pth"):
    chkpt_path = str(f)
    break
# Aussi chercher dans chkpnts/
if chkpt_path is None:
    for f in Path(OUTPUT_DIR).rglob("chkpnt*.pth"):
        chkpt_path = str(f)
        break

if chkpt_path is None:
    # Parfois train.py sauvegarde seulement le point_cloud, pas de checkpoint .pth
    # On peut quand même lancer RecGS avec --start_checkpoint pointant sur le ply
    print("  ⚠️  Pas de checkpoint .pth trouvé — RecGS va s'entraîner from scratch avec le modèle pré-chargé")
    cmd_recgs = [
        PYTHON, os.path.join(RECGS_DIR, "train_recgs.py"),
        "-s", DATA_DIR,
        "-m", OUTPUT_DIR,
        "--iterations", "3000",
        "--resolution", "2",
        "--quiet",
    ]
else:
    print(f"  Checkpoint: {chkpt_path}")
    cmd_recgs = [
        PYTHON, os.path.join(RECGS_DIR, "train_recgs.py"),
        "-s", DATA_DIR,
        "-m", OUTPUT_DIR,
        "--start_checkpoint", chkpt_path,
        "--iterations", "3000",
        "--resolution", "2",
        "--quiet",
    ]

with open(LOG_RECGS, "w") as logf:
    proc = subprocess.Popen(cmd_recgs, stdout=logf, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONPATH": RECGS_DIR})
    proc.wait()

if proc.returncode != 0:
    print(f"  ❌ RecGS a échoué (code {proc.returncode}). Voir {LOG_RECGS}")
    with open(LOG_RECGS) as f:
        lines = f.readlines()
    print("".join(lines[-30:]))
    sys.exit(1)
print("  ✅ RecGS entraîné")

# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — Render RecGS → vidéo
# ════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Rendering RecGS...")

render_dir = os.path.join(OUTPUT_DIR, "renders_recgs")
os.makedirs(render_dir, exist_ok=True)

cmd_render = [
    PYTHON, os.path.join(RECGS_DIR, "render_recgs.py"),
    "-s", DATA_DIR,
    "-m", OUTPUT_DIR,
    "--skip_train",
    "--quiet",
]

log_render = "/home/mickael/damien/11_deep_learning_models/recgs_render.log"
with open(log_render, "w") as logf:
    proc = subprocess.Popen(cmd_render, stdout=logf, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONPATH": RECGS_DIR})
    proc.wait()

if proc.returncode != 0:
    print(f"  ❌ Render a échoué — voir {log_render}")
    with open(log_render) as f:
        lines = f.readlines()
    print("".join(lines[-20:]))
    sys.exit(1)

# Assembler les frames rendues en vidéo
renders_path = os.path.join(OUTPUT_DIR, "test", "ours_3000", "renders")
if not os.path.exists(renders_path):
    # Chercher d'autres dossiers de renders
    for p in Path(OUTPUT_DIR).rglob("renders"):
        renders_path = str(p)
        break

print(f"  Frames rendues: {renders_path}")
rendered_frames = sorted(Path(renders_path).glob("*.png")) if os.path.exists(renders_path) else []
print(f"  {len(rendered_frames)} frames rendues")

if rendered_frames:
    first = cv2.imread(str(rendered_frames[0]))
    rH, rW = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(RESULT_VIDEO, fourcc, fps, (rW, rH))
    for fp in tqdm(rendered_frames, desc="Assemblage vidéo"):
        f = cv2.imread(str(fp))
        if f is not None: out_vid.write(f)
    out_vid.release()
    print(f"\n  ✅ Vidéo RecGS: {RESULT_VIDEO}")
else:
    print("  ⚠️  Aucune frame rendue trouvée")

print("\n✅ Pipeline RecGS terminé !")
print(f"   3DGS log   → {LOG_3DGS}")
print(f"   RecGS log  → {LOG_RECGS}")
print(f"   Résultat   → {RESULT_VIDEO}")
