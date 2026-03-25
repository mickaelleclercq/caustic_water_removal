#!/usr/bin/env python3
"""
Créer une image de comparaison de plusieurs frames
Original vs FUnIE-GAN
"""
import cv2
import numpy as np
from pathlib import Path

# Config
video_orig = "/home/mickael/damien/01_extraction/subclip_5s.mp4"
video_funie = "/home/mickael/damien/11_deep_learning_models/result_funiegan.mp4"
output_img = "/home/mickael/damien/11_deep_learning_models/comparison_grid.jpg"

# Frames à comparer (frame 0, 30, 60, 90, 120)
frame_indices = [0, 30, 60, 90, 120]

# Ouvrir les vidéos
cap_orig = cv2.VideoCapture(video_orig)
cap_funie = cv2.VideoCapture(video_funie)

# Collecter les frames
frames_orig = []
frames_funie = []

for idx in frame_indices:
    # Original
    cap_orig.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap_orig.read()
    if ret:
        # Réduire la résolution pour la grille (1/4)
        frame_resized = cv2.resize(frame, (960, 540))
        frames_orig.append(frame_resized)
    
    # FUnIE-GAN
    cap_funie.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap_funie.read()
    if ret:
        frame_resized = cv2.resize(frame, (960, 540))
        frames_funie.append(frame_resized)

cap_orig.release()
cap_funie.release()

# Créer la grille: 5 colonnes (frames) x 2 lignes (orig/funie)
# Ligne 1: Original
row1 = np.hstack(frames_orig)
# Ligne 2: FUnIE-GAN
row2 = np.hstack(frames_funie)

# Ajouter des labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(row1, "Original", (20, 50), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
cv2.putText(row1, "Original", (20, 50), font, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

cv2.putText(row2, "FUnIE-GAN", (20, 50), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
cv2.putText(row2, "FUnIE-GAN", (20, 50), font, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

# Combiner les lignes
grid = np.vstack([row1, row2])

# Ajouter des numéros de frame
for i, idx in enumerate(frame_indices):
    x_pos = i * 960 + 850
    text = f"#{idx}"
    cv2.putText(grid, text, (x_pos, 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(grid, text, (x_pos, row1.shape[0] + 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

# Sauvegarder
cv2.imwrite(output_img, grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"Image de comparaison sauvegardée: {output_img}")
print(f"Dimensions: {grid.shape[1]}x{grid.shape[0]}")
