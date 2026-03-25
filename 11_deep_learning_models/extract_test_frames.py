#!/usr/bin/env python3
"""
Extraire des frames de la vidéo de test pour tester FUnIE-GAN
"""
import cv2
import os

# Config
video_path = "/home/mickael/damien/01_extraction/subclip_5s.mp4"
output_dir = "/home/mickael/damien/11_deep_learning_models/test_frames"
num_frames = 10  # Extraire 10 frames uniformément réparties

os.makedirs(output_dir, exist_ok=True)

# Ouvrir la vidéo
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Vidéo: {video_path}")
print(f"Total frames: {total_frames}")

# Calculer les indices de frames à extraire (uniformément répartis)
frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

frames_extracted = 0
for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    
    if ret:
        output_path = os.path.join(output_dir, f"frame_{idx:04d}.png")
        cv2.imwrite(output_path, frame)
        print(f"Frame {idx} sauvegardée: {output_path}")
        frames_extracted += 1

cap.release()
print(f"\n{frames_extracted} frames extraites dans {output_dir}")
