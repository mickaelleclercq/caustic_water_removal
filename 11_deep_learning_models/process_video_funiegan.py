#!/usr/bin/env python3
"""
Appliquer FUnIE-GAN sur toute la vidéo de 5 secondes
"""
import os
import sys
import time
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms

# Ajouter le chemin des nets de FUnIE-GAN
sys.path.insert(0, '/home/mickael/damien/11_deep_learning_models/FUnIE-GAN/PyTorch')
from nets import funiegan

# Config
video_input = "/home/mickael/damien/01_extraction/subclip_5s.mp4"
video_output = "/home/mickael/damien/11_deep_learning_models/result_funiegan.mp4"
model_path = "/home/mickael/damien/11_deep_learning_models/FUnIE-GAN/PyTorch/models/funie_generator.pth"

# Charger le modèle
print("Chargement du modèle FUnIE-GAN...")
is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor 

model = funiegan.GeneratorFunieGAN()
model.load_state_dict(torch.load(model_path, map_location='cpu'))
if is_cuda: 
    model.cuda()
    print("Utilisation du GPU")
else:
    print("Utilisation du CPU")
model.eval()

# Transformations d'image (256x256 pour le modèle)
img_width, img_height = 256, 256
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transform = transforms.Compose(transforms_)

# Fonction pour dénormaliser l'image
def denormalize(tensor):
    tensor = tensor * 0.5 + 0.5  # De [-1, 1] à [0, 1]
    return tensor.clamp(0, 1)

# Ouvrir la vidéo d'entrée
cap = cv2.VideoCapture(video_input)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"\nVidéo d'entrée: {video_input}")
print(f"Résolution originale: {orig_width}x{orig_height}")
print(f"Total frames: {total_frames}")
print(f"FPS: {fps}")

# Créer le writer vidéo (garder la résolution originale)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output, fourcc, fps, (orig_width, orig_height))

print(f"\nTraitement de la vidéo...")
processing_times = []

with torch.no_grad():
    for frame_idx in tqdm(range(total_frames), desc="Frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir BGR (OpenCV) en RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Transformer pour le modèle
        inp_img = transform(pil_img)
        inp_img = Variable(inp_img).type(Tensor).unsqueeze(0)
        
        # Traiter avec le modèle
        start = time.time()
        gen_img = model(inp_img)
        processing_times.append(time.time() - start)
        
        # Dénormaliser et convertir en numpy
        gen_img = denormalize(gen_img.squeeze(0).cpu())
        gen_np = gen_img.permute(1, 2, 0).numpy()  # CHW -> HWC
        gen_np = (gen_np * 255).astype(np.uint8)
        
        # Redimensionner à la résolution originale
        gen_resized = cv2.resize(gen_np, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)
        
        # Convertir RGB en BGR pour OpenCV
        gen_bgr = cv2.cvtColor(gen_resized, cv2.COLOR_RGB2BGR)
        
        # Écrire la frame
        out.write(gen_bgr)

cap.release()
out.release()

# Statistiques
total_time = sum(processing_times)
mean_time = np.mean(processing_times)
fps_processing = 1.0 / mean_time if mean_time > 0 else 0

print(f"\n✓ Traitement terminé !")
print(f"Vidéo de sortie: {video_output}")
print(f"Temps total: {total_time:.2f} sec")
print(f"Temps moyen par frame: {mean_time*1000:.2f} ms")
print(f"FPS de traitement: {fps_processing:.1f}")

# Créer une comparaison côte à côte
print(f"\nCréation d'une vidéo de comparaison...")
comparison_output = "/home/mickael/damien/11_deep_learning_models/comparison_funiegan.mp4"

cap1 = cv2.VideoCapture(video_input)
cap2 = cv2.VideoCapture(video_output)
out_comp = cv2.VideoWriter(comparison_output, fourcc, fps, (orig_width * 2, orig_height))

for _ in tqdm(range(total_frames), desc="Comparaison"):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break
    
    # Ajouter des labels
    cv2.putText(frame1, "Original", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame2, "FUnIE-GAN", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, (255, 255, 255), 3, cv2.LINE_AA)
    
    comparison = np.hstack([frame1, frame2])
    out_comp.write(comparison)

cap1.release()
cap2.release()
out_comp.release()

print(f"✓ Vidéo de comparaison: {comparison_output}")
print(f"\nTerminé !")
