#!/usr/bin/env python3
"""
Pipeline hybride: FUnIE-GAN (couleurs) + RAFT médiane (caustiques)

Approche:
1. Traiter la vidéo avec FUnIE-GAN pour correction des couleurs
2. Utiliser le code RAFT + médiane sur la vidéo améliorée pour supprimer les caustiques

Ou inversement:
1. RAFT + médiane d'abord (suppression caustiques)
2. FUnIE-GAN ensuite (amélioration couleurs)
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

print("=" * 60)
print("Pipeline Hybride: FUnIE-GAN + RAFT Médiane")
print("=" * 60)

# Config
video_input = "/home/mickael/damien/01_extraction/subclip_5s.mp4"
raft_result = "/home/mickael/damien/08_raft_median/result_raft_N5.mp4"
mask_inpaint_result = "/home/mickael/damien/10_mask_temporal_inpaint/result_raft_mask_N7.mp4"
funie_result = "/home/mickael/damien/11_deep_learning_models/result_funiegan.mp4"

print("\nFichiers disponibles:")
print(f"1. Original:           {video_input}")
print(f"2. RAFT + Médiane:     {raft_result}")
print(f"3. Mask Inpaint RAFT:  {mask_inpaint_result}")
print(f"4. FUnIE-GAN:          {funie_result}")

# Vérifier l'existence des fichiers
files_exist = all([
    Path(video_input).exists(),
    Path(raft_result).exists(),
    Path(mask_inpaint_result).exists(),
    Path(funie_result).exists()
])

if not files_exist:
    print("\n⚠️  Certains fichiers sont manquants. Vérifiez les chemins.")
    sys.exit(1)

print("\n" + "=" * 60)
print("Option 1: Appliquer FUnIE-GAN sur la sortie RAFT")
print("=" * 60)
print("\nCette approche:")
print("  1. Prend la vidéo déjà traitée par RAFT (caustiques supprimées)")
print("  2. Applique FUnIE-GAN pour améliorer les couleurs")
print("\nPour exécuter:")
print("  cd 11_deep_learning_models/FUnIE-GAN/PyTorch")
print("  # Extraire frames de result_raft_N5.mp4")
print("  # Appliquer FUnIE-GAN")
print("  # Reconstruire la vidéo")

print("\n" + "=" * 60)
print("Option 2: Appliquer RAFT sur la sortie FUnIE-GAN")
print("=" * 60)
print("\nCette approche:")
print("  1. Prend la vidéo FUnIE-GAN (couleurs améliorées)")
print("  2. Applique RAFT + médiane pour supprimer les caustiques")
print("\nPour exécuter:")
print("  cd 08_raft_median")
print("  # Modifier process.py pour utiliser result_funiegan.mp4")
print("  # Exécuter le traitement")

print("\n" + "=" * 60)
print("Recommandation")
print("=" * 60)
print("\nBasé sur les résultats actuels:")
print("\n✅ MEILLEURE APPROCHE: Mask Temporal Inpaint RAFT (approche 10)")
print("   - Suppression efficace des caustiques")
print("   - Préservation de la netteté du fond")
print("   - Pas besoin de post-traitement")

print("\n⚡ APPROCHE RAPIDE: FUnIE-GAN seul")
print("   - Très rapide (temps réel)")
print("   - Amélioration couleurs et contraste")
print("   - Mais caustiques toujours présentes")

print("\n🎨 APPROCHE HYBRIDE POTENTIELLE:")
print("   1. Mask Temporal Inpaint RAFT (suppression caustiques)")
print("   2. FUnIE-GAN (amélioration couleurs)")
print("   = Meilleur des deux mondes ?")

print("\n" + "=" * 60)
print("Comparaison visuelle recommandée")
print("=" * 60)
print("\nPour comparer visuellement:")
print("  1. Ouvrir comparison_grid.jpg")
print("  2. Comparer avec les résultats des approches 8 et 10")
print("  3. Décider si le pipeline hybride vaut le coup")

print("\n✓ Script d'information terminé")
print("  Tous les fichiers de résultats sont disponibles dans:")
print("  /home/mickael/damien/11_deep_learning_models/")
