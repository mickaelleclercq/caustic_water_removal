# Deep Learning Models for Caustics Removal

Ce dossier contient les implémentations de modèles deep learning pour la suppression de caustiques.

## Modèles disponibles

Voir le document principal : `../MODELES_DL_CAUSTIQUES.md` pour la description complète.

### 1. Seafloor-Invariant Caustics Removal (2023)
- **GitHub:** https://github.com/pagraf/Seafloor-type-Invariant-Removal-of-Caustics-from-Underwater-Imagery
- **Statut:** Code disponible, MAJ récente (24 février 2025)
- **Recommandé pour:** Eaux peu profondes, fond corallien/sableux

### 2. RecGS - Recurrent Gaussian Splatting (2024)
- **GitHub:** https://github.com/tyz1030/recgs
- **Statut:** Code disponible, MAJ récente (21 février 2025)
- **Recommandé pour:** Approche 3D temporelle, séquences vidéo

## Installation rapide

```bash
# Depuis le répertoire racine du projet
cd 11_deep_learning_models

# Cloner les repos
git clone https://github.com/pagraf/Seafloor-type-Invariant-Removal-of-Caustics-from-Underwater-Imagery.git seafloor_invariant
git clone https://github.com/tyz1030/recgs.git

# Activer l'environnement
source ../myenv/bin/activate

# Installer les dépendances pour chaque modèle
cd seafloor_invariant
pip install -r requirements.txt 2>/dev/null || echo "Vérifier les dépendances dans le repo"

cd ../recgs
pip install -r requirements.txt 2>/dev/null || echo "Vérifier les dépendances dans le repo"
```

## Utilisation

Référez-vous aux README de chaque repo pour les instructions spécifiques.

Les modèles devront probablement être testés sur :
- Frames individuelles extraites de `../subclip_5s.mp4`
- Ou directement sur la vidéo (selon le modèle)

## Comparaison avec les autres approches

Après tests, comparer les résultats avec :
- Approche 06 : Homographie + Médiane
- Approche 07 : Optical Flow + Médiane
- Approche 08 : RAFT + Médiane
- Approche 10 : Mask Temporal Inpaint

Critères de comparaison :
- Qualité de suppression des caustiques
- Préservation de la netteté du fond
- Temps de traitement
- Artefacts visuels
