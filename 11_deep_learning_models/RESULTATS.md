# Résultats des Tests - Modèles Deep Learning

## Date: 25 mars 2026

## Vidéo de test
- **Fichier:** `/home/mickael/damien/01_extraction/subclip_5s.mp4`
- **Résolution:** 3840×2160 (4K)
- **Durée:** 5 secondes
- **Frames:** 150 frames @ 29.97 FPS

---

## 1. FUnIE-GAN (Fast Underwater Image Enhancement)

### Méthodologie
- **Modèle:** FUnIE-GAN (GAN basé sur Pix2Pix)
- **Publication:** IEEE RA-L 2020
- **Code source:** https://github.com/xahidbuffon/FUnIE-GAN
- **Modèle pré-entraîné:** Fourni avec le repo (27 MB)

### Implémentation
- PyTorch avec GPU (CUDA)
- Résolution de traitement: 256×256 (upscalé à 4K)
- Normalisation: [-1, 1]

### Performance
- ✅ **FPS de traitement:** 405.8 FPS sur GPU
- ✅ **Temps moyen par frame:** 2.46 ms
- ✅ **Temps total:** 0.37 sec (pour le traitement pur)
- ✅ **Temps total avec I/O:** ~25 sec

### Résultats

#### Fichiers générés:
1. **result_funiegan.mp4** (108 MB)
   - Vidéo complète traitée en 4K
   
2. **comparison_funiegan.mp4** (410 MB)
   - Comparaison côte à côte: Original | FUnIE-GAN
   
3. **comparison_grid.jpg** (3.1 MB)
   - Grille de comparaison de 5 frames clés
   - Dimensions: 4800×1080 pixels

#### Observations:

**Points positifs:**
- ✅ **Très rapide** - Traitement en temps réel possible
- ✅ **Amélioration de la couleur** - Correction de la dominante bleue/verte
- ✅ **Amélioration du contraste** - Meilleure visibilité globale
- ✅ **Code prêt à l'emploi** - Modèle pré-entraîné disponible

**Limitations:**
- ⚠️ **Pas spécifique aux caustiques** - Amélioration générale, pas de suppression ciblée
- ⚠️ **Caustiques toujours visibles** - Elles sont atténuées mais pas supprimées
- ⚠️ **Augmentation du bruit** - Parfois visible dans les zones sombres
- ⚠️ **Résolution de traitement** - Modèle entraîné sur 256×256, upscalé à 4K

**Verdict:**
FUnIE-GAN améliore l'apparence générale de la vidéo sous-marine (couleurs, contraste) mais ne supprime PAS spécifiquement les caustiques. Les motifs de lumière ondulants restent clairement visibles.

**Recommandation:**
Ce modèle peut être utilisé comme **pré-traitement** ou **post-traitement** en combinaison avec d'autres approches (RAFT + médiane temporelle, etc.), mais pas comme solution standalone pour la suppression de caustiques.

---

## 2. Seafloor-Invariant Caustics Removal (2023)

### Statut
❌ **Non testé** - Modèle pré-entraîné non fourni dans le repo

### Détails
- Repo cloné: ✅
- Code disponible: ✅ (Jupyter Notebook)
- Modèle pré-entraîné: ❌ Non fourni
- Dataset: Disponible sur Zenodo (https://doi.org/10.5281/zenodo.6467283)

### Prochaines étapes
Pour tester ce modèle, il faudrait:
1. Télécharger le dataset de formation
2. Entraîner le modèle (ou contacter les auteurs pour les poids)
3. Adapter le code du notebook pour l'inférence sur vidéo

---

## 3. RecGS - Recurrent Gaussian Splatting (2024)

### Statut
❌ **Non testé** - Nécessite configuration complexe

### Détails
- Repo cloné: ✅
- Dépendances: Gaussian Splatting framework complet
- Environnement: Nécessite conda environment spécifique
- Workflow: Entraînement 3DGS → RecGS → Rendering

### Complexité
RecGS nécessite:
1. Entraîner d'abord un modèle 3D Gaussian Splatting vanilla
2. Puis fine-tuner avec RecGS
3. Render les résultats

C'est une approche 3D complète, pas un simple modèle d'amélioration d'image.

### Prochaines étapes
Pour tester RecGS:
1. Installer l'environnement Gaussian Splatting complet
2. Préparer les données au format requis (caméra poses, etc.)
3. Suivre le workflow d'entraînement complet
4. Temps estimé: Plusieurs heures/jours

---

## Comparaison avec les Approches Précédentes

| Approche | Suppression Caustiques | Netteté | Vitesse | Prêt à l'emploi |
|----------|------------------------|---------|---------|-----------------|
| Top-Hat + Inpaint | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ✅ |
| MSRCP Retinex | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ✅ |
| ECC + Médiane | ⭐⭐ | ⭐ | ⭐⭐ | ✅ |
| Homographie + Médiane | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ✅ |
| RAFT + Médiane | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ✅ |
| Mask Temporal Inpaint | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ✅ |
| **FUnIE-GAN** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ |
| Seafloor-Invariant | ? | ? | ? | ❌ |
| RecGS | ? | ? | ? | ❌ |

---

## Conclusions et Recommandations

### Modèles DL testables immédiatement:
1. ✅ **FUnIE-GAN** - Testé, amélioration générale mais pas de suppression de caustiques

### Modèles DL prometteurs mais nécessitant plus de travail:
2. ⏳ **Seafloor-Invariant** - Spécialisé caustiques, mais pas de modèle pré-entraîné
3. ⏳ **RecGS** - Approche 3D avancée, mais setup complexe

### Meilleure approche actuelle:
La combinaison **RAFT + Médiane temporelle** ou **Mask Temporal Inpaint** (approches 8 et 10) reste la meilleure solution pour la suppression de caustiques pour l'instant.

### Approche hybride recommandée:
1. **Prétraitement:** FUnIE-GAN pour correction des couleurs
2. **Suppression caustiques:** RAFT + Médiane temporelle OU Mask Temporal Inpaint
3. **Post-traitement:** Léger sharpening si nécessaire

Ou:
1. **Suppression caustiques:** RAFT + Médiane
2. **Amélioration couleurs:** FUnIE-GAN

### Pour aller plus loin avec DL:
- Contacter les auteurs de Seafloor-Invariant pour obtenir les poids pré-entraînés
- Ou entraîner le modèle sur le dataset fourni
- Ou chercher d'autres modèles DL avec poids disponibles (Reti-Diff, UIEDP, etc.)

---

## Fichiers générés

```
11_deep_learning_models/
├── result_funiegan.mp4              (108 MB) - Vidéo traitée
├── comparison_funiegan.mp4          (410 MB) - Comparaison côte à côte
├── comparison_grid.jpg              (3.1 MB) - Grille de comparaison
├── test_frames/                     (10 frames PNG extraites)
├── funie_output/                    (10 frames traitées PNG)
├── FUnIE-GAN/                       (Repo cloné)
├── seafloor_invariant/              (Repo cloné)
├── recgs/                           (Repo cloné)
├── process_video_funiegan.py        (Script principal)
├── create_comparison_grid.py        (Script de comparaison)
└── extract_test_frames.py           (Script d'extraction)
```
