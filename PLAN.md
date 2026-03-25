# Plan d'attaque — Suppression des caustiques sous-marines

## Matériel disponible
- **Vidéo source** : `subclip_5s.mp4` — 3840×2160 (4K), 29.97 fps, 150 frames (5 sec)
- **GPUs** : 2× NVIDIA A100-SXM4-80GB (PyTorch CUDA OK)
- **CPUs** : 26 cœurs
- **PyTorch** CUDA enabledya, OpenCV 4.13 (CPU only)

## Pourquoi les approches précédentes ont échoué
1. **Mono-frame** (Top-Hat, Retinex, CLAHE) : les caustiques ont la même texture que le fond corallien → impossible de les isoler fiablement sur une seule image.
2. **Médiane temporelle + ECC** : la caméra avance en 3D → l'alignement 2D (translation/euclidien) est insuffisant → le fond est flou après médiane car les pixels ne se superposent pas bien.

## Nouvelles approches à tester (par ordre de faisabilité)

### Approche A — Homographie RANSAC + Médiane glissante
**Principe** : au lieu d'ECC (qui échoue sur les grands déplacements), utiliser l'extraction de points d'intérêt (ORB/SIFT) + estimation d'homographie par RANSAC. L'homographie modélise les 8 degrés de liberté (perspective) et supporte mieux un mouvement de caméra qui avance.
- Fenêtre N=5 frames consécutives
- SIFT + FLANN matcher → homographie RANSAC
- Warp perspectif des voisins sur la frame centrale
- Médiane sur les 5 frames alignées
- **GPU** : non nécessaire (OpenCV CPU rapide pour SIFT)
- **Dossier** : `06_homography_median/`

### Approche B — Optical Flow dense (Farnebäck) + Médiane glissante
**Principe** : au lieu d'un modèle global (homographie), utiliser un champ de déplacement dense pixel-par-pixel. L'optical flow dense compense les déformations locales dues à la parallaxe 3D. Chaque pixel est recarté individuellement.
- Fenêtre N=5 frames
- `cv2.calcOpticalFlowFarneback` entre chaque voisin et la frame centrale
- `cv2.remap` pour warper chaque voisin
- Médiane sur les frames warpées
- **GPU** : possible avec RAFT (PyTorch) pour une meilleure qualité
- **Dossier** : `07_optflow_median/`

### Approche C — RAFT Optical Flow (Deep Learning, GPU) + Médiane
**Principe** : RAFT (Recurrent All-pairs Field Transforms) est un réseau deep learning qui produit un optical flow bien plus précis que Farnebäck. Utilise les A100 pour une inférence rapide.
- Charger RAFT pré-entraîné via `torchvision.models.optical_flow`
- Calcul de l'optical flow entre frame centrale et chaque voisin
- Warp dense via `torch.nn.functional.grid_sample` (GPU)
- Médiane sur les frames warpées
- On peut paralléliser sur les 2 GPUs
- **Dossier** : `08_raft_median/`

### Approche D — Séparation fréquentielle temporelle
**Principe** : les caustiques sont des variations haute-fréquence temporelle (elles scintillent). Le fond est basse-fréquence temporelle (stable). Sans alignement, on peut appliquer un filtre passe-bas temporel **par pixel** après stabilisation minimale.
- Stabiliser par homographie
- Pour chaque pixel (x,y), extraire la série temporelle I(x,y,t)
- Appliquer un filtre passe-bas temporel (gaussien σ~2 frames)
- Les caustiques scintillantes sont supprimées, le fond stable reste
- **GPU** : PyTorch vectorisation sur tenseur 3D
- **Dossier** : `09_temporal_lowpass/`

### Approche E — Détection caustic mask + inpainting guidé
**Principe** : détecter les pixels de caustiques (saturation haute, luminosité V élevée, gradient fort) puis appliquer un inpainting guidé par les frames voisines (pas juste spatial).
- Masque de caustiques : top-hat + seuillage adaptatif sur le gradient
- Pour chaque pixel masqué, prendre la valeur médiane de ce même pixel dans les N frames voisines (après alignement par homographie)
- Le fond hors-masque reste intact → pas de flou
- **Dossier** : `10_mask_temporal_inpaint/`

### Approche F — Modèles Deep Learning Spécialisés ⭐ NOUVEAU
**Principe** : utiliser des modèles de deep learning pré-entraînés spécifiquement pour la suppression de caustiques sous-marines.
- **Seafloor-Invariant Caustics Removal** (2023) : CNN invariant au type de fond (corail, sable, roches)
  - Modèle spécialisé pour eaux peu profondes → exactement notre cas
  - Code disponible sur GitHub (MAJ février 2025)
  - Single-frame ou temporel
- **RecGS** (2024) : Recurrent Gaussian Splatting pour suppression de caustiques
  - Approche 3D temporelle ultra-moderne
  - Reconstruit la scène en 3DGS en séparant fond statique et caustiques dynamiques
  - Code disponible sur GitHub (MAJ février 2025)
  - Nécessite séquence vidéo
- **Hybride possible** : utiliser DL pour masque de détection + approches A/C/E pour suppression
- **GPU** : parfait pour les 2× A100
- **Dossier** : `11_deep_learning_models/`
- **Documentation complète** : `MODELES_DL_CAUSTIQUES.md`

### Approche H — Décomposition base/détail + médiane temporelle sur la base
**Principe** : séparer chaque frame en `base = GaussianBlur(σ=25)` (illumination + caustiques) et `détail = frame - base` (texture pure). Appliquer la médiane temporelle uniquement sur la couche base ; recombiner avec le détail original intact.
- Les erreurs d'alignement SIFT/homographie sont invisibles sur la couche lisse
- La netteté est quasiment inchangée : **-0.1 à -0.2 % de sharpness**
- Les caustiques larges sont bien supprimées, mais les caustiques fines restent légèrement visibles
- **Script** : `10_mask_temporal_inpaint/process_H_decompose.py`

### Approche I — Pyramide Laplacienne multi-niveaux ⭐ MEILLEURE ACTUELLEMENT
**Principe** : décomposer chaque frame en pyramide de Laplace à L=4 niveaux. Conserver intact seulement le niveau le plus fin (texture haute fréquence pure). Appliquer la médiane temporelle sur tous les autres niveaux (bandes médio-fréquentes → caustiques à toutes échelles, + fond basse fréquence).
- Capture les caustiques à **toutes les échelles spatiales** (larges ET fines)
- Netteté préservée à **-1.2 à -4.3 %** (mesuré en direct, avant codec vidéo)
- Suppression visuelle nettement meilleure que H, surtout sur fond sableux (frame 130)
- CPU-only, pas de GPU nécessaire
- **Script** : `10_mask_temporal_inpaint/process_I_pyramid.py`

---

## Résultats quantitatifs (Sharpness — Laplacian variance, après codec mp4v)

| Approche     | Frame 15 | Frame 75 | Frame 130 |
|:-------------|:--------:|:--------:|:---------:|
| Original     | 8702     | 6007     | 22092     |
| A Homographie| -74.6%   | -71.5%   | -81.3%    |
| D Lowpass    | -90.8%   | -90.5%   | -89.6%    |
| G RAFT+Mask  | -63.9%   | -68.3%   | -76.4%    |
| H BaseDetail | -17.0%   | -19.4%   | -12.7%    |
| **I Pyramid**| **-15.1%** | **-18.0%** | **-12.2%** |

*Note : les mesures post-codec incluent ~13% de flou dû à la compression mp4v.
 En direct (avant encodage) H=-0.1...-0.2%, I=-1.2...-4.3%.*

---

## Ordre d'exécution
1. **F1** (Seafloor-Invariant DL) — **PRIORITÉ** : modèle DL spécialisé, code dispo, rapide à tester
2. **A** (Homography + Médiane) — baseline améliorée, rapide
3. **B** (Farnebäck flow) — compare avec homographie
4. **C** (RAFT flow GPU) — meilleur flow possible
5. **F2** (RecGS) — approche 3D DL si Seafloor-Invariant prometteur
6. **D** (Temporal lowpass) — approche complètement différente
7. **E** (Mask + temporal inpaint) — hybride : netteté maximum sur le fond
8. **Hybride F+C** — DL pour masque + RAFT pour suppression temporelle

## Livrables pour chaque approche
- Script Python dans son dossier
- Image de comparaison : 3 frames (avant / après) côte à côte
- Vidéo de sortie de 5s (si résultat prometteur)
- Mesure objective : PSNR / SSIM par rapport à l'original (pour vérifier qu'on ne dégrade pas)
