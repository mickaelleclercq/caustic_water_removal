# Suppression des caustiques — Vidéo sous-marine

Contexte : vidéo sous-marine (GoPro 4K, `GX010236_synced_enhanced.MP4`).
Objectif : supprimer ou atténuer les **caustiques** (les reflets lumineux en réseau qui se projettent sur le fond corallien/sableux) tout en conservant la netteté du fond.

---

## Approches tentées

### 1. Extraction / préparation

| Fichier | Description |
|---|---|
| `extract.py` | Extraction de frames fixes + sous-clip avec OpenCV |
| `extract_media.py` | Même chose via MoviePy |
| `extract_media_cv2.py` | Idem, version CV2 plus robuste |
| `mk_preview.py` | Génération de previews réduites (×0.15) |
| `mk_preview2.py` | Previews des résultats MSRCP |

Sous-clip de travail : `subclip_5s.mp4` (5 sec, résolution 4K originale).

---

### 2. Approches mono-frame (traitement image par image)

#### 2a. Morphologie / Top-Hat + Inpainting
**Fichiers :** `test_caustiques_inpaint.py`, `process_video_morph.py`, `process_video_morph_fast.py`

Principe :
1. Passer en HSV, isoler le canal V (luminosité)
2. Appliquer une **ouverture morphologique** (kernel elliptique ~15–21 px) → fond sans caustiques
3. Top-Hat = V − V_ouvert → détecte uniquement les pics lumineux
4. Masque binaire + inpainting OpenCV (`cv2.inpaint`) sur la zone détectée

Résultats : atténuation partielle, mais les bords de masque sont souvent visibles et le résultat semble artificiel sur les motifs fins.

---

#### 2b. Soustraction d'illumination subtile
**Fichier :** `test_subtraction.py`

Principe :
- Top-Hat sur canal V HSV
- Soustraction directe : `V_corrigé = V − caustics × 0.8`
- Légère augmentation de la saturation pour compenser

Résultats : atténuation douce, mais les caustiques brillantes restent bien visibles ; pas de vrai "retrait".

---

#### 2c. Filtres divers
**Fichier :** `test_filters.py`

Quatre variantes testées en parallèle sur une frame :
1. Original (référence)
2. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) sur canal L (LAB)
3. **Bilateral Filter** fort (d=15, σ=80) — lissage en préservant les bords
4. **Illumination Subtraction** via ouverture morphologique (kernel 31×31) sur canal V

Résultats : CLAHE n'enlève pas les caustiques. Bilateral lisse mais floute. L'illumination subtraction écrase la texture.

---

#### 2d. Multi-Scale Retinex (MSR)
**Fichier :** `test_retinex.py`

Principe : MSR canal par canal (B, G, R) avec σ = 15, 80, 250 px.
- `retinex = log(I) − log(I * G(σ))` pour chaque échelle
- Normalisation percentile (p2–p98)

Résultats : renforce le contraste local mais n'efface pas les caustiques — les pics lumineux restent extrêmement présents.

---

#### 2e. MSRCP (Multi-Scale Retinex with Colour Preservation)
**Fichier :** `test_msrcp.py`

Variante du MSR appliqué sur la luminance totale (moyenne des canaux), avec ratio de couleur conservé per-pixel.
Testé sur les frames à 5 s, 15 s, 30 s.

Résultats : légèrement meilleur rendu colorimétrique, mais les caustiques persistent tout autant.

---

### 3. Approches temporelles (médiane multi-frames)

Principe commun : les caustiques bougent d'une frame à l'autre → elles sont **incohérentes temporellement** → une **médiane** sur plusieurs frames alignées les efface. Le fond, lui, est stable après alignement → reste net.

---

#### 3a. Médiane globale + ECC (toute la vidéo)
**Fichier :** `test_ecc_median.py`

- Downscale à 0.25× (4K → ~960×540)
- Alignement de **toutes les frames** de la vidéo sur la frame centrale via **ECC** (`MOTION_TRANSLATION`)
- Médiane sur le stack complet

Résultats : les caustiques disparaissent bien mais le fond est **très flou** car : (1) la caméra se déplace sur 5 secondes → déformations importantes, (2) ECC translationnel pur ne compense pas la rotation ni la perspective.

---

#### 3b. Fenêtre glissante courte N=5
**Fichier :** `test_shortwindow.py`

- Fenêtre de N=5 frames consécutives autour de chaque frame cible
- Alignement **ECC Euclidien** (translation + rotation) des voisins sur la frame centrale
- Médiane sur les 5 frames alignées
- Test sur frames aux indices 15, 75, 130 à 0.25×

Résultats : meilleur que la médiane globale, mais toujours flou. La fenêtre est peut-être encore trop large ou l'ECC euclidien insuffisant (caméra qui avance = changement de perspective).

---

#### 3c. Fenêtre glissante N=9 et N=13
**Fichier :** `test_N_comparison.py`

Même approche que 3b, comparaison N=9 vs N=13.

Résultats : N=9 déjà trop flou. N=13 encore pire. L'augmentation de N n'aide pas ; elle amplifie le flou dû au mouvement de caméra.

---

#### 3d. Pipeline complet fenêtre courte (N=9, résolution 0.25×)
**Fichier :** `process_video_shortwindow.py`

Pipeline le plus abouti :
- ECC calculé à **basse résolution** pour la rapidité
- Warp appliqué à la **résolution de sortie** (0.25×) pour la précision
- Fenêtre de N=9 frames consécutives
- Médiane sur 9 frames alignées, frame par frame sur toute la vidéo
- Sortie : `subclip_5s_shortwindow_N9.mp4`

Résultats : toujours **flou**, malgré l'optimisation. Problème fondamental non résolu.

---

### 4. Approches temporelles améliorées — Alignement par homographie et optical flow

#### 4a. Homographie RANSAC + Médiane glissante (Approche A)
**Fichier :** `06_homography_median/process.py`

Remplacement de l'ECC par SIFT + FLANN matcher + homographie RANSAC (8 degrés de liberté). L'homographie modélise mieux le mouvement de caméra qui avance (parallaxe partielle). Fenêtre glissante N=5 frames.

Résultats : nettement meilleur que ECC euclidien pour l'alignement, mais la netteté reste dégradée (−74 % de sharpness). Le fond est trop warped par les erreurs résiduelles de perspective.

---

#### 4b. Optical Flow Farnebäck dense + Médiane glissante (Approche B)
**Fichier :** `07_optflow_median/process.py`

Warp dense pixel-par-pixel via `cv2.calcOpticalFlowFarneback` entre chaque voisin et la frame centrale, puis `cv2.remap`. Fenêtre N=5.

Résultats : qualité comparable à l'homographie. Le flow dense pixel-par-pixel ne suffit pas à compenser la parallaxe 3D — les zones profondes et les zones proches se déplacent différemment.

---

#### 4c. RAFT Optical Flow GPU + Médiane glissante (Approche C)
**Fichier :** `08_raft_median/process.py`

Utilisation du réseau RAFT (Recurrent All-pairs Field Transforms, `torchvision.models.optical_flow`) sur les A100. Flow plus précis que Farnebäck, warp GPU via `torch.nn.functional.grid_sample`. Fenêtre N=5.

Résultats : meilleur flow mais même problème de fond : la médiane floute toujours sur les zones à forte parallaxe.

---

#### 4d. Filtre passe-bas temporel après homographie (Approche D)
**Fichier :** `09_temporal_lowpass/process.py`

Au lieu de la médiane, application d'un filtre gaussien temporel (σ=2 frames) sur la série temporelle de chaque pixel après alignement par homographie. Les caustiques scintillantes (haute fréquence temporelle) sont lissées.

Résultats : suppression correcte des caustiques mais netteté très dégradée (−90 %). Le filtre temporel lisse autant le fond que les caustiques.

---

### 5. Masque caustiques + Remplacement temporel sélectif

#### 5a. Masque top-hat + médiane temporelle sélective (Approche E)
**Fichier :** `10_mask_temporal_inpaint/process.py`

Détection des caustiques via top-hat morphologique sur canal V (HSV) + seuillage adaptatif. Seuls les pixels masqués sont remplacés par la médiane temporelle des frames voisines alignées (N=7, homographie RANSAC). Les pixels hors-masque sont conservés intacts.

Résultats : la netteté du fond est quasiment préservée, mais les caustiques non détectées par le masque persistent. La qualité du masque est le point limitant.

---

#### 5b. RAFT + Masque élargi + N=15 (Approche G)
**Fichier :** `10_mask_temporal_inpaint/process_G_final.py`

Version raffinée : flow RAFT pour un warp plus précis, masque top-hat multi-seuils plus inclusif, fenêtre élargie à N=15 frames (0,5 s) pour plus de variation temporelle des caustiques, remplacement progressif via masque doux.

Résultats : meilleure suppression que E mais encore −64 à −76 % de sharpness sur les zones reconstituées.

---

### 6. Décomposition fréquentielle + médiane temporelle

#### 6a. Décomposition base/détail (Approche H)
**Fichier :** `10_mask_temporal_inpaint/process_H_decompose.py`

Séparation de chaque frame en `base = GaussianBlur(σ=25)` (illumination + caustiques) et `détail = frame − base` (texture pure). La médiane temporelle n'est appliquée que sur la couche base après alignement. Les erreurs d'alignement sont invisibles sur la couche lisse. La couche détail est conservée intacte.

Résultats : **−17 à −19 % de sharpness** — nette amélioration par rapport aux approches précédentes. Les caustiques larges disparaissent bien, mais les caustiques fines (fond sableux) qui ont des composantes haute fréquence persistent dans la couche détail.

---

#### 6b. Pyramide Laplacienne multi-niveaux (Approche I) ⭐
**Fichier :** `10_mask_temporal_inpaint/process_I_pyramid.py`

Décomposition en pyramide de Laplace à L=4 niveaux. Seul le niveau 0 (texture haute fréquence pure) est conservé intact. Tous les autres niveaux (bandes médio-fréquences + fond) reçoivent la médiane temporelle. Capture les caustiques à **toutes les échelles spatiales** (larges et fines).

Résultats : **−15 à −18 % de sharpness** — meilleure approche mesurée à date. Suppression visuelle nettement meilleure qu'en H, surtout sur fond sableux (frame 130).

---

#### 6c. Sweep paramétrique sur la pyramide (configurations I)
**Fichier :** `10_mask_temporal_inpaint/sweep_pyramid_params.py`

Exploration systématique des combinaisons `(keep_fine, levels, N)` sur 3 frames témoins, avec génération d'une grille visuelle et mesure de sharpness pour chaque config.

Configurations testées :
- `I_base` : keep_fine=1, L=4, N=9
- `I_keep0` : keep_fine=0, L=4, N=9 (tous les niveaux médianés)
- `I_L5` : keep_fine=1, L=5, N=9
- `I_L5_k0` : keep_fine=0, L=5, N=9
- `I_N13` : keep_fine=1, L=4, N=13

---

#### 6d. Pyramide Laplacienne + traitement sélectif du niveau fin (Approche J)
**Fichier :** `10_mask_temporal_inpaint/process_J_selective.py`

Extension de l'approche I : le niveau 0 (le plus fin) n'est pas conservé tel quel mais traité sélectivement. Un masque top-hat multi-échelle identifie les arêtes brillantes des caustiques fines dans ce niveau. Seuls les pixels du masque sont remplacés par la médiane ; la texture équilibrée (corail, sable) reste intacte.

Résultats : suppression des caustiques fines résiduelles de I, netteté comparable à I (−1 à −5 %).

---

### 7. Comparaison globale
**Fichier :** `10_mask_temporal_inpaint/compare_all.py`

Grille de comparaison multi-approches (Original, A, D, G, H, I) sur 3 frames témoins.
Sortie : `comparaison_all_approaches.jpg` + `sharpness_report.txt`.

---

## Bilan quantitatif des approches

### Réduction caustiques vs. netteté

| Approche | Réduction caustiques | Netteté fond | Practicabilité |
|---|---|---|---|
| Top-Hat inpainting | Partielle | Bonne | OK (lent) |
| Soustraction subtile | Faible | Bonne | Rapide |
| Bilateral / CLAHE | Nul | Dégradée | Rapide |
| MSR / MSRCP | Nul | OK | Rapide |
| Médiane globale ECC | Bonne | **Très floue** | Lent |
| Fenêtre N=5 ECC | Moyenne | **Floue** | Moyen |
| Fenêtre N=9 ECC | Meilleure | **Toujours floue** | Lent |
| A — Homographie | Bonne | **Floue (−75 %)** | Moyen |
| B — Farnebäck | Comparable à A | **Floue** | Moyen |
| C — RAFT flow | Bonne | **Floue** | GPU |
| D — Lowpass temporel | Bonne | **Très floue (−91 %)** | Moyen |
| E — Masque + médiane | Partielle | Bonne (hors masque) | Moyen |
| G — RAFT + masque | Bonne | **Floue (−64 à −76 %)** | GPU |
| H — Base/Détail | Bonne (larges) | **−17 à −19 %** | CPU |
| **I — Pyramide Laplace** | **Bonne (toutes échelles)** | **−15 à −18 %** | **CPU** |
| J — Pyramide sélective | **Bonne + fines** | **≈ I** | CPU |

### Sharpness absolue (variance Laplacien, après encodage mp4v)

| Approche | Frame 15 | Frame 75 | Frame 130 |
|:---|:---:|:---:|:---:|
| Original | 8703 | 6007 | 22092 |
| A Homographie | −74,6 % | −71,5 % | −81,3 % |
| D Lowpass | −90,8 % | −90,5 % | −89,6 % |
| G RAFT + Masque | −63,9 % | −68,3 % | −76,4 % |
| H Base/Détail | −17,0 % | −19,4 % | −12,7 % |
| **I Pyramide** | **−15,1 %** | **−18,0 %** | **−12,2 %** |

*Note : les mesures post-codec incluent ~13 % de flou dû à la compression mp4v.
En direct (avant encodage), H = −0,1 … −0,2 %, I = −1,2 … −4,3 %.*

**Cause racine des méthodes floues :** la caméra avance (translation 3D). L'alignement 2D (ECC, homographie, optical flow) ne peut pas compenser la parallaxe 3D → le fond reste légèrement différent d'une frame à l'autre → la médiane globale floute. La solution est de n'appliquer la médiane que sur les bandes fréquentielles qui contiennent réellement les caustiques (approches H, I, J).

---

### 8. Approches Deep Learning — Gaussian Splatting

#### 8a. FUnIE-GAN
**Dossier :** `11_deep_learning_models/`

Réseau génératif entraîné sur des paires d'images sous-marines dégradées/restaurées. Testé en inférence directe sur les frames.

Résultats : **inefficace** — le réseau améliore la colorimétrie générale mais n'a aucun effet sur les caustiques (non représentées dans son jeu d'entraînement).

---

#### 8b. Seafloor UNet (segmentation)
**Dossier :** `11_deep_learning_models/`

UNet entraîné pour segmenter le fond marin et en déduire les zones affectées par les caustiques.

Résultats : **trop agressif** — 87–97 % des pixels masqués, sur-segmentation massive.

---

#### 8c. RecGS — Recurrent Gaussian Splatting ⭐
**Dossier :** `11_deep_learning_models/recgs_output/`

Approche 3D : reconstruction de la scène statique via **3D Gaussian Splatting**, puis entraînement d'un module récurrent (RecGS) pour séparer l'illumination stable (fond) des variations temporelles (caustiques).

**Pipeline :**
1. Reconstruction COLMAP sur 75 frames à 1280×720 (74/75 caméras enregistrées, 20 156 points 3D, modèle PINHOLE)
2. Entraînement 3DGS vanilla — `sh_degree=3`, 30 000 itérations → loss = **0.107**
3. Entraînement RecGS — 30 000 itérations supplémentaires (checkpoint 30k→60k) → loss = **0.155**
4. Rendu des 75 frames : `gt/`, `renders/`, `compen/`, `fdiff/`

**Métriques finales (640×360, 75 frames) :**

| Métrique | Valeur |
|---|---|
| Luminosité GT | 114.7 ± 6.5 |
| Luminosité compen | 106.6 ± 11.5 |
| Réduction de luminosité | −8.0 ± 5.9 |
| MAE(gt, compen) | **21.7 px** (min 12.9, max 29.4) |
| Variance FDiff (énergie caustiques) | 19.9 |

**Sorties :**
- `recgs_output/result_recgs_compen_60k.mp4` — scène sans caustiques (4.3 MB)
- `recgs_output/result_recgs_comparison_60k.mp4` — GT | Compen côte à côte (8.6 MB)
- `recgs_output/result_recgs_4way_60k.mp4` — GT | Renders | Compen | FDiff (17.2 MB)
- `recgs_output/grid_recgs_60k.jpg` — grille visuelle 3 frames × 4 colonnes

**Résultats : mitigés.** Sur certaines vues (frame centrale ~37), la suppression est convaincante. Sur d'autres (frames 10 et 65), la reconstruction 3DGS génère des artéfacts de couleur (dominante rose/rouge, zones sombres) qui contaminent la sortie `compen`. La variance de luminosité augmente dans `compen` (±11.5 vs ±6.5 GT), signe d'instabilité temporelle. La résolution de sortie est également limitée à 640×360 (×6 en dessous du 4K d'origine).

**Conclusion RecGS :** L'approche est théoriquement solide (séparation illumination stable / caustiques via représentation 3D explicite) mais en pratique insuffisamment convaincante sur cette scène. La qualité dépend fortement de la couverture de vue COLMAP et de la convergence 3DGS — deux points difficiles à garantir sur une vidéo en mouvement continu. Durée totale : ~2h (30 min COLMAP + 49 min 3DGS + 1h07 RecGS) sur A100.

---

#### 8d. CausticsNet / BackscatterNet (ECCV 2024)
**Sources :** article + dépôt public `josauder/backscatternet_causticsnet`

Approche deep learning spécifiquement conçue pour l'imagerie sous-marine :
- **CausticsNet** vise la suppression des caustiques
- **BackscatterNet** vise la suppression du voile diffusant (backscatter)

Le point fort théorique de cette méthode est qu'elle est **auto-supervisée à partir de vidéos sous-marines**, avec une perte formulée via **monocular SLAM** et erreur de reprojection. C'est, sur le papier, l'approche deep learning la plus proche de notre problématique réelle (vidéo, mouvement caméra, absence de ground truth).

**Pourquoi elle n'a pas pu être utilisée ici :**
1. Le dépôt public n'expose pas de procédure d'inférence prête à l'emploi sur une vidéo ou un dossier d'images
2. Aucun checkpoint d'inférence public n'est fourni
3. Le README du dépôt indique encore : *"the full source code, model checkpoints & demo will be available"*
4. Le code disponible est surtout orienté **entraînement**
5. Le script `train_causticsnet.py` dépend d'un checkpoint externe de modèle SfM (`JointExperimentFixedIntrinsicsCenterLargeDatasetkeepalpha_best.pth`) non fourni

**Conclusion CausticsNet :** très bonne piste de recherche, probablement la plus pertinente parmi les approches deep learning trouvées, mais **non exploitable en pratique dans ce projet sans réentraînement ni artefacts supplémentaires publiés par les auteurs**. Dans notre contrainte actuelle (`pas de réentraînement`), la méthode doit donc être écartée.

---

## Conclusion générale

| Approche | Qualité suppression caustiques | Netteté | Temps calcul | Verdict |
|---|---|---|---|---|
| Top-Hat Inpainting | ★★☆ Partielle | ★★★ Bonne | Rapide | Acceptable |
| Médiane temporelle (meilleure config I) | ★★★ Bonne | ★★☆ −15 à −18 % | CPU moyen | **Meilleur compromis** |
| Pyramide sélective J | ★★★ Bonne + fines | ★★★ ≈ I | CPU moyen | **Meilleure mesurée** |
| FUnIE-GAN | ★☆☆ Nulle | ★★★ | Rapide (GPU) | ❌ Hors-sujet |
| RecGS (3DGS + récurrent) | ★★☆ Partielle | ★★☆ Artéfacts | ~2h (A100) | ❌ Pas satisfaisant |
| CausticsNet (ECCV 2024) | Théoriquement très pertinente | Non testée | N/A | ❌ Inutilisable sans poids/checkpoints |

**Meilleure approche opérationnelle :** Pyramide Laplacienne J (`process_J_selective.py`) — aucune dépendance deep learning, traitement CPU, −1 à −5 % de netteté réelle, suppression correcte à toutes les échelles spatiales.

---

## ⚠️ État final — Résultat insuffisant (approches CPU)

**Le résultat 4K final (`result_J_3840x2160_N9.mp4`) reste visuellement flou et insatisfaisant.**

Malgré l'approche J en pleine résolution 4K (−15 à −18 % de sharpness mesuré), la vidéo ne rend pas visuellement comme attendu : le rendu final apparaît flou. Aucune des approches testées n'a permis de supprimer les caustiques tout en conservant une netteté visuelle acceptable. Le problème fondamental reste non résolu.

---

## 🏆 Meilleures approches visuelles — Versions GPU (dossiers 12–15)

Les 4 meilleures méthodes identifiées visuellement ont été réimplémentées avec accélération GPU (PyTorch + 2× NVIDIA A100-SXM4-80GB) et appliquées sur la **vidéo source complète** (`GX010236_synced_enhanced.MP4`, 2,3 Go, 3840×2160, 1161 frames, ≈ 39s).

### Matériel et environnement
- **GPU :** 2× NVIDIA A100-SXM4-80GB (`cuda:0`, `cuda:1`)
- **RAM :** 456 Go — toutes les frames 4K chargées en mémoire (~15 Go/processus)
- **OpenCV :** 4.13.0 sans CUDA → toutes les opérations GPU via PyTorch (`F.grid_sample`, `F.avg_pool2d`, `torch.sort`)
- **Chargement frames :** ~15s (toutes les 1161 frames)

### Stratégie de parallélisation
- Homographies SIFT/RANSAC précomputées une seule fois (`precompute_homographies.py`), sauvegardées dans `homography_cache_half4.pkl`, chargées instantanément par chaque méthode
- GPU0 ← Méthode A + Méthode D (simultanément)
- GPU1 ← Méthode E + Méthode J (simultanément)
- Lancement via `run_gpu_all.sh`

### Résultats GPU sur la vidéo complète (1161 frames 4K)

| Méthode | Dossier | Paramètres | Temps/frame | Temps total | Taille sortie |
|---|---|---|---|---|---|
| A — Homographie + médiane | `12_gpu_homography/` | N=5 | **0,53 s/frame** | 612 s ≈ 10 min | 1,4 Go |
| D — Lowpass gaussien | `13_gpu_lowpass/` | N=9, σ=2 | **0,52 s/frame** | 608 s ≈ 10 min | 839 Mo |
| E — Masque top-hat + médiane | `14_gpu_mask_inpaint/` | N=7 | **0,87 s/frame** | 1012 s ≈ 17 min | 2,0 Go |
| J — Pyramide Laplacienne sélective | `15_gpu_pyramid_J/` | L=4, N=9 | **0,89 s/frame** | 1029 s ≈ 17 min | 2,1 Go |

**Accélération GPU vs CPU (méthode J référence) :** 0,89 s/frame GPU vs ~8,5 s/frame CPU → **×9,5 de speedup**.
