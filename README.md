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
