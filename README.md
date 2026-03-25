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

## Bilan & problème fondamental

| Approche | Réduction caustiques | Netteté fond | Practicabilité |
|---|---|---|---|
| Top-Hat inpainting | Partielle | Bonne | OK (lent) |
| Soustraction subtile | Faible | Bonne | Rapide |
| Bilateral / CLAHE | Nul | Dégradée | Rapide |
| MSR / MSRCP | Nul | OK | Rapide |
| Médiane globale ECC | Bonne | **Très floue** | Lent |
| Fenêtre N=5 ECC | Moyenne | **Floue** | Moyen |
| Fenêtre N=9 ECC | Meilleure | **Toujours floue** | Lent |

**Cause racine identifiée :** la caméra avance (mouvement de translation 3D, pas juste 2D). L'ECC 2D Euclidien (ou même Homographique) ne peut pas compenser un mouvement de caméra qui génère un **changement de parallaxe** sur une scène 3D. Résultat : le fond reste légèrement différent d'une frame à l'autre → la médiane floute.

---

## Pistes non encore explorées

- **Optical Flow dense** (Farnebäck, RAFT) pour warp plus précis que ECC
- **Homographie** (`MOTION_HOMOGRAPHY`) à la place d'ECC Euclidien
- **Détection & masquage direct** des caustiques par apprentissage profond (segmentation sémantique)
- **Fréquences spatiales** : séparer hautes fréquences (caustiques) / basses fréquences (fond) puis traiter uniquement les HF temporellement
- **Stabilisation robuste** avec RANSAC sur points d'intérêt (SIFT/ORB) avant la médiane
- **Approche fréquentielle** : les caustiques ont une fréquence temporelle élevée → filtre passe-bas temporel par pixel après alignement
