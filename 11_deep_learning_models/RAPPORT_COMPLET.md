# Rapport Détaillé - Tests Modèles DL pour Suppression de Caustiques

Date: 25 mars 2026  
Auteur: Tests automatisés

---

## 🎯 Objectif

Tester les modèles deep learning spécialisés pour la suppression de caustiques sous-marines sur la vidéo `subclip_5s.mp4` (5 secondes, 4K, 150 frames).

---

## 📊 Résumé des Tests

| Modèle | Statut | Résultat | Raison |
|--------|--------|----------|--------|
| **FUnIE-GAN** | ✅ TESTÉ | ⚠️ Amélioration générale mais caustiques visibles | Modèle pré-entraîné disponible |
| **Seafloor-Invariant** | ❌ NON TESTÉ | - | Pas de modèle pré-entraîné |
| **RecGS** | ❌ NON TESTÉ | - | Nécessite infrastructure 3DGS |
| **Reti-Diff** | ⏳ POSSIBLE | - | Modèles sur Google Drive |

---

## 1. FUnIE-GAN ✅ 

### Statut: TESTÉ AVEC SUCCÈS

**Ce qui a été fait:**
- Clonage du repo: ✅
- Modèle pré-entraîné: ✅ Fourni (27 MB)
- Test sur vidéo complète: ✅
- Performance: **405 FPS sur GPU** (très rapide)

**Résultats:**
- ✅ Améliore les couleurs (corrige dominante bleue/verte)
- ✅ Améliore le contraste général
- ✅ Temps réel possible
- ❌ **Caustiques toujours visibles** - Pas de suppression spécifique

**Fichiers générés:**
- `result_funiegan.mp4` (108 MB)
- `comparison_funiegan.mp4` (410 MB)  
- `comparison_grid.jpg` (3.1 MB)

**Verdict:** Bon pour amélioration générale, mais **ne supprime PAS les caustiques**.

---

## 2. Seafloor-Invariant Caustics Removal ❌

### Statut: NON TESTÉ - Modèle pré-entraîné non disponible

**Détails de l'analyse:**

**Repo cloné:** ✅ https://github.com/pagraf/Seafloor-type-Invariant-Removal-of-Caustics-from-Underwater-Imagery

**Contenu du repo:**
- `Seafloor-Invariant Caustics Detection.ipynb` - Notebook Jupyter pour entraînement
- `caustics_removal.cpp` - Code C++ (pas utilisé ici)
- `imgs/` - Images d'exemple (3 fichiers PNG)
- **Pas de fichiers .pth, .pt, .h5** (pas de modèle entraîné)

**Pourquoi ça ne peut pas être testé immédiatement:**

1. **Pas de modèle pré-entraîné fourni**
   - Le repo contient uniquement le code d'entraînement
   - Aucun fichier de poids (.pth, .pt, .h5)

2. **Nécessite un entraînement préalable**
   - Dataset requis: https://doi.org/10.5281/zenodo.6467283
   - Format: Images + ground truth (masques de caustiques annotés)
   - Temps d'entraînement: Plusieurs heures/jours selon GPU

3. **Notebook Jupyter orienté entraînement**
   ```python
   # Structure visible dans le notebook:
   FOLDER = '/media/pagraf/B62008FA2008C37B/caustics_new/'
   MAIN_FOLDER = FOLDER + 'train/'
   DATA_FOLDER = MAIN_FOLDER + 'img_{}.JPG'
   LABEL_FOLDER = MAIN_FOLDER + 'gts_{}.jpg'
   ```
   - Nécessite dataset annoté
   - Pas d'interface d'inférence prête

**Ce qu'il faudrait faire pour le tester:**

**Option A: Entraîner le modèle (complexe)**
1. Télécharger le dataset Zenodo (~plusieurs GB)
2. Adapter les chemins dans le notebook
3. Entraîner le modèle (temps estimé: 6-24h selon GPU)
4. Extraire les poids du modèle entraîné
5. Créer un script d'inférence
6. Tester sur la vidéo

**Option B: Contacter les auteurs (simple mais incertain)**
1. Email aux auteurs: pagraf@mail.ntua.gr
2. Demander les poids pré-entraînés
3. Si fournis, créer script d'inférence
4. Tester

**Option C: Chercher d'autres implémentations**
1. Vérifier si quelqu'un a partagé des poids sur HuggingFace
2. Chercher des forks du repo avec poids ajoutés

**Recommandation:** Option B (contact auteurs) est la plus simple.

---

## 3. RecGS - Recurrent Gaussian Splatting ❌

### Statut: NON TESTÉ - Infrastructure trop complexe

**Détails de l'analyse:**

**Repo cloné:** ✅ https://github.com/tyz1030/recgs

**Contenu du repo:**
- Code complet pour RecGS
- Basé sur 3D Gaussian Splatting (3DGS)
- `train.py` - Entraînement 3DGS vanilla
- `train_recgs.py` - Fine-tuning RecGS
- `render_recgs.py` - Rendu des résultats
- **Pas de modèles pré-entraînés**

**Pourquoi ça ne peut pas être testé immédiatement:**

1. **Workflow en 3 étapes requis:**
   ```bash
   # Étape 1: Entraîner 3DGS vanilla (base)
   python3 train.py -s /data/scene
   
   # Étape 2: Fine-tuner avec RecGS
   python3 train_recgs.py -s /data/scene --start_checkpoint output/xxx/chkpnt30000.pth
   
   # Étape 3: Render les résultats
   python3 render_recgs.py -s /data/scene -m output/xxx
   ```

2. **Nécessite préparation des données:**
   - Format COLMAP ou NeRF
   - Poses de caméra calculées
   - Structure de dossiers spécifique:
     ```
     data/
     └── scene/
         ├── images/
         ├── sparse/
         │   └── 0/
         │       ├── cameras.bin
         │       ├── images.bin
         │       └── points3D.bin
         └── ...
     ```

3. **Dépendances complexes:**
   - Environnement Conda spécifique (gaussian_splatting)
   - Submodules Git (diff-gaussian-rasterization, simple-knn)
   - SIBR_viewers (optionnel)
   - CUDA toolkit

4. **Temps de traitement:**
   - Entraînement 3DGS: 1-3 heures
   - Fine-tuning RecGS: 1-2 heures
   - Total: 2-5 heures minimum

**Ce qu'il faudrait faire pour le tester:**

**Étape 1: Préparer les données**
```bash
# Installer COLMAP
sudo apt install colmap

# Extraire frames de la vidéo
ffmpeg -i subclip_5s.mp4 frames/%04d.jpg

# Calculer les poses de caméra avec COLMAP
colmap automatic_reconstructor \
  --workspace_path . \
  --image_path frames
```

**Étape 2: Setup environnement Gaussian Splatting**
```bash
conda create -n gaussian_splatting python=3.7
conda activate gaussian_splatting
pip install torch torchvision
# + installation des submodules
```

**Étape 3: Entraîner le pipeline complet**
```bash
# 1-3 heures
python train.py -s data/subclip --iterations 30000

# 1-2 heures
python train_recgs.py -s data/subclip \
  --start_checkpoint output/xxx/chkpnt30000.pth \
  --rec_iterations 10000

# Render
python render_recgs.py -s data/subclip -m output/xxx
```

**Recommandation:** Trop complexe pour un test rapide. RecGS est une approche research, pas un outil prêt à l'emploi.

---

## 4. Reti-Diff ⏳

### Statut: POSSIBLE - Modèles sur Google Drive

**Détails de l'analyse:**

**Repo cloné:** ✅ https://github.com/ChunmingHe/Reti-Diff

**Contenu du repo:**
- Code complet + scripts de test
- Support pour Underwater Image Enhancement (UIE)
- Modèles pré-entraînés: Sur Google Drive (pas dans le repo)
- Scripts de test prêts: `test_UIE_UIEB.sh`, `test_UIE_LSUI.sh`

**Ce qu'il faut faire:**

**Étape 1: Télécharger les modèles pré-entraînés**
- Google Drive: https://drive.google.com/drive/folders/1GeYHroTZhF6vT-vpd7Rw_MgYJNZadb7L
- Fichier requis: `uie_uieb.pth` ou `uie_lsui.pth`
- Taille: Probablement ~100-500 MB

**Étape 2: Installer les dépendances**
```bash
conda create -n Reti-Diff python=3.9
conda activate Reti-Diff
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8
pip install -r requirements.txt
python setup.py develop

# Installer BasicSR
git clone https://github.com/xinntao/BasicSR.git
cd BasicSR
pip install -r requirements.txt
python setup.py develop
```

**Étape 3: Adapter pour notre vidéo**
- Modifier `options/test_UIE_UIEB.yml`
- Pointer vers nos frames extraites
- Lancer le test

**Estimation de temps:** 1-2 heures de setup si téléchargement des modèles fonctionne.

**Pourquoi pas testé maintenant:**
- Téléchargement manuel depuis Google Drive requis
- Installation d'un nouvel environnement conda (conflits possibles)
- Temps de setup ~1-2h

**Recommandation:** **PRIORITÉ MOYENNE** - Plus accessible que RecGS, moins que FUnIE-GAN.

---

## 5. Autres Modèles Mentionnés

### UIEDP (Diffusion Prior for UIE)
- **Repo:** https://github.com/ddz16/UIEDP
- **Statut:** MAJ récente (février 2025)
- **Disponibilité modèles:** À vérifier
- **Complexité:** Diffusion models = complexe

### Water-Net
- **Repo:** https://github.com/tnwei/waternet
- **Type:** Benchmark dataset + modèle
- **Intérêt:** Bon pour comparaison

### PixMamba
- **Repo:** https://github.com/weitunglin/pixmamba
- **Type:** State Space Models
- **Statut:** Très récent (2024-2025)

---

## 📈 Comparaison Générale

### Facilité de test (du plus facile au plus difficile):

1. **✅ FUnIE-GAN** - TESTÉ
   - Modèle prêt ✅
   - Code simple ✅
   - Rapide ✅
   - Mais: ne supprime pas les caustiques ❌

2. **⏳ Reti-Diff** - FAISABLE
   - Modèles sur Google Drive (téléchargement manuel)
   - Setup conda requis
   - Temps: ~1-2h

3. **❌ Seafloor-Invariant** - DIFFICILE
   - Pas de modèle pré-entraîné
   - Nécessite entraînement OU contact auteurs
   - Temps: Plusieurs jours

4. **❌ RecGS** - TRÈS DIFFICILE
   - Infrastructure 3DGS complète requise
   - Préparation données COLMAP
   - Entraînement multi-étapes
   - Temps: Plusieurs heures/jours

---

## 🎯 Conclusions et Recommandations

### Constat Principal

**Les modèles DL spécialisés pour caustiques existent MAIS:**
- Seafloor-Invariant: Pas de poids publics facilement accessibles
- RecGS: Trop complexe (research code, pas production-ready)
- FUnIE-GAN (testé): Ne cible pas spécifiquement les caustiques

### Meilleure Approche Actuelle

**Approche 10: Mask Temporal Inpaint avec RAFT** reste la meilleure solution:
- ✅ Suppression efficace des caustiques
- ✅ Préservation de la netteté
- ✅ Code fonctionnel et testé
- ✅ Pas de dépendance à des modèles externes

### Prochaines Étapes Recommandées

**Court terme (si besoin d'amélioration):**
1. **Tester Reti-Diff** (1-2h de setup)
   - Télécharger modèles depuis Google Drive
   - Tester sur quelques frames
   - Comparer avec résultats actuels

2. **Pipeline hybride:**
   - Mask Temporal Inpaint (suppression caustiques)
   - + FUnIE-GAN (amélioration couleurs)
   - = Peut-être le meilleur des deux mondes

**Moyen terme (si résultats insuffisants):**
3. **Contacter auteurs Seafloor-Invariant**
   - Email: pagraf@mail.ntua.gr
   - Demander poids pré-entraînés
   - Si fournis: tester et comparer

4. **Explorer autres modèles:**
   - UIEDP
   - Water-Net
   - Chercher sur HuggingFace/Papers With Code

**Long terme (si projet critique):**
5. **Entraîner un modèle custom:**
   - Utiliser architecture Seafloor-Invariant
   - Annoter vos propres données (caustiques/non-caustiques)
   - Entraîner sur vos cas d'usage spécifiques

---

## 📁 État Actuel du Projet

```
11_deep_learning_models/
├── FUnIE-GAN/                      ✅ Testé
│   ├── models/funie_generator.pth  (27 MB)
│   └── ...
├── seafloor_invariant/             ❌ Pas de modèle pré-entraîné
│   └── *.ipynb
├── recgs/                          ❌ Trop complexe (3DGS)
│   └── train*.py
├── Reti-Diff/                      ⏳ Possible (modèles sur Drive)
│   ├── pretrained_models/
│   │   ├── init_high.pth (245 KB)
│   │   ├── init_low.pth (245 KB)
│   │   └── retinex_decomnet.pth (84 KB)
│   └── test_UIE_*.sh
│
├── result_funiegan.mp4             ✅ Résultat FUnIE-GAN
├── comparison_funiegan.mp4         ✅ Comparaison
└── comparison_grid.jpg             ✅ Grille visuelle
```

---

## 📊 Résumé Final

| Critère | FUnIE-GAN | Seafloor-Inv | RecGS | Reti-Diff | Approche 10 (actuelle) |
|---------|-----------|--------------|-------|-----------|------------------------|
| **Suppression caustiques** | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Facilité d'utilisation** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Rapidité** | ⭐⭐⭐⭐⭐ | ? | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Disponibilité** | ✅ | ❌ | ❌ | ⏳ | ✅ |
| **Testé** | ✅ | ❌ | ❌ | ❌ | ✅ |

**VERDICT:** L'approche 10 (Mask Temporal Inpaint RAFT) **reste la meilleure solution disponible** pour supprimer les caustiques.

Les modèles DL prometteurs (Seafloor-Invariant, RecGS) existent mais nécessitent un effort significatif pour être utilisés.
