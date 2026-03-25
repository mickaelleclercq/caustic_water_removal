#!/bin/bash
# Script d'installation et test de Reti-Diff pour Underwater Image Enhancement
# Ce script guide l'utilisateur à travers le processus de setup de Reti-Diff

set -e

echo "================================================================="
echo "  Reti-Diff - Setup pour Underwater Image Enhancement"
echo "================================================================="
echo ""

echo "⚠️  ATTENTION: Ce script nécessite:"
echo "   1. Téléchargement manuel de modèles depuis Google Drive (~500 MB)"
echo "   2. Installation d'un nouvel environnement conda"
echo "   3. Installation de BasicSR (bibliothèque tierce)"
echo "   4. Temps estimé: 1-2 heures"
echo ""
echo "❓ Voulez-vous continuer? (y/n)"
read -r response

if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Installation annulée."
    exit 0
fi

RETI_DIR="/home/mickael/damien/11_deep_learning_models/Reti-Diff"
cd "$RETI_DIR"

echo ""
echo "================================================================="
echo "ÉTAPE 1: Téléchargement des modèles pré-entraînés"
echo "================================================================="
echo ""
echo "Les modèles doivent être téléchargés manuellement depuis:"
echo "https://drive.google.com/drive/folders/1GeYHroTZhF6vT-vpd7Rw_MgYJNZadb7L"
echo ""
echo "Fichiers requis pour UIE (Underwater Image Enhancement):"
echo "  - uie_uieb.pth"
echo "  - uie_lsui.pth"
echo ""
echo "Placez-les dans: $RETI_DIR/pretrained_models/"
echo ""
echo "Appuyez sur ENTRÉE quand c'est fait..."
read -r

# Vérifier si les modèles sont là
if [ ! -f "pretrained_models/uie_uieb.pth" ] && [ ! -f "pretrained_models/uie_lsui.pth" ]; then
    echo "❌ ERREUR: Aucun modèle UIE trouvé dans pretrained_models/"
    echo "   Veuillez télécharger au moins un des modèles UIE."
    exit 1
fi

echo "✅ Modèles trouvés!"

echo ""
echo "================================================================="
echo "ÉTAPE 2: Installation de l'environnement conda"
echo "================================================================="
echo ""
echo "⚠️  Ceci va créer un nouvel environnement conda 'Reti-Diff'"
echo "   Continuer? (y/n)"
read -r response

if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Installation annulée."
    exit 0
fi

# Créer l'environnement
conda create -n Reti-Diff python=3.9 -y
conda activate Reti-Diff

# Installer PyTorch
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Installer les dépendances
pip install numpy==1.25.2
pip install -r requirements.txt
python setup.py develop

echo "✅ Environnement Reti-Diff installé!"

echo ""
echo "================================================================="
echo "ÉTAPE 3: Installation de BasicSR"
echo "================================================================="
echo ""

cd /home/mickael/damien/11_deep_learning_models

if [ -d "BasicSR" ]; then
    echo "BasicSR déjà cloné, mise à jour..."
    cd BasicSR
    git pull
else
    echo "Clonage de BasicSR..."
    git clone https://github.com/xinntao/BasicSR.git
    cd BasicSR
fi

pip install tb-nightly
pip install -r requirements.txt
python setup.py develop

echo "✅ BasicSR installé!"

echo ""
echo "================================================================="
echo "ÉTAPE 4: Préparation des données de test"
echo "================================================================="
echo ""

cd "$RETI_DIR"

# Créer dossier pour les données de test
mkdir -p Datasets/UIE_test/input
mkdir -p Datasets/UIE_test/output

echo "Extraction de frames de test depuis la vidéo..."
python3 << 'EOF'
import cv2
import os

video_path = "/home/mickael/damien/01_extraction/subclip_5s.mp4"
output_dir = "/home/mickael/damien/11_deep_learning_models/Reti-Diff/Datasets/UIE_test/input"

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Extraire 10 frames uniformément réparties
frame_indices = [int(i * total_frames / 10) for i in range(10)]

for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f"{output_dir}/frame_{idx:04d}.png", frame)
        print(f"Frame {idx} extraite")

cap.release()
print(f"\n✅ {len(frame_indices)} frames extraites dans {output_dir}")
EOF

echo ""
echo "================================================================="
echo "ÉTAPE 5: Modification du fichier de configuration"
echo "================================================================="
echo ""

# Créer un fichier de config personnalisé
cat > options/test_UIE_custom.yml << 'EOFCONFIG'
# general settings
name: UIE_custom
model_type: S2_Interface_Model
scale: 1
num_gpu: auto
manual_seed: 0

# dataset and data loader settings
datasets:
  val_1:
    name: CustomTestset
    type: DeblurPairedDataset
    dataroot_gt: Datasets/UIE_test/input  # Same as input for testing
    dataroot_lq: Datasets/UIE_test/input  # Our frames
    dataset_type: chaos
    gt_size: 256
    io_backend:
      type: disk

# network structures
network_g:
  type: RetiDiffS2_Interface
  n_encoder_res: 5
  inp_channels: 3
  out_channels: 3
  dim: 64
  num_blocks: [3,3,3,3]
  num_refinement_blocks: 3
  heads: [1,2,4,8]
  ffn_expansion_factor: 2.66
  bias: False
  LayerNorm_type: WithBias
  n_denoise_res: 1
  linear_start: 0.1
  linear_end: 0.99
  timesteps: 4

# path
path:
  pretrain_network_g: ./pretrained_models/uie_uieb.pth
  param_key_g: params_ema
  strict_load_g: False

pretrain_decomnet_low: pretrained_models/retinex_decomnet.pth

val:
  window_size: 8
  save_img: True
  suffix: ~
EOFCONFIG

echo "✅ Fichier de configuration créé: options/test_UIE_custom.yml"

echo ""
echo "================================================================="
echo "ÉTAPE 6: Test du modèle"
echo "================================================================="
echo ""
echo "Lancement du test..."

CUDA_VISIBLE_DEVICES=0 python3 Reti-Diff/test.py -opt options/test_UIE_custom.yml

echo ""
echo "================================================================="
echo "✅ INSTALLATION ET TEST TERMINÉS!"
echo "================================================================="
echo ""
echo "Résultats sauvegardés dans:"
echo "  $RETI_DIR/results/"
echo ""
echo "Pour utiliser Reti-Diff à l'avenir:"
echo "  conda activate Reti-Diff"
echo "  cd $RETI_DIR"
echo "  CUDA_VISIBLE_DEVICES=0 python3 Reti-Diff/test.py -opt options/test_UIE_custom.yml"
echo ""
