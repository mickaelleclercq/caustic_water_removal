#!/bin/bash

# Script d'installation des modèles deep learning pour suppression de caustiques
# Usage: ./setup_dl_models.sh

set -e  # Arrêt en cas d'erreur

echo "========================================="
echo "Installation des modèles DL - Caustiques"
echo "========================================="
echo ""

# Vérifier qu'on est dans le bon répertoire
if [ ! -f "../PLAN.md" ]; then
    echo "ERREUR: Ce script doit être exécuté depuis 11_deep_learning_models/"
    exit 1
fi

# Activer l'environnement virtuel
echo "Activation de l'environnement virtuel..."
source ../myenv/bin/activate

# Créer les sous-dossiers
echo ""
echo "Création des dossiers..."
mkdir -p seafloor_invariant recgs

# Cloner Seafloor-Invariant
echo ""
echo "========================================="
echo "1. Seafloor-Invariant Caustics Removal"
echo "========================================="
if [ -d "seafloor_invariant/.git" ]; then
    echo "Le repo existe déjà, pull des dernières modifications..."
    cd seafloor_invariant
    git pull
    cd ..
else
    echo "Clonage du repo..."
    git clone https://github.com/pagraf/Seafloor-type-Invariant-Removal-of-Caustics-from-Underwater-Imagery.git seafloor_invariant
fi

# Installer dépendances Seafloor-Invariant
cd seafloor_invariant
if [ -f "requirements.txt" ]; then
    echo "Installation des dépendances..."
    pip install -r requirements.txt
else
    echo "Pas de requirements.txt trouvé, installation manuelle nécessaire"
    echo "Dépendances typiques: opencv-python, numpy, scipy, matplotlib, pillow"
fi
cd ..

# Cloner RecGS
echo ""
echo "========================================="
echo "2. RecGS - Recurrent Gaussian Splatting"
echo "========================================="
if [ -d "recgs/.git" ]; then
    echo "Le repo existe déjà, pull des dernières modifications..."
    cd recgs
    git pull
    cd ..
else
    echo "Clonage du repo..."
    git clone https://github.com/tyz1030/recgs.git
fi

# Installer dépendances RecGS
cd recgs
if [ -f "requirements.txt" ]; then
    echo "Installation des dépendances..."
    pip install -r requirements.txt
else
    echo "Pas de requirements.txt trouvé, installation manuelle nécessaire"
    echo "Dépendances typiques: torch, numpy, opencv-python, matplotlib"
fi
cd ..

echo ""
echo "========================================="
echo "Installation terminée !"
echo "========================================="
echo ""
echo "Prochaines étapes:"
echo "1. Lire les README dans chaque dossier:"
echo "   - seafloor_invariant/README.md"
echo "   - recgs/README.md"
echo ""
echo "2. Préparer une frame de test:"
echo "   python ../01_extraction/extract.py"
echo ""
echo "3. Tester Seafloor-Invariant d'abord (plus simple)"
echo ""
echo "4. Si prometteur, tester RecGS (plus avancé)"
echo ""
echo "Bon courage !"
