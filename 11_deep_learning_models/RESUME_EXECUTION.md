# ✅ TESTS DES MODÈLES DEEP LEARNING - RÉSUMÉ

Date: 25 mars 2026

## Ce qui a été fait

### 1. Installation
- ✅ Clonage de 3 repos GitHub:
  - FUnIE-GAN (avec modèle pré-entraîné)
  - Seafloor-Invariant Caustics Removal
  - RecGS (Recurrent Gaussian Splatting)

### 2. Tests effectués

#### FUnIE-GAN ⚡
- **Statut:** ✅ TESTÉ avec succès
- **Performance:** 405 FPS sur GPU (très rapide)
- **Résultat:** Améliore couleurs et contraste, mais NE SUPPRIME PAS les caustiques

**Fichiers générés:**
- `result_funiegan.mp4` (108 MB) - Vidéo complète traitée
- `comparison_funiegan.mp4` (410 MB) - Comparaison côte à côte
- `comparison_grid.jpg` (3.1 MB) - Grille de 5 frames

#### Seafloor-Invariant
- **Statut:** ⏳ NON testé
- **Raison:** Pas de modèle pré-entraîné fourni
- **Pour tester:** Faudrait entraîner le modèle ou contacter les auteurs

#### RecGS
- **Statut:** ⏳ NON testé
- **Raison:** Setup complexe (nécessite infrastructure Gaussian Splatting)
- **Pour tester:** Plusieurs heures/jours de travail

## 📊 Résultats FUnIE-GAN

### Points positifs ✅
- Très rapide (temps réel possible: 405 FPS)
- Améliore les couleurs (corrige la dominante bleue/verte)
- Améliore le contraste général
- Code prêt à l'emploi avec modèle pré-entraîné

### Limitations ⚠️
- **Les caustiques restent clairement visibles**
- Pas de suppression spécifique des motifs lumineux
- Augmentation du bruit dans certaines zones
- Modèle générique (pas spécialisé caustiques)

## 🎯 Conclusion

### Réponse à la question "Modèles DL pour supprimer les caustiques ?"

**OUI, ils existent** (Seafloor-Invariant, RecGS) **MAIS:**
- Seafloor-Invariant: pas de poids pré-entraînés facilement accessibles
- RecGS: setup trop complexe pour un test rapide
- FUnIE-GAN (testé): amélioration générale mais PAS de suppression de caustiques

### Meilleure approche actuelle

**Approche 10: Mask Temporal Inpaint avec RAFT**
- Suppression efficace des caustiques ✅
- Préservation de la netteté ✅
- Déjà testé et fonctionnel ✅

### Approche hybride possible

Option 1: RAFT Mask Inpaint + FUnIE-GAN
1. Supprimer caustiques avec RAFT (approche 10)
2. Améliorer couleurs avec FUnIE-GAN
= Meilleur des deux mondes ?

Option 2: FUnIE-GAN + RAFT Mask Inpaint
1. Améliorer couleurs avec FUnIE-GAN
2. Supprimer caustiques avec RAFT
= Ordre inversé

### Recommandation finale

**Pour l'instant:** Utiliser **Mask Temporal Inpaint RAFT** (dossier 10)

**Si besoin d'amélioration couleurs:** Ajouter FUnIE-GAN avant ou après

**Pour aller plus loin avec DL:**
- Contacter auteurs de Seafloor-Invariant pour les poids
- Ou chercher d'autres modèles avec poids disponibles (Reti-Diff, UIEDP)
- Ou entraîner un modèle custom

## 📁 Où trouver les résultats

```
11_deep_learning_models/
├── result_funiegan.mp4           ← Vidéo traitée par FUnIE-GAN
├── comparison_funiegan.mp4       ← Comparaison Original vs FUnIE-GAN
├── comparison_grid.jpg           ← Grille de comparaison visuelle
├── RESULTATS.md                  ← Documentation détaillée
├── process_video_funiegan.py     ← Script de traitement
└── [autres fichiers et repos clonés]
```

## ⏭️ Prochaines étapes

Si les résultats de l'approche 10 (Mask Temporal Inpaint) ne sont pas satisfaisants:
1. Tester l'approche hybride RAFT + FUnIE-GAN
2. Contacter les auteurs de Seafloor-Invariant
3. Explorer d'autres modèles DL (Reti-Diff, UIEDP)
4. Considérer l'entraînement d'un modèle custom

Si les résultats sont satisfaisants:
- Optimiser les paramètres
- Tester sur la vidéo complète (pas juste 5s)
- Documenter le pipeline final
