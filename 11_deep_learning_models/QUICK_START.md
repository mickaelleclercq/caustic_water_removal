# Quick Start - Modèles DL pour Caustiques

## TL;DR - Réponse à votre question

**✅ OUI, il existe des modèles deep learning spécialisés pour enlever les caustiques !**

## Les 2 meilleurs modèles (code disponible)

### 1. Seafloor-Invariant Caustics Removal (2023) ⭐ RECOMMANDÉ
- **Repo:** https://github.com/pagraf/Seafloor-type-Invariant-Removal-of-Caustics-from-Underwater-Imagery
- **Statut:** Code dispo, MAJ il y a 1 jour
- **Pourquoi:** Conçu exactement pour votre cas (eaux peu profondes, fond corallien/sableux)

### 2. RecGS - Recurrent Gaussian Splatting (2024) 🚀
- **Repo:** https://github.com/tyz1030/recgs
- **Statut:** Code dispo, MAJ il y a 4 jours
- **Pourquoi:** Approche 3D temporelle ultra-moderne, parfait pour vos 2× A100

## Installation rapide

```bash
cd 11_deep_learning_models
./setup_dl_models.sh
```

## Documentation complète

- **Détails complets:** [MODELES_DL_CAUSTIQUES.md](../MODELES_DL_CAUSTIQUES.md)
- **Plan mis à jour:** [PLAN.md](../PLAN.md) - Approche F ajoutée en priorité

## Stratégie recommandée

1. **Tester Seafloor-Invariant d'abord** (plus simple, spécialisé)
2. Si prometteur → **Tester RecGS** (plus avancé, 3D)
3. **Comparer** avec vos approches RAFT/Homographie (approches A-E)
4. **Combiner** : utiliser DL pour détection + vos méthodes temporelles pour suppression

## Pourquoi ces modèles peuvent fonctionner

- Ils sont **spécialisés** pour les caustiques (pas juste amélioration générale)
- **Entraînés** sur des données sous-marines réelles
- **Validés** par des publications scientifiques récentes (2023-2024)
- **Code maintenu** activement (mises à jour février 2025)

## Prochaine étape suggérée

Au lieu de continuer avec les approches A-E du plan original, **tester d'abord Seafloor-Invariant**. 
Cela peut vous faire gagner beaucoup de temps si ça fonctionne bien !
