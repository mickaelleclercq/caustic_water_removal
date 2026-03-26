#!/bin/bash
# Lancement parallèle des 4 méthodes GPU sur GX010236_synced_enhanced.MP4
#
# Répartition GPU :
#   GPU0 : Méthode A (12_gpu_homography)  +  Méthode D (13_gpu_lowpass)
#   GPU1 : Méthode E (14_gpu_mask_inpaint) +  Méthode J (15_gpu_pyramid_J)
#
# Les 4 processus tournent simultanément ; les logs sont dans /tmp/gpu_runs/.
#
# Usage :
#   bash run_gpu_all.sh           # pleine vidéo
#   bash run_gpu_all.sh --test    # test-only (3 frames témoins, rapide)

set -e
cd "$(dirname "$0")"
source myenv/bin/activate

MODE=""
if [[ "$1" == "--test" ]]; then
    MODE="--test-only"
    echo "=== Mode test-only (3 frames) ==="
else
    echo "=== Mode pleine vidéo (1161 frames, 4K) ==="
fi

LOG_DIR=/tmp/gpu_runs
mkdir -p "$LOG_DIR"

echo ""
echo "Étape 0 : Précomputation homographies partagées (half=4, ~3.5 min)…"
PYTHONUNBUFFERED=1 python -u precompute_homographies.py 2>&1 | tee "$LOG_DIR/precompute.log"
echo "Homographies précomputées → chargement instantané dans les 4 méthodes."
echo ""
echo "GPU0 ← Méthode A (homographie+médiane N=5)  +  Méthode D (lowpass N=9)"
echo "GPU1 ← Méthode E (masque+médiane N=7)       +  Méthode J (pyramide L=4 N=9)"
echo "Logs : $LOG_DIR/"
echo ""

# Lancement des 4 processus en arrière-plan
PYTHONUNBUFFERED=1 python -u 12_gpu_homography/process.py --gpu 0 $MODE \
    2>&1 | tee "$LOG_DIR/method_A.log" &
PID_A=$!

PYTHONUNBUFFERED=1 python -u 13_gpu_lowpass/process.py --gpu 0 $MODE \
    2>&1 | tee "$LOG_DIR/method_D.log" &
PID_D=$!

PYTHONUNBUFFERED=1 python -u 14_gpu_mask_inpaint/process.py --gpu 1 $MODE \
    2>&1 | tee "$LOG_DIR/method_E.log" &
PID_E=$!

PYTHONUNBUFFERED=1 python -u 15_gpu_pyramid_J/process.py --gpu 1 $MODE \
    2>&1 | tee "$LOG_DIR/method_J.log" &
PID_J=$!

echo "PIDs lancés : A=$PID_A  D=$PID_D  E=$PID_E  J=$PID_J"
echo ""
echo "Suivi en temps réel :"
echo "  tail -f $LOG_DIR/method_A.log"
echo "  tail -f $LOG_DIR/method_D.log"
echo "  tail -f $LOG_DIR/method_E.log"
echo "  tail -f $LOG_DIR/method_J.log"
echo ""

# Attente de tous les processus et rapport des codes de retour
FAILED=0
wait $PID_A && echo "✓ Méthode A terminée" || { echo "✗ Méthode A ÉCHOUÉE (exit $?)"; FAILED=1; }
wait $PID_D && echo "✓ Méthode D terminée" || { echo "✗ Méthode D ÉCHOUÉE (exit $?)"; FAILED=1; }
wait $PID_E && echo "✓ Méthode E terminée" || { echo "✗ Méthode E ÉCHOUÉE (exit $?)"; FAILED=1; }
wait $PID_J && echo "✓ Méthode J terminée" || { echo "✗ Méthode J ÉCHOUÉE (exit $?)"; FAILED=1; }

echo ""
if [[ $FAILED -eq 0 ]]; then
    echo "=== Toutes les méthodes terminées avec succès ==="
    echo ""
    echo "Résultats :"
    ls -lh 12_gpu_homography/result_A_gpu_4k_N5.mp4  2>/dev/null || true
    ls -lh 13_gpu_lowpass/result_D_gpu_4k_N9.mp4      2>/dev/null || true
    ls -lh 14_gpu_mask_inpaint/result_E_gpu_4k_N7.mp4 2>/dev/null || true
    ls -lh 15_gpu_pyramid_J/result_J_gpu_4k_N9.mp4    2>/dev/null || true
else
    echo "=== Certaines méthodes ont échoué — vérifiez les logs dans $LOG_DIR/ ==="
    exit 1
fi
