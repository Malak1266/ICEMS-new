#!/bin/bash
# =============================================================================
# Soumet l'expérience extreme->intermediate sur Narval (SLURM array GPU).
#
# Usage depuis Narval :
#   cd ~/ICEMS-main
#   bash experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/slurm/submit_narval.sh
#
# Smoke (1 seed) :
#   SMOKE=1 bash experiments/.../slurm/submit_narval.sh
#
# Run A — MSE plate (protocole papier, pas HOEL) :
#   USE_HOEL=0 bash experiments/.../slurm/submit_narval.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

EXP="experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer"
SBATCH_FILE="${EXP}/slurm/run_parallel.sbatch"
mkdir -p "${EXP}/logs" "${EXP}/results"

SMOKE="${SMOKE:-0}"
USE_HOEL="${USE_HOEL:-1}"

if [[ "$SMOKE" == "1" ]]; then
    echo "[submit] Mode SMOKE — 1 seed (array task 0) USE_HOEL=${USE_HOEL}"
    JOB_ID=$(SMOKE="${SMOKE}" USE_HOEL="${USE_HOEL}" sbatch --export=ALL --array=0 "${SBATCH_FILE}" | awk '{print $NF}')
else
    echo "[submit] Run complet — 5 seeds (array 0-4) USE_HOEL=${USE_HOEL}"
    JOB_ID=$(SMOKE="${SMOKE}" USE_HOEL="${USE_HOEL}" sbatch --export=ALL "${SBATCH_FILE}" | awk '{print $NF}')
fi

echo ""
echo "============================================================"
echo " Job soumis : ${JOB_ID}"
echo " USE_HOEL   : ${USE_HOEL}  (0 = MSE plate / Run A)"
echo " Suivi      : squeue -u \$USER"
echo " Logs       : ${EXP}/logs/parallel-${JOB_ID}_*.out"
echo ""
echo " Apres completion de TOUS les tasks :"
echo "   python ${EXP}/aggregate_results.py"
echo "============================================================"
