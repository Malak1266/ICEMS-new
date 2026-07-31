#!/bin/bash
# =============================================================================
# ICEMS — Hybrid LSTM-Transformer LOPO : comparaison weighted_mse vs hierarchical
#
# Deux bras identiques (données, archi, seed) — seule la loss change.
#
# Smoke test (local ou Narval) :
#   SMOKE=1 bash slurm/run_hybrid.sh
#
# Run complet Narval :
#   sbatch slurm/run_hybrid.sh
# =============================================================================
#SBATCH --account=def-hgueziri
#SBATCH --job-name=icems_hybrid
#SBATCH --output=logs/hybrid-%j.out
#SBATCH --error=logs/hybrid-%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

set -euo pipefail
export PYTHONUNBUFFERED=1

SMOKE="${SMOKE:-0}"
SEED="${SEED:-42}"
ROOT="${SLURM_SUBMIT_DIR:-$HOME/icems/ICEMS-main}"
cd "$ROOT"
mkdir -p logs results

module purge 2>/dev/null || true
module load StdEnv/2023 2>/dev/null || true
module load python/3.11 2>/dev/null || true
module load cuda/12.2 2>/dev/null || true

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

echo "============================================================"
echo " ICEMS Hybrid LOPO — weighted_mse + hierarchical"
echo " Job ID  : ${SLURM_JOB_ID:-local}"
echo " Node    : ${SLURMD_NODENAME:-local}"
echo " SMOKE   : ${SMOKE}"
echo " SEED    : ${SEED}"
echo " Date    : $(date)"
echo "============================================================"

python -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"

[[ -f data/continuous_per_trial.pkl ]] || { echo "ERREUR: data/continuous_per_trial.pkl manquant"; exit 1; }
[[ -f "data/Exvivo_trial_Participants(Sheet1).csv" ]] || { echo "ERREUR: CSV participants manquant"; exit 1; }

SMOKE_FLAG=()
if [[ "$SMOKE" == "1" ]]; then
    SMOKE_FLAG=(--smoke)
fi

for LOSS in weighted_mse hierarchical; do
    OUT="results/hybrid_${LOSS}"
    echo ""
    echo "--- Bras: --loss ${LOSS} → ${OUT} ---"
    python -u src/train/train_hybrid_lopo.py \
        --loss "${LOSS}" \
        --out "${OUT}" \
        --seed "${SEED}" \
        "${SMOKE_FLAG[@]}"
done

echo ""
echo "============================================================"
echo " Terminé — résultats :"
echo "   results/hybrid_weighted_mse/"
echo "   results/hybrid_hierarchical/"
echo "============================================================"
