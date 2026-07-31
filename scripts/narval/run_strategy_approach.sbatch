#!/bin/bash
# =============================================================================
# ICEMS — Stratégie expérimentale (approches 1–4, une variable à la fois)
#
# Étape 0 (local ou Narval) :
#   python src/diagnose_tsne_baseline.py
#
# Smoke test approche 1 :
#   sbatch --export=ALL,APPROACH=1,SMOKE=1 scripts/narval/run_strategy_approach.sbatch
#
# Run complet approche 2 :
#   sbatch --export=ALL,APPROACH=2 scripts/narval/run_strategy_approach.sbatch
# =============================================================================
#SBATCH --account=def-hgueziri
#SBATCH --job-name=icems_strategy
#SBATCH --output=logs/strategy-%j.out
#SBATCH --error=logs/strategy-%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

APPROACH="${APPROACH:-1}"
SMOKE="${SMOKE:-0}"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-25}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MC_PASSES="${MC_PASSES:-30}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-}"

ROOT="${SLURM_SUBMIT_DIR:-$HOME/ICEMS-main}"
cd "$ROOT"
mkdir -p logs results

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2

source .venv/bin/activate

echo "============================================================"
echo " ICEMS — Stratégie approche ${APPROACH}"
echo " Job ID    : ${SLURM_JOB_ID:-local}"
echo " SMOKE     : ${SMOKE}"
echo "============================================================"

python -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"

if [[ ! -f data/continuous_per_trial.pkl ]]; then
    echo "ERREUR: data/continuous_per_trial.pkl introuvable"
    exit 1
fi

EXTRA=()
[[ -n "$OUT_DIR" ]] && EXTRA+=(--out "$OUT_DIR")

if [[ "$SMOKE" == "1" ]]; then
    python run_lopo_corrected.py --smoke-test --approach "$APPROACH" \
        --epochs 3 --mc-passes 3 --patience 3 "${EXTRA[@]}"
else
    python run_lopo_corrected.py --full --approach "$APPROACH" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --batch-size "$BATCH_SIZE" --mc-passes "$MC_PASSES" \
        --seed "$SEED" "${EXTRA[@]}"
fi

echo ""
echo "============================================================"
echo " Terminé — approche ${APPROACH}"
echo "============================================================"
