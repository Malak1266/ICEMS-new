#!/bin/bash
# =============================================================================
# ICEMS — LOPO corrigé (nested validation anti-fuite)
#
# Soumission sur Narval :
#   cd ~/ICEMS-main
#   mkdir -p logs results
#   sbatch scripts/narval/run_lopo_corrected.sbatch
#
# Test rapide (2 folds) :
#   sbatch --export=ALL,SMOKE=1 scripts/narval/run_lopo_corrected.sbatch
# =============================================================================
#SBATCH --account=def-hgueziri
#SBATCH --job-name=lopo_corrected
#SBATCH --output=logs/lopo_corrected-%j.out
#SBATCH --error=logs/lopo_corrected-%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

AUG_TARGET="${AUG_TARGET:-200}"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-25}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MC_PASSES="${MC_PASSES:-30}"
SEED="${SEED:-42}"
SMOKE="${SMOKE:-0}"
OUT_DIR="${OUT_DIR:-results/lopo_corrected/$(date +%Y-%m-%d)}"

ROOT="${SLURM_SUBMIT_DIR:-$HOME/ICEMS-main}"
cd "$ROOT"
mkdir -p logs results

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2

source .venv/bin/activate

echo "============================================================"
echo " ICEMS — LOPO corrigé (nested validation)"
echo " Job ID    : ${SLURM_JOB_ID:-local}"
echo " Node      : ${SLURMD_NODENAME:-local}"
echo " Date      : $(date)"
echo " OUT_DIR   : $OUT_DIR"
echo " SMOKE     : $SMOKE"
echo "============================================================"

python -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"

if [[ ! -f run_lopo_corrected.py ]]; then
    echo "ERREUR: run_lopo_corrected.py introuvable — sync depuis git ou scp (voir README)."
    exit 1
fi

if [[ ! -f data/continuous_per_trial.pkl ]]; then
    echo "ERREUR: data/continuous_per_trial.pkl introuvable"
    exit 1
fi

if [[ "$SMOKE" == "1" ]]; then
    python run_lopo_corrected.py --smoke-test --out "$OUT_DIR" \
        --epochs 3 --mc-passes 3 --patience 3
else
    python run_lopo_corrected.py --full --out "$OUT_DIR" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --aug-target "$AUG_TARGET" --batch-size "$BATCH_SIZE" \
        --mc-passes "$MC_PASSES" --seed "$SEED"
fi

echo ""
echo "============================================================"
echo " Terminé — résultats dans : $OUT_DIR"
echo "   - lopo_predictions.pkl"
echo "   - metrics_summary.json"
echo "============================================================"
