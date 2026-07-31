#!/bin/bash
# =============================================================================
# ICEMS — LOPO 9 sous-niveaux (y9) + focus Senior/Expert
#
# Usage :
#   cd ~/ICEMS-main && mkdir -p logs results
#   sbatch scripts/narval/run_sublevel_lopo.sbatch
#
# Smoke test (3 folds) :
#   sbatch --export=ALL,MAX_FOLDS=3,EPOCHS=30,MC_PASSES=10,OUT_DIR=results/sublevel_smoke \
#       scripts/narval/run_sublevel_lopo.sbatch
# =============================================================================
#SBATCH --account=def-hgueziri
#SBATCH --job-name=icems_y9
#SBATCH --output=logs/icems_y9-%j.out
#SBATCH --error=logs/icems_y9-%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail
export PYTHONUNBUFFERED=1

AUG_TARGET="${AUG_TARGET:-22}"       # top-tier only : PGY6/Fellow/Staff → ~22 trials
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MC_PASSES="${MC_PASSES:-30}"
MAX_FOLDS="${MAX_FOLDS:-}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-results/sublevel_run}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-1.0}"
TOP_TIER_WEIGHT="${TOP_TIER_WEIGHT:-1.0}"
NO_INLINE_AUG="${NO_INLINE_AUG:-0}"

ROOT="${SLURM_SUBMIT_DIR:-$HOME/ICEMS-main}"
cd "$ROOT"
mkdir -p logs results

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
source .venv/bin/activate

echo "============================================================"
echo " ICEMS Step B — LOPO 9 sous-niveaux"
echo " Job ID    : ${SLURM_JOB_ID:-local}"
echo " Node      : ${SLURMD_NODENAME:-local}"
echo " Date      : $(date)"
echo " OUT_DIR   : $OUT_DIR"
echo " AUG_TARGET: $AUG_TARGET (PGY6/Fellow/Staff uniquement)"
echo " EPOCHS    : $EPOCHS"
echo " MAX_FOLDS : ${MAX_FOLDS:-47 (all)}"
echo "============================================================"

python -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"

[[ -f data/continuous_per_trial.pkl ]] || { echo "ERREUR: data/continuous_per_trial.pkl manquant"; exit 1; }

EXTRA=()
[[ -n "$MAX_FOLDS" ]] && EXTRA+=(--max_folds "$MAX_FOLDS")
[[ "$NO_INLINE_AUG" == "1" ]] && EXTRA+=(--no-inline-aug)

python -u src/step_B_classification.py \
    --data data/continuous_per_trial.pkl \
    --out "$OUT_DIR" \
    --aug-target "$AUG_TARGET" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch-size "$BATCH_SIZE" \
    --mc-passes "$MC_PASSES" \
    --anchor-weight "$ANCHOR_WEIGHT" \
    --top-tier-weight "$TOP_TIER_WEIGHT" \
    --seed "$SEED" \
    "${EXTRA[@]}"

echo ""
echo "============================================================"
echo " ✅ Terminé — $OUT_DIR"
echo "   lopo_predictions.pkl"
echo "   confusion_matrix_4class.png / confusion_matrix_9sublevels.png"
echo "   senior_expert_scores.png"
echo "   sublevel/icems_lopo_violins_9sublevels.png"
echo "   diagnostic/diagnostic_fellow_vs_staff.png"
echo "============================================================"
