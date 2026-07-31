#!/bin/bash
# =============================================================================
# ICEMS — LOPO 9 sous-niveaux v2
#   Mod 2 : epochs=150, patience=25
#   Mod 3 : tête binaire Staff (staff-weight)
#   Mod 4 : features cinématiques dérivées (13 canaux)
#   Graphes : sublevels_time/score_vs_time_{MS,PGY1,...,Staff}.png (9 fichiers)
#
# Usage :
#   cd ~/ICEMS-main && mkdir -p logs results
#   sbatch scripts/narval/run_sublevel_lopo_v2.sbatch
#
# Smoke test (3 folds) :
#   sbatch --export=ALL,MAX_FOLDS=3,EPOCHS=40,MC_PASSES=10,OUT_DIR=results/sublevel_smoke_v2 \
#       scripts/narval/run_sublevel_lopo_v2.sbatch
# =============================================================================
#SBATCH --account=def-hgueziri
#SBATCH --job-name=icems_y9_v2
#SBATCH --output=logs/icems_y9_v2-%j.out
#SBATCH --error=logs/icems_y9_v2-%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail
export PYTHONUNBUFFERED=1

AUG_TARGET="${AUG_TARGET:-22}"
EPOCHS="${EPOCHS:-150}"
PATIENCE="${PATIENCE:-25}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MC_PASSES="${MC_PASSES:-30}"
MAX_FOLDS="${MAX_FOLDS:-}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-results/sublevel_run_v2}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-1.0}"
TOP_TIER_WEIGHT="${TOP_TIER_WEIGHT:-1.0}"
STAFF_WEIGHT="${STAFF_WEIGHT:-1.0}"
STAFF_POS_WEIGHT="${STAFF_POS_WEIGHT:-5.0}"
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
echo " ICEMS Step B — LOPO 9 sous-niveaux v2"
echo " Job ID    : ${SLURM_JOB_ID:-local}"
echo " Node      : ${SLURMD_NODENAME:-local}"
echo " Date      : $(date)"
echo " OUT_DIR   : $OUT_DIR"
echo " AUG_TARGET: $AUG_TARGET (PGY6/Fellow/Staff uniquement)"
echo " EPOCHS    : $EPOCHS  |  PATIENCE : $PATIENCE"
echo " STAFF_W   : $STAFF_WEIGHT (pos_weight=$STAFF_POS_WEIGHT)"
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
    --staff-weight "$STAFF_WEIGHT" \
    --staff-pos-weight "$STAFF_POS_WEIGHT" \
    --seed "$SEED" \
    "${EXTRA[@]}"

echo ""
echo "============================================================"
echo " ✅ Terminé — $OUT_DIR"
echo "   lopo_predictions.pkl"
echo "   sublevels_time/score_vs_time_*.png  (9 graphes détaillés)"
echo "   score_vs_time_9sublevels.png / score_vs_time_9sublevels_grid.png"
echo "   confusion_matrix_4class.png / confusion_matrix_9sublevels.png"
echo "   senior_expert_scores.png"
echo "============================================================"
