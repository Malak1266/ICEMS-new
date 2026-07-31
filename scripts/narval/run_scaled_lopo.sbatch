#!/bin/bash
# =============================================================================
# ICEMS — LOPO Step B avec DBA+jitter scaled (--aug-target)
#
# Soumission sur Narval :
#   cd ~/ICEMS-main
#   mkdir -p logs results
#   sbatch scripts/narval/run_scaled_lopo.sbatch
#
# Variables optionnelles (override à la soumission) :
#   sbatch --export=ALL,AUG_TARGET=100,MAX_FOLDS=5 scripts/narval/run_scaled_lopo.sbatch
# =============================================================================
#SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT    # ex. def-abc12 ou rrg-xyz45
#SBATCH --job-name=icems_scaled
#SBATCH --output=logs/icems_scaled-%j.out
#SBATCH --error=logs/icems_scaled-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

# ── Paramètres du run (modifiables via sbatch --export) ───────────────────
AUG_TARGET="${AUG_TARGET:-100}"      # 100 = test ; 200 = ~800 trials/fold
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MC_PASSES="${MC_PASSES:-30}"
MAX_FOLDS="${MAX_FOLDS:-}"           # vide = 47 folds LOPO complets
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-results/run_scaled_aug${AUG_TARGET}}"

# ── Environnement ───────────────────────────────────────────────────────────
ROOT="${SLURM_SUBMIT_DIR:-$HOME/ICEMS-main}"
cd "$ROOT"
mkdir -p logs results

module purge
module load StdEnv/2023
module load python/3.11
# module load cuda/12.2   # décommenter si PyTorch CUDA installé

source .venv/bin/activate

echo "============================================================"
echo " ICEMS Step B — Narval"
echo " Job ID    : ${SLURM_JOB_ID:-local}"
echo " Node      : ${SLURMD_NODENAME:-local}"
echo " Date      : $(date)"
echo " ROOT      : $ROOT"
echo " AUG_TARGET: $AUG_TARGET"
echo " EPOCHS    : $EPOCHS"
echo " OUT_DIR   : $OUT_DIR"
echo " MAX_FOLDS : ${MAX_FOLDS:-all}"
echo "============================================================"

python -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"

if [[ ! -f data/continuous_per_trial.pkl ]]; then
    echo "ERREUR: data/continuous_per_trial.pkl introuvable dans $ROOT/data/"
    exit 1
fi

# ── Lancement ─────────────────────────────────────────────────────────────
EXTRA_FOLDS=()
if [[ -n "$MAX_FOLDS" ]]; then
    EXTRA_FOLDS=(--max_folds "$MAX_FOLDS")
fi

python src/step_B_classification.py \
    --data data/continuous_per_trial.pkl \
    --out "$OUT_DIR" \
    --aug-target "$AUG_TARGET" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch-size "$BATCH_SIZE" \
    --mc-passes "$MC_PASSES" \
    --seed "$SEED" \
    "${EXTRA_FOLDS[@]}"

echo ""
echo "============================================================"
echo " ✅ Terminé — résultats dans : $OUT_DIR"
echo "   - lopo_predictions.pkl"
echo "   - confusion_matrix.png"
echo "   - scatter.png"
echo "   - score_vs_time.png"
echo "============================================================"
