#!/bin/bash
# =============================================================================
# ICEMS — Entrainement multi-tache (y4 + y9) sur Narval
#
# Soumission :
#   cd ~/ICEMS-main
#   mkdir -p logs results/clf_multitask
#   sbatch scripts/narval/run_clf_multitask.sbatch
#
# Smoke test (2 folds, 3 epochs) :
#   sbatch --export=ALL,SMOKE=1,OUT_DIR=results/clf_multitask/smoke \
#       scripts/narval/run_clf_multitask.sbatch
#
# Run complet LOPO (47 folds) :
#   sbatch --time=48:00:00 --export=ALL,OUT_DIR=results/clf_multitask/full \
#       scripts/narval/run_clf_multitask.sbatch
#
# Sans augmentation y9 :
#   sbatch --export=ALL,NO_AUG=1,OUT_DIR=results/clf_multitask/no_aug \
#       scripts/narval/run_clf_multitask.sbatch
# =============================================================================
#SBATCH --account=def-hgueziri
#SBATCH --job-name=icems_clf_mt
#SBATCH --output=logs/icems_clf_mt-%j.out
#SBATCH --error=logs/icems_clf_mt-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

export PYTHONUNBUFFERED=1

# ── Parametres (override via sbatch --export) ───────────────────────────────
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ALPHA="${ALPHA:-0.4}"
SEED="${SEED:-42}"
MAX_FOLDS="${MAX_FOLDS:-}"
OUT_DIR="${OUT_DIR:-results/clf_multitask/$(date +%Y-%m-%d)_full}"
SMOKE="${SMOKE:-0}"
NO_AUG="${NO_AUG:-0}"

# ── Environnement ───────────────────────────────────────────────────────────
ROOT="${SLURM_SUBMIT_DIR:-$HOME/ICEMS-main}"
cd "$ROOT"
mkdir -p logs results/clf_multitask

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2

source .venv/bin/activate

echo "============================================================"
echo " ICEMS Multi-task (y4 + y9) — Narval"
echo " Job ID    : ${SLURM_JOB_ID:-local}"
echo " Node      : ${SLURMD_NODENAME:-local}"
echo " Date      : $(date)"
echo " ROOT      : $ROOT"
echo " EPOCHS    : $EPOCHS"
echo " PATIENCE  : $PATIENCE"
echo " BATCH_SIZE: $BATCH_SIZE"
echo " ALPHA     : $ALPHA"
echo " OUT_DIR   : $OUT_DIR"
echo " SMOKE     : $SMOKE"
echo " NO_AUG    : $NO_AUG"
echo " MAX_FOLDS : ${MAX_FOLDS:-all}"
echo "============================================================"

python -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"

for f in data/continuous_per_trial.pkl data/filtered_data.json; do
    if [[ ! -f "$f" ]]; then
        echo "ERREUR: $f introuvable dans $ROOT/"
        exit 1
    fi
done

# ── Arguments ───────────────────────────────────────────────────────────────
ARGS=(--out-dir "$OUT_DIR" --epochs "$EPOCHS" --patience "$PATIENCE" \
      --batch-size "$BATCH_SIZE" --alpha "$ALPHA" --seed "$SEED")

if [[ "$SMOKE" == "1" ]]; then
    ARGS=(--smoke-test "${ARGS[@]}")
else
    ARGS=(--full "${ARGS[@]}")
fi

if [[ -n "$MAX_FOLDS" ]]; then
    ARGS+=(--max-folds "$MAX_FOLDS")
fi

if [[ "$NO_AUG" == "1" ]]; then
    ARGS+=(--no-aug)
fi

# ── Lancement ───────────────────────────────────────────────────────────────
python -u src/train_clf_multitask.py "${ARGS[@]}"

echo ""
echo "============================================================"
echo " Termine — resultats dans : $OUT_DIR"
echo "   - predictions.csv"
echo "   - metrics.json"
echo "============================================================"
