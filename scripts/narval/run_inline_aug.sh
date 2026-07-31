#!/bin/bash
# =============================================================================
# ICEMS — LOPO Step B (DBA+jitter inline) pour Narval
# Chemin projet sur Narval : ~/icems/ICEMS-main
#
# Usage :
#   cd ~/icems/ICEMS-main
#   mkdir -p logs results
#   sbatch scripts/narval/run_inline_aug.sh
#
# Override à la soumission :
#   sbatch --export=ALL,AUG_TARGET=100,OUT_DIR=results/run_aug100 scripts/narval/run_inline_aug.sh
# =============================================================================
#SBATCH --account=def-hgueziri
#SBATCH --job-name=lopo_inline
#SBATCH --output=results/lopo_inline_%j.out
#SBATCH --error=results/lopo_inline_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$HOME/icems/ICEMS-main}"
cd "$ROOT"

AUG_TARGET="${AUG_TARGET:-}"          # vide = Run 5 (classe majoritaire)
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MC_PASSES="${MC_PASSES:-30}"
MAX_FOLDS="${MAX_FOLDS:-}"
OUT_DIR="${OUT_DIR:-results/level1_inline}"
SEED="${SEED:-42}"

echo "=== LOPO inline augmentation ==="
echo "Debut : $(date)"
echo "ROOT  : $ROOT"
echo "OUT   : $OUT_DIR"
echo "AUG_TARGET : ${AUG_TARGET:-Run5-majoritaire}"

module purge
module load StdEnv/2023
module load python/3.11

# Priorité au venv du projet ; sinon pip --user (déjà installé sur ton compte)
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
else
    export PYTHONNOUSERSITE=0
fi

python -c "import tslearn; import torch; print('tslearn OK | torch', torch.__version__, 'cuda=', torch.cuda.is_available())"

if [[ ! -f data/continuous_per_trial.pkl ]]; then
    echo "ERREUR: data/continuous_per_trial.pkl manquant"
    exit 1
fi

mkdir -p "$OUT_DIR"

EXTRA=()
[[ -n "$AUG_TARGET" ]] && EXTRA+=(--aug-target "$AUG_TARGET")
[[ -n "$MAX_FOLDS" ]] && EXTRA+=(--max_folds "$MAX_FOLDS")

python src/step_B_classification.py \
    --data data/continuous_per_trial.pkl \
    --out "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch-size "$BATCH_SIZE" \
    --mc-passes "$MC_PASSES" \
    --seed "$SEED" \
    "${EXTRA[@]}"

echo "Fin : $(date)"
