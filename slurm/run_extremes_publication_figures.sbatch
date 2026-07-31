#!/bin/bash
# =============================================================================
# Figures publication — TRAIN extrêmes → TEST milieu
# Prérequis : results/hybrid_extremes/ avec model_best.pt, predictions_*.pkl
# =============================================================================
#SBATCH --account=def-hgueziri
#SBATCH --job-name=extremes_figures
#SBATCH --output=logs/extremes_figures-%j.out
#SBATCH --error=logs/extremes_figures-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT="${SLURM_SUBMIT_DIR:-$HOME/ICEMS-main}"
RUN_DIR="${RUN_DIR:-results/hybrid_extremes}"
cd "$ROOT"
mkdir -p logs

module purge 2>/dev/null || true
module load StdEnv/2023 2>/dev/null || true
module load python/3.11 2>/dev/null || true

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

echo "=== Extraction per-frame (extremes) ==="
python -m eval.extract_frame_scores \
    --run-dir "${RUN_DIR}" \
    --pkl data/continuous_per_trial.pkl \
    --device cpu

echo "=== Figures publication extremes ==="
python -m eval.figures_publication_extremes \
    --predictions-middle "${RUN_DIR}/predictions_middle.pkl" \
    --predictions-extremes "${RUN_DIR}/predictions_extremes.pkl" \
    --frames "${RUN_DIR}/frame_predictions.pkl" \
    --output "${RUN_DIR}/"

echo "=== Terminé ==="
ls -lh "${RUN_DIR}"/figure_extremes_*.pdf
