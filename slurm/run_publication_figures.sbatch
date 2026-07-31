#!/bin/bash
# =============================================================================
# ICEMS — Publication figures (Hybrid 1 · Extreme Validation)
#
# Protocole inchangé : train_hybrid_extremes → predictions → figures.
#
# Prérequis sur Narval :
#   data/continuous_per_trial.pkl
#   data/Exvivo_trial_Participants(Sheet1).csv
#   .venv avec torch / matplotlib / seaborn / scipy / scikit-learn
#
# Run complet (entraîne si besoin, puis figures) :
#   cd ~/ICEMS-main && mkdir -p logs
#   sbatch slurm/run_publication_figures.sbatch
#
# Réutiliser un checkpoint déjà présent :
#   TRAIN=0 sbatch slurm/run_publication_figures.sbatch
#
# Smoke test :
#   SMOKE=1 sbatch slurm/run_publication_figures.sbatch
# =============================================================================
#SBATCH --account=def-hgueziri
#SBATCH --job-name=pub_figures
#SBATCH --output=logs/publication_figures-%j.out
#SBATCH --error=logs/publication_figures-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

set -euo pipefail
export PYTHONUNBUFFERED=1

TRAIN="${TRAIN:-1}"
SMOKE="${SMOKE:-0}"
FORCE_FRAMES="${FORCE_FRAMES:-0}"
SEED="${SEED:-42}"
RUN_DIR="${RUN_DIR:-results/hybrid_extremes}"
OUT="${OUT:-results/publication_figures}"
ROOT="${SLURM_SUBMIT_DIR:-$HOME/ICEMS-main}"

cd "$ROOT"
mkdir -p logs results

module purge 2>/dev/null || true
module load StdEnv/2023 2>/dev/null || true
module load python/3.11 2>/dev/null || true
module load cuda/12.2 2>/dev/null || true

if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"

echo "============================================================"
echo " ICEMS publication figures — Hybrid 1 Extreme Validation"
echo " Job ID  : ${SLURM_JOB_ID:-local}"
echo " Node    : ${SLURMD_NODENAME:-local}"
echo " TRAIN   : ${TRAIN}"
echo " SMOKE   : ${SMOKE}"
echo " RUN_DIR : ${RUN_DIR}"
echo " OUT     : ${OUT}"
echo " Date    : $(date)"
echo "============================================================"

python -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"

[[ -f data/continuous_per_trial.pkl ]] || { echo "ERREUR: data/continuous_per_trial.pkl manquant"; exit 1; }
[[ -f "data/Exvivo_trial_Participants(Sheet1).csv" ]] || { echo "ERREUR: CSV participants manquant"; exit 1; }

# Si TRAIN=1 et pas encore de checkpoint, entraîner via le script canonique
if [[ "$TRAIN" == "1" && ! -f "${RUN_DIR}/model_best.pt" ]]; then
    echo "=== Extreme Validation training (train_hybrid_extremes) ==="
    SMOKE_FLAG=()
    if [[ "$SMOKE" == "1" ]]; then
        SMOKE_FLAG=(--smoke)
    fi
    python -u src/train/train_hybrid_extremes.py \
        --output "${RUN_DIR}" \
        --seed "${SEED}" \
        "${SMOKE_FLAG[@]}"
    if [[ "$SMOKE" == "1" ]]; then
        RUN_DIR="${RUN_DIR}/smoke"
    fi
elif [[ "$TRAIN" == "1" && -f "${RUN_DIR}/model_best.pt" ]]; then
    echo "=== Checkpoint présent — entraînement ignoré ==="
fi

[[ -f "${RUN_DIR}/model_best.pt" ]] || { echo "ERREUR: ${RUN_DIR}/model_best.pt introuvable"; exit 1; }

DEVICE="cuda"
python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" || DEVICE="cpu"

EXTRA=()
if [[ "$FORCE_FRAMES" == "1" ]]; then
    EXTRA+=(--force-frames)
fi

echo "=== Publication evaluation + figures ==="
python -u -m evaluation_publication.run_publication \
    --run-dir "${RUN_DIR}" \
    --out "${OUT}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    "${EXTRA[@]}"

echo ""
echo "============================================================"
echo " Terminé — ${OUT}/"
echo "   metrics.json"
echo "   figures/fig1_temporal_stability.{pdf,svg,png}"
echo "   figures/fig2_ordinal_monotonicity.{pdf,svg,png}"
echo "   figures/fig3_confusion_matrix.{pdf,svg,png}"
echo "============================================================"
ls -lh "${OUT}/figures"/fig*.{pdf,png,svg} 2>/dev/null || true
