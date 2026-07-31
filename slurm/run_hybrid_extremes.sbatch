#!/bin/bash
# =============================================================================
# ICEMS — Hybrid LSTM-Transformer train-extrêmes → test milieu + HOEL
#
# Protocole : MS + Staff en entraînement, PGY1…Fellow en test (pas de LOPO).
# Hyperparamètres : TrainConfig/HybridConfig/make_hoel() dans train_hybrid_extremes.py
#
# Run complet Narval :
#   cd ~/ICEMS-main && mkdir -p logs results
#   sbatch slurm/run_hybrid_extremes.sbatch
#
# Smoke test :
#   SMOKE=1 sbatch slurm/run_hybrid_extremes.sbatch
#
# Ablation sans augmentation :
#   AUGMENT=0 sbatch slurm/run_hybrid_extremes.sbatch
#
# Facteur d'augmentation (2 ou 3) :
#   AUGMENT_FACTOR=3 sbatch slurm/run_hybrid_extremes.sbatch
# =============================================================================
#SBATCH --account=def-hgueziri
#SBATCH --job-name=hybrid_extremes
#SBATCH --output=logs/hybrid_extremes-%j.out
#SBATCH --error=logs/hybrid_extremes-%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

set -euo pipefail
export PYTHONUNBUFFERED=1

SMOKE="${SMOKE:-0}"
SEED="${SEED:-42}"
AUGMENT="${AUGMENT:-1}"
AUGMENT_FACTOR="${AUGMENT_FACTOR:-2}"
OUT="${OUT:-results/hybrid_extremes}"
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
echo " ICEMS Hybrid — train extrêmes → test milieu (HOEL)"
echo " Job ID  : ${SLURM_JOB_ID:-local}"
echo " Node    : ${SLURMD_NODENAME:-local}"
echo " SMOKE   : ${SMOKE}"
echo " SEED    : ${SEED}"
echo " AUGMENT : ${AUGMENT} (factor=${AUGMENT_FACTOR})"
echo " OUT     : ${OUT}"
echo " Loss    : HOEL (lambda_asym=0.20, lambda_pair=0.30, lambda_top=0.40)"
echo " Date    : $(date)"
echo "============================================================"

python -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"

[[ -f data/continuous_per_trial.pkl ]] || { echo "ERREUR: data/continuous_per_trial.pkl manquant"; exit 1; }
[[ -f "data/Exvivo_trial_Participants(Sheet1).csv" ]] || { echo "ERREUR: CSV participants manquant"; exit 1; }

SMOKE_FLAG=()
if [[ "$SMOKE" == "1" ]]; then
    SMOKE_FLAG=(--smoke)
fi

python -u src/train/train_hybrid_extremes.py \
    --output "${OUT}" \
    --seed "${SEED}" \
    --augment "${AUGMENT}" \
    --augment-factor "${AUGMENT_FACTOR}" \
    "${SMOKE_FLAG[@]}"

python -u src/eval/eval_extremes.py \
    --results "${OUT}" \
    --out "${OUT}"

echo ""
echo "============================================================"
echo " Terminé — résultats : ${OUT}/"
echo "   predictions_middle.pkl"
echo "   table_extremes_metriques.csv"
echo "   figure_extremes_*.pdf"
echo "============================================================"
