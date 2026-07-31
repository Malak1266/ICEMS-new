#!/bin/bash
# =============================================================================
# ICEMS — Hybrid LSTM-Transformer LOPO + HOEL
#
# Loss : HOEL (Hierarchical Ordinal Expertise Loss) — voir src/losses/hoel.py
# Config : SUBLEVEL_TO_SCORE / SUBLEVEL_TO_RANK / SUBLEVEL_TO_TIER dans src/config.py
# Hyperparamètres figés dans TrainConfig/HybridConfig/make_hoel() de train_hybrid_lopo.py
#
# Ce sont des CIBLES d'entraînement modifiées AVANT l'entraînement — pas les
# sorties du modèle. Aucune fuite de données.
#
# Run complet Narval :
#   cd ~/ICEMS-main && mkdir -p logs results
#   sbatch slurm/run_hybrid_fixed.sbatch
#
# Smoke test (2 folds, 2 epochs) :
#   SMOKE=1 sbatch slurm/run_hybrid_fixed.sbatch
# =============================================================================
#SBATCH --account=def-hgueziri
#SBATCH --job-name=icems_hybrid_fixed
#SBATCH --output=logs/hybrid_fixed-%j.out
#SBATCH --error=logs/hybrid_fixed-%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

set -euo pipefail
export PYTHONUNBUFFERED=1

SMOKE="${SMOKE:-0}"
SEED="${SEED:-42}"
OUT="${OUT:-results/hybrid_hoel}"
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
echo " ICEMS Hybrid LOPO — HOEL"
echo " Job ID  : ${SLURM_JOB_ID:-local}"
echo " Node    : ${SLURMD_NODENAME:-local}"
echo " SMOKE   : ${SMOKE}"
echo " SEED    : ${SEED}   (identique au job 64364891)"
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

python -u src/train/train_hybrid_lopo.py \
    --out "${OUT}" \
    --seed "${SEED}" \
    "${SMOKE_FLAG[@]}"

echo ""
echo "============================================================"
echo " Terminé — résultats : ${OUT}/predictions.pkl"
echo "   metrics.json : r_trial, kendall_tau, rmse, tier_accuracy"
echo "   (comparer avec baseline : results/hybrid_hierarchical/predictions.pkl)"
echo "============================================================"
