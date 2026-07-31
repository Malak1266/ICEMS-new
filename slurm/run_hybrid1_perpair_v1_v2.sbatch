#!/bin/bash
# =============================================================================
# ICEMS — Hybrid1 per-pair v1 vs v2 (comparaison tracking_flag)
#
# Deux runs comparables (meme seed, protocole, architecture) :
#   --data v1  -> results/hybrid1_perpair_v1/
#   --data v2  -> results/hybrid1_perpair_v2/
#
# Narval :
#   cd ~/icems/ICEMS-main && mkdir -p logs results
#   sbatch slurm/run_hybrid1_perpair_v1_v2.sbatch
#
# Smoke (1 paire) :
#   PAIRS="0" BATCH_SIZE=2 sbatch slurm/run_hybrid1_perpair_v1_v2.sbatch
# =============================================================================
#SBATCH --account=def-hgueziri
#SBATCH --job-name=hybrid1_v1v2
#SBATCH --output=logs/hybrid1_v1v2-%j.out
#SBATCH --error=logs/hybrid1_v1v2-%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

set -euo pipefail
export PYTHONUNBUFFERED=1

SEED="${SEED:-42}"
MAX_LEN="${MAX_LEN:-5000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PAIRS="${PAIRS:-}"
DATA_ONLY="${DATA_ONLY:-}"
ROOT="${SLURM_SUBMIT_DIR:-$HOME/icems/ICEMS-main}"
VENV="${VENV:-$HOME/icems/.venv}"

cd "$ROOT"
mkdir -p logs results

module purge 2>/dev/null || true
module load StdEnv/2023 2>/dev/null || true
module load python/3.11 2>/dev/null || true
module load cuda/12.2 2>/dev/null || true

if [[ -f "${VENV}/bin/activate" ]]; then
    source "${VENV}/bin/activate"
elif [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

export PYTHONPATH="$PWD:$PWD/src:$PYTHONPATH"

python -c "
from models_hybrid1 import Hybrid1Config, Hybrid1ModelA, count_params
m = Hybrid1ModelA()
print('arch OK, params:', count_params(m))
" && ls data/trial_tensor_v2.pkl data/trial_tensor_v2.pkl && echo 'ALL CHECKS PASSED' || {
    echo "ERREUR: pre-check echoue (imports models_hybrid1 ou donnees v1/v2 manquantes)"
    exit 1
}

echo "============================================================"
echo " ICEMS Hybrid1 per-pair — comparaison v1 vs v2"
echo " Job ID   : ${SLURM_JOB_ID:-local}"
echo " Node     : ${SLURMD_NODENAME:-local}"
echo " SEED     : ${SEED}"
echo " MAX_LEN  : ${MAX_LEN}"
echo " BATCH    : ${BATCH_SIZE}"
echo " VENV     : ${VENV}"
echo " Date     : $(date)"
echo "============================================================"

python -c "
import torch
print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
"

PAIR_ARGS=()
if [[ -n "${PAIRS}" ]]; then
    PAIR_ARGS=(--pairs ${PAIRS})
fi

run_version() {
    local VER="$1"
    echo ""
    echo "---------- Hybrid1 per-pair data=${VER} ----------"
    python -u run_hybrid1_perpair.py \
        --data "${VER}" \
        --seed "${SEED}" \
        --max-len "${MAX_LEN}" \
        --batch-size "${BATCH_SIZE}" \
        "${PAIR_ARGS[@]}"
}

if [[ -n "${DATA_ONLY}" ]]; then
    run_version "${DATA_ONLY}"
else
    run_version v1
    run_version v2
    python -u compare_hybrid1_v1_v2.py
fi

echo ""
echo "============================================================"
echo " Termine"
echo "   results/hybrid1_perpair_v1/"
echo "   results/hybrid1_perpair_v2/"
echo "   results/hybrid1_perpair_v2/v1_vs_v2_scores.md"
echo "============================================================"
