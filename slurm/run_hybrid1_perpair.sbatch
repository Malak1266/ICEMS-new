#!/bin/bash
# =============================================================================
# ICEMS — Hybrid1 per-pair (eApp 2) : 10 modeles x (train + score composite)
#
# Pour comparaison v1 vs v2, preferer :
#   sbatch slurm/run_hybrid1_perpair_v1_v2.sbatch
#
# Run v1 seul :
#   DATA_ONLY=v1 sbatch slurm/run_hybrid1_perpair_v1_v2.sbatch
# =============================================================================
#SBATCH --account=def-hgueziri
#SBATCH --job-name=hybrid1_perpair
#SBATCH --output=logs/hybrid1_perpair-%j.out
#SBATCH --error=logs/hybrid1_perpair-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

set -euo pipefail
export PYTHONUNBUFFERED=1

SEED="${SEED:-42}"
MAX_LEN="${MAX_LEN:-5000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DATA="${DATA:-v1}"
PAIRS="${PAIRS:-}"
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

export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

PAIR_ARGS=()
if [[ -n "${PAIRS}" ]]; then
    PAIR_ARGS=(--pairs ${PAIRS})
fi

python -u run_hybrid1_perpair.py \
    --data "${DATA}" \
    --seed "${SEED}" \
    --max-len "${MAX_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    "${PAIR_ARGS[@]}"
