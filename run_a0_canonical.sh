#!/bin/bash
#SBATCH --account=def-hgueziri
#SBATCH --job-name=a0_canon
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/a0_%A_%a.out
#SBATCH --error=logs/a0_%A_%a.err
#
# A0 canonique SPIE — Version A, d_in=10, pool=gap, agg=mean, 10 seeds.
# Prérequis : test_repro_seed42.sh PASS (|Δ|<1e-3) AVANT ce sbatch.
#
# Usage (depuis ~/icems/ICEMS-main) :
#   bash test_repro_seed42.sh          # gate déterminisme
#   sbatch run_a0_canonical.sh         # 10 seeds en parallèle

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$HOME/icems/ICEMS-main}"
cd "$ROOT"
mkdir -p logs results/a0_canonical

module purge
module load StdEnv/2023 python/3.12 cuda/12.2 2>/dev/null \
  || module load python/3.12
source /home/malek1/icems/.venv/bin/activate

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=0

SEEDS=(42 123 456 789 2024 1337 2718 3141 9001 5555)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
PKL="${PKL:-data/trial_tensor_v2.pkl}"
OUT="${OUT:-results/a0_canonical}"

echo "=========================================="
echo "A0 canonical  seed=$SEED  task=$SLURM_ARRAY_TASK_ID"
echo "host=$(hostname)  date=$(date -Is)"
echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "pkl=$PKL  out=$OUT"
echo "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
echo "=========================================="

# 1) Entraînement from scratch (GAP = A0)
python train_hybrid1.py \
    --pool-type gap \
    --seed "$SEED" \
    --pkl "$PKL" \
    --out "$OUT"

# 2) Scoring milieu + métriques SPIE (mean principal + median annexe)
python eval_hybrid1.py \
    --run "$OUT" \
    --seed "$SEED" \
    --pkl "$PKL" \
    --agg mean \
    --agg-both

echo "A0 seed $SEED done -> $OUT/seed${SEED}/"
echo "termine : $(date -Is)"
