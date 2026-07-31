#!/bin/bash
# =============================================================================
# ICEMS — Expérience parallel : Train extrêmes → Test intermédiaires
# Hybrid LSTM-Transformer (isolated under experiments/)
#
# SOUMISSION NARVAL (5 seeds GPU en parallèle) :
#   cd ~/ICEMS-main
#   bash experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/slurm/submit_narval.sh
#
# Smoke test (1 seed, ~10 min GPU) :
#   SMOKE=1 bash .../slurm/submit_narval.sh
#
# Vérifier les jobs :
#   squeue -u $USER
#   sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS
# =============================================================================
#SBATCH --account=def-hgueziri
#SBATCH --job-name=icems_ext_int
#SBATCH --output=experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/logs/parallel-%A_%a.out
#SBATCH --error=experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/logs/parallel-%A_%a.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%5
# Note: time/mem reduits pour priorite file (run ~3-15 min/seed). GPU generique = plus de slots.

set -euo pipefail
export PYTHONUNBUFFERED=1

SMOKE="${SMOKE:-0}"
# USE_HOEL=0 → MSE plate (Run A / protocole papier). Defaut 1 = HOEL.
USE_HOEL="${USE_HOEL:-1}"
ROOT="${SLURM_SUBMIT_DIR:-$HOME/ICEMS-main}"
EXP="experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer"
SEEDS=(42 123 456 789 2024)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${SEEDS[$TASK_ID]}"

cd "$ROOT"
mkdir -p "${EXP}/logs" "${EXP}/results"

echo "============================================================"
echo " ICEMS extreme->intermediate (Hybrid LSTM-Transformer)"
echo " Job ID  : ${SLURM_JOB_ID:-local}  Array task: ${TASK_ID}"
echo " Seed    : ${SEED}"
echo " Node    : ${SLURMD_NODENAME:-local}"
echo " ROOT    : ${ROOT}"
echo " SMOKE   : ${SMOKE}"
echo " USE_HOEL: ${USE_HOEL}"
echo " Date    : $(date)"
echo "============================================================"

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2

if [[ ! -f .venv/bin/activate ]]; then
    echo "ERREUR: .venv introuvable dans ${ROOT}"
    echo "  Creez l'environnement :"
    echo "    module load python/3.11"
    echo "    python -m venv .venv && source .venv/bin/activate"
    echo "    pip install -r requirements.txt  # ou setup du projet"
    exit 1
fi
source .venv/bin/activate

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

python -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available(), 'device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

[[ -f "data/Exvivo_trial_Participants(Sheet1).csv" ]] || {
    echo "ERREUR: data/Exvivo_trial_Participants(Sheet1).csv manquant"; exit 1; }
[[ -f data/filtered_data.json ]] || {
    echo "ERREUR: data/filtered_data.json manquant"; exit 1; }

if [[ ! -f data/continuous_per_trial.pkl ]]; then
    echo "[build] Generation de continuous_per_trial.pkl ..."
    python src/build_continuous_dataset.py --output data/continuous_per_trial.pkl
fi

SMOKE_FLAG=()
if [[ "$SMOKE" == "1" ]]; then
    SMOKE_FLAG=(--smoke)
fi

HOEL_FLAG=()
if [[ "$USE_HOEL" == "0" ]]; then
    HOEL_FLAG=(--no-hoel)
fi

python -u "${EXP}/run_experiment.py" --seed "${SEED}" "${SMOKE_FLAG[@]}" "${HOEL_FLAG[@]}"

echo "============================================================"
echo " Termine seed=${SEED}"
echo " Resultats : ${EXP}/results/seed_${SEED}/"
echo "============================================================"
