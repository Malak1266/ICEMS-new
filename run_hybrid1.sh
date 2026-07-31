#!/bin/bash
#SBATCH --account=def-hgueziri
#SBATCH --job-name=icems_hyb1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/hyb1_seed%a_%A.out
#SBATCH --error=logs/hyb1_seed%a_%A.err
#SBATCH --array=0-4                # indices 0..4 → seeds PARALLEL_SEEDS

# ---------------------------------------------------------------- garde-fous
set -euo pipefail

ROOT=/home/malek1/icems/ICEMS-main
cd "$ROOT"
mkdir -p logs runs results

# Seeds gelées (ExpConfig.PARALLEL_SEEDS) — pas d'optimisation
SEEDS=(42 123 456 789 2024)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
export SEED

# 1. Refus de tourner sur un arbre non commite
if [[ -n "$(git status --porcelain | grep -v PROVENANCE.md || true)" ]]; then
  echo "ARRET : arbre git modifie. Commiter avant de lancer." >&2
  exit 1
fi
GIT_HASH=$(git rev-parse HEAD)

# 2. Refus de tourner si l'audit n'est pas au vert
if [[ ! -f docs/PROVENANCE.md ]]; then
  echo "ARRET : docs/PROVENANCE.md absent. Lancer audit_pipeline.py d'abord." >&2
  exit 1
fi
if grep -qE '`(ECHEC|INCONNU)`' docs/PROVENANCE.md; then
  echo "ARRET : l'audit contient ECHEC ou INCONNU." >&2
  exit 1
fi

# ---------------------------------------------------------------- environnement
module purge
module load StdEnv/2023 python/3.11 cuda/12.2
source /home/malek1/icems/.venv/bin/activate
export PYTHONPATH="$ROOT/src:$ROOT/experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1

PKL=data/continuous_per_trial.pkl

# ---------------------------------------------------------------- tracabilite
echo "=========================================="
echo "job          : ${SLURM_JOB_ID} / tache ${SLURM_ARRAY_TASK_ID}"
echo "noeud        : $(hostname)"
echo "date         : $(date -Is)"
echo "commit git   : ${GIT_HASH}"
echo "seed         : ${SEED}"
echo "gpu          : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "tenseur md5  : $(md5sum "$PKL" | cut -d' ' -f1)"
echo "=========================================="
python -c "import numpy,scipy,pandas,torch; print('numpy',numpy.__version__,'scipy',scipy.__version__,'torch',torch.__version__)"

# ---------------------------------------------------------------- entrainement
# Point d'entrée réel : run_experiment.py (train.py n'a pas de CLI)
srun python experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/run_experiment.py \
     --seed "${SEED}" \
     --pkl  "${PKL}"

# ---------------------------------------------------------------- verification
python - <<'PY'
import os, pandas as pd
f = f"runs/seed_{os.environ['SEED']}.csv"
d = pd.read_csv(f)
need = {"participant", "group", "split", "raw_score"}
assert need.issubset(d.columns), f"colonnes manquantes : {need - set(d.columns)}"
assert d.raw_score.between(-1, 1).all(), "raw_score hors (-1,1) : calibration a fuite dans train"
tr = set(d.loc[d.split == "train", "participant"])
te = set(d.loc[d.split == "test", "participant"])
va = set(d.loc[d.split == "val", "participant"])
assert not (tr & te), f"FUITE : participants partages train∩test {tr & te}"
assert not (tr & va), f"FUITE : participants partages train∩val {tr & va}"
mid = d[(d.split == "train") & d.group.str.lower().isin(["junior", "senior"])]
assert mid.empty, f"FUITE GRAVE : middle en train\n{mid}"
print(f"{f} : {len(d)} lignes — toutes les assertions passent")
PY

echo "termine : $(date -Is)"
