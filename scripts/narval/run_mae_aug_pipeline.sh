#!/bin/bash
# =============================================================================
# ICEMS — Pipeline priorité 1 : MAE propre → augment → pretrain → ablation
#
# Usage :
#   cd ~/icems/ICEMS-main
#   sbatch scripts/narval/run_mae_aug_pipeline.sh
#
# Override :
#   sbatch --export=ALL,MAE_BASE=~/icems,SKIP_ABLATION=1 scripts/narval/run_mae_aug_pipeline.sh
# =============================================================================
#SBATCH --account=def-hgueziri
#SBATCH --job-name=mae_aug_pipe
#SBATCH --output=results/mae_aug_pipeline_%j.out
#SBATCH --error=results/mae_aug_pipeline_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$HOME/icems/ICEMS-main}"
cd "$ROOT"

MAE_BASE="${MAE_BASE:-$HOME/icems}"
DATA_DIR="${DATA_DIR:-$MAE_BASE/data}"
MAE_OUT="${MAE_OUT:-$HOME/icems/results/mae_aug_run1}"
ABLATION_OUT="${ABLATION_OUT:-results/ablation_aug}"
AUG_FACTOR="${AUG_FACTOR:-4}"
MAE_EPOCHS="${MAE_EPOCHS:-120}"
SESSIONS="${SESSIONS:-2026-05-25}"
SKIP_ABLATION="${SKIP_ABLATION:-}"
# Si pas de pipeline_output sur Narval : pointer vers un .npy existant
DATA_SOURCE="${DATA_SOURCE:-$HOME/icems/data/X_pretrain_v1_enrichi.npy}"
SKIP_BUILD="${SKIP_BUILD:-}"

echo "=== Pipeline MAE augmenté ==="
echo "Debut : $(date)"
echo "BASE  : $MAE_BASE"
echo "OUT   : $MAE_OUT"

module purge
module load StdEnv/2023
module load python/3.11

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
else
    export PYTHONNOUSERSITE=0
fi

python -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"

mkdir -p "$DATA_DIR" "$MAE_OUT"

echo ""
echo "--- Étape 1/4 : dataset MAE propre ---"
CLEAN_NPY="$DATA_DIR/X_pretrain_clean_4ch.npy"
if [[ -n "$SKIP_BUILD" && -f "$CLEAN_NPY" ]]; then
    echo "SKIP_BUILD : réutilise $CLEAN_NPY"
elif [[ -f "$DATA_SOURCE" ]]; then
    echo "Pas de pipeline_output → copie source existante : $DATA_SOURCE"
    python -c "
import numpy as np, shutil, sys
from pathlib import Path
src = Path('$DATA_SOURCE')
dst = Path('$CLEAN_NPY')
X = np.load(src)
if X.ndim != 3:
    sys.exit(f'Shape invalide: {X.shape}')
if X.shape[-1] == 6:
    X = X[:, :, [0, 1, 2, 5]]
elif X.shape[-1] != 4:
    sys.exit(f'Attendu 4 ou 6 canaux, reçu {X.shape}')
dst.parent.mkdir(parents=True, exist_ok=True)
np.save(dst, X.astype('float32'))
print(f'  → {dst}  shape={X.shape}')
"
elif ls -d "$MAE_BASE"/pipeline_output* 1>/dev/null 2>&1; then
    python src/build_mae_clean_dataset.py \
        --base "$MAE_BASE" \
        --out "$DATA_DIR" \
        --sessions $SESSIONS
else
    echo "ERREUR: ni DATA_SOURCE=$DATA_SOURCE ni pipeline_output sous $MAE_BASE"
    echo "  Définir DATA_SOURCE=~/icems/data/X_pretrain_v1_enrichi.npy"
    exit 1
fi

echo ""
echo "--- Étape 2/4 : augmentation (×${AUG_FACTOR}) ---"
python src/augment_mae_dataset.py \
    --in "$DATA_DIR/X_pretrain_clean_4ch.npy" \
    --out "$DATA_DIR/X_pretrain_aug_4ch.npy" \
    --factor "$AUG_FACTOR"

echo ""
echo "--- Étape 3/4 : pré-entraînement MAE ---"
python src/mae_pretrain.py \
    --data "$DATA_DIR/X_pretrain_aug_4ch.npy" \
    --output "$MAE_OUT" \
    --epochs "$MAE_EPOCHS"

ENCODER="$MAE_OUT/best_encoder.pt"
if [[ ! -f "$ENCODER" ]]; then
    echo "ERREUR: $ENCODER introuvable"
    exit 1
fi

if [[ -n "$SKIP_ABLATION" ]]; then
    echo "SKIP_ABLATION=1 — ablation non lancée."
    echo "Fin : $(date)"
    exit 0
fi

echo ""
echo "--- Étape 4/4 : ablation A vs B ---"
python src/ablation_atracsys.py \
    --data data/continuous_per_trial.pkl \
    --out "$ABLATION_OUT" \
    --encoder "$ENCODER" \
    --condition both

echo "Fin : $(date)"
echo "Encoder : $ENCODER"
echo "Ablation : $ABLATION_OUT/ablation_results.json"
