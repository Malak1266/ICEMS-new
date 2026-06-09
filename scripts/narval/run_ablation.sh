#!/bin/bash
# =============================================================================
# ICEMS — Ablation MAE Atracsys (Condition A vs B) sur Narval
#
# Usage :
#   cd ~/icems/ICEMS-main
#   sbatch scripts/narval/run_ablation.sh
#
# Override :
#   sbatch --export=ALL,ENCODER=~/icems/results/mae_run8/best_encoder.pt,MAX_FOLDS=3 scripts/narval/run_ablation.sh
# =============================================================================
#SBATCH --account=def-hgueziri
#SBATCH --job-name=ablation_mae
#SBATCH --output=results/ablation_atracsys_%j.out
#SBATCH --error=results/ablation_atracsys_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$HOME/icems/ICEMS-main}"
cd "$ROOT"

resolve_encoder() {
    # Cherche best_encoder.pt / encoder.pt dans ~/icems/results/mae_run*
    local candidates=()
    local d f
    for d in "$HOME/icems/results"/mae_run* "$HOME/icems/results"/*/; do
        [[ -d "$d" ]] || continue
        for f in best_encoder.pt encoder.pt best_encoder.pth; do
            [[ -f "$d/$f" ]] && candidates+=("$d/$f")
        done
    done
    if [[ ${#candidates[@]} -eq 0 ]]; then
        return 1
    fi
    # Prendre le plus récent (mtime)
    local best="" best_mtime=0 mtime
    for c in "${candidates[@]}"; do
        mtime=$(stat -c %Y "$c" 2>/dev/null || stat -f %m "$c")
        if [[ "$mtime" -gt "$best_mtime" ]]; then
            best_mtime=$mtime
            best=$c
        fi
    done
    echo "$best"
}

if [[ -z "${ENCODER:-}" ]]; then
    ENCODER=$(resolve_encoder || true)
    if [[ -z "$ENCODER" ]]; then
        ENCODER="$HOME/icems/results/mae_run8_2026-05-25/best_encoder.pt"
    else
        echo "ENCODER auto-détecté : $ENCODER"
    fi
fi
OUT_DIR="${OUT_DIR:-results/ablation_atracsys}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MC_PASSES="${MC_PASSES:-30}"
MAX_FOLDS="${MAX_FOLDS:-}"
SEED="${SEED:-42}"
AUG_TARGET="${AUG_TARGET:-}"
CONDITION="${CONDITION:-both}"

echo "=== Ablation MAE Atracsys ==="
echo "Debut : $(date)"
echo "ROOT  : $ROOT"
echo "OUT   : $OUT_DIR"
echo "ENCODER : $ENCODER"
echo "CONDITION : $CONDITION"

module purge
module load StdEnv/2023
module load python/3.11

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
else
    export PYTHONNOUSERSITE=0
fi

python -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"

if [[ ! -f data/continuous_per_trial.pkl ]]; then
    echo "ERREUR: data/continuous_per_trial.pkl manquant"
    exit 1
fi

if [[ "$CONDITION" == "both" || "$CONDITION" == "A" ]]; then
    if [[ ! -f "$ENCODER" ]]; then
        echo "ERREUR: checkpoint MAE introuvable : $ENCODER"
        echo ""
        echo "Recherche dans ~/icems :"
        find "$HOME/icems" -maxdepth 4 \( -name 'best_encoder.pt' -o -name 'encoder.pt' -o -name 'best_encoder.pth' \) 2>/dev/null || true
        echo ""
        echo "Options :"
        echo "  1) Relancer avec le bon chemin :"
        echo "     sbatch --export=ALL,ENCODER=/chemin/vers/best_encoder.pt scripts/narval/run_ablation.sh"
        echo "  2) Lancer condition B seule (random init) en attendant :"
        echo "     sbatch --export=ALL,CONDITION=B scripts/narval/run_ablation.sh"
        exit 1
    fi
fi

mkdir -p "$OUT_DIR"

EXTRA=()
[[ -n "$AUG_TARGET" ]] && EXTRA+=(--aug-target "$AUG_TARGET")
[[ -n "$MAX_FOLDS" ]] && EXTRA+=(--max-folds "$MAX_FOLDS")
[[ -n "${FREEZE_ENCODER:-}" ]] && EXTRA+=(--freeze-encoder)

python src/ablation_atracsys.py \
    --data data/continuous_per_trial.pkl \
    --out "$OUT_DIR" \
    --encoder "$ENCODER" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch-size "$BATCH_SIZE" \
    --mc-passes "$MC_PASSES" \
    --seed "$SEED" \
    --condition "$CONDITION" \
    "${EXTRA[@]}"

echo "Fin : $(date)"
