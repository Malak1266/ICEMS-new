#!/bin/bash
# =============================================================================
# ICEMS — Comparaison contrôlée GRU (Approche A/1) vs Hybrid LSTM+Transformer
#         (Approche E/2), à protocole IDENTIQUE, sur plusieurs seeds.
#
# Une seule variable change entre les deux bras : le MODÈLE.
#   - causal_gru  (--approach 1) : baseline séquentielle locale
#   - hybrid      (--approach 2) : BiLSTM (local) + Transformer (global)
# Loss = MSE, augmentation = global, epochs/patience/batch identiques.
#
# Exécution en SLURM ARRAY : chaque tâche = un couple (seed, modèle),
# exécutés en parallèle. 3 seeds × 2 modèles = 6 tâches → --array=0-5.
#
# SMOKE TEST (1 tâche, 2 folds, 3 epochs) avant le run complet :
#   sbatch --array=0 --export=ALL,SMOKE=1 scripts/narval/run_hybrid_vs_gru.sbatch
#
# RUN COMPLET (6 tâches) :
#   sbatch --array=0-5 scripts/narval/run_hybrid_vs_gru.sbatch
#
# Puis agrégation statistique (sur un nœud de connexion ou en local) :
#   python compare_architectures.py --root results/arch_compare
# =============================================================================
#SBATCH --account=def-hgueziri
#SBATCH --job-name=icems_gru_vs_hybrid
#SBATCH --output=logs/gru_vs_hybrid-%A_%a.out
#SBATCH --error=logs/gru_vs_hybrid-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5

set -euo pipefail
export PYTHONUNBUFFERED=1

# ── Grille de comparaison ────────────────────────────────────────────────────
SEEDS=(42 43 44)              # 3 seeds → variabilité d'initialisation
MODELS=(gru hybrid)           # 2 bras
# approach 1 = causal_gru, approach 2 = hybrid (cf. icems_strategy.APPROACHES)
declare -A APPROACH=( ["gru"]=1 ["hybrid"]=2 )

N_SEEDS=${#SEEDS[@]}
N_MODELS=${#MODELS[@]}

# Décodage de l'index array → (seed, modèle)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED_IDX=$(( TASK_ID / N_MODELS ))
MODEL_IDX=$(( TASK_ID % N_MODELS ))
SEED="${SEEDS[$SEED_IDX]}"
MODEL="${MODELS[$MODEL_IDX]}"
APP="${APPROACH[$MODEL]}"

# ── Hyperparamètres COMMUNS aux deux bras (ne pas diverger !) ────────────────
SMOKE="${SMOKE:-0}"
EPOCHS="${EPOCHS:-100}"        # aligné sur la baseline publiée
PATIENCE="${PATIENCE:-25}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MC_PASSES="${MC_PASSES:-30}"
AUG_MODE="${AUG_MODE:-global}" # identique pour gru et hybrid
LOSS="${LOSS:-mse}"            # identique pour gru et hybrid

ROOT="${SLURM_SUBMIT_DIR:-$HOME/ICEMS-main}"
cd "$ROOT"
mkdir -p logs results

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
source .venv/bin/activate

OUT_DIR="results/arch_compare/seed${SEED}/${MODEL}"

echo "============================================================"
echo " ICEMS — GRU vs Hybrid | tâche array ${TASK_ID}"
echo " Modèle    : ${MODEL} (approach ${APP})"
echo " Seed      : ${SEED}"
echo " Loss/Aug  : ${LOSS} / ${AUG_MODE}"
echo " Epochs    : ${EPOCHS}  Patience : ${PATIENCE}"
echo " OUT_DIR   : ${OUT_DIR}"
echo " SMOKE     : ${SMOKE}"
echo " Date      : $(date)"
echo "============================================================"

python -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"
[[ -f data/continuous_per_trial.pkl ]] || { echo "ERREUR: data/continuous_per_trial.pkl manquant"; exit 1; }

COMMON=(
    --data data/continuous_per_trial.pkl
    --out "$OUT_DIR"
    --approach "$APP"
    --model "$MODEL"
    --loss "$LOSS"
    --aug-mode "$AUG_MODE"
    --patience "$PATIENCE"
    --batch-size "$BATCH_SIZE"
    --seed "$SEED"
)

if [[ "$SMOKE" == "1" ]]; then
    python -u run_lopo_corrected.py --smoke-test "${COMMON[@]}" \
        --epochs 3 --mc-passes 3 --max-folds 2
else
    python -u run_lopo_corrected.py --full "${COMMON[@]}" \
        --epochs "$EPOCHS" --mc-passes "$MC_PASSES"
fi

echo ""
echo "============================================================"
echo " ✅ Terminé : ${MODEL} seed=${SEED} → ${OUT_DIR}/lopo_predictions.pkl"
echo "   Agrégation finale : python compare_architectures.py --root results/arch_compare"
echo "============================================================"
