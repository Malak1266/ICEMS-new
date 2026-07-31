#!/bin/bash
# =============================================================================
# ICEMS — Génération des figures d'analyse d'expertise (post-run HOEL)
#
# Figures 1–3 : depuis predictions.pkl uniquement
# Figures 4–6 : nécessitent checkpoints/ (entraînement avec --save-checkpoints)
#
# Usage :
#   cd ~/ICEMS-main
#   sbatch slurm/run_expertise_figures.sbatch
#
# Variables :
#   PREDS=results/hybrid_hoel_b/predictions.pkl
#   OUT=results/hybrid_hoel_b/figures
#   CKPT=results/hybrid_hoel_b/checkpoints   (optionnel)
# =============================================================================
#SBATCH --account=def-hgueziri
#SBATCH --job-name=icems_figures
#SBATCH --output=logs/figures-%j.out
#SBATCH --error=logs/figures-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail
export PYTHONUNBUFFERED=1

PREDS="${PREDS:-results/hybrid_hoel_b/predictions.pkl}"
OUT="${OUT:-results/hybrid_hoel_b/figures}"
CKPT="${CKPT:-results/hybrid_hoel_b/checkpoints}"
ROOT="${SLURM_SUBMIT_DIR:-$HOME/ICEMS-main}"
cd "$ROOT"
mkdir -p logs

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 cuda/12.2 2>/dev/null || true
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

echo "=== Figures expertise — $(date) ==="
echo "PREDS=$PREDS  OUT=$OUT  CKPT=$CKPT"

[[ -f "$PREDS" ]] || { echo "ERREUR: $PREDS manquant — attendre fin du run HOEL"; exit 1; }

CKPT_ARGS=()
if [[ -d "$CKPT" ]] && compgen -G "$CKPT/fold_*/model.pt" > /dev/null; then
    CKPT_ARGS=(--checkpoints "$CKPT" --pkl data/continuous_per_trial.pkl --device cuda)
    echo "Checkpoints trouvés → figures 4–6 incluses"
else
    echo "Pas de checkpoints → figures 1–3 seulement"
fi

python -u -m eval.expertise_figure_suite \
    --preds "$PREDS" \
    --out "$OUT" \
    "${CKPT_ARGS[@]}"

echo "=== TERMINÉ — $OUT ==="
