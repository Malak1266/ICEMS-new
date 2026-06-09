#!/bin/bash
# =============================================================================
# ICEMS — configuration environnement sur Narval (Alliance Canada)
# Usage (depuis la racine ICEMS-main sur Narval) :
#   bash scripts/narval/setup_env.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "=== ICEMS setup Narval ==="
echo "Répertoire : $ROOT"

# Modules Alliance (Narval)
module purge
module load StdEnv/2023
module load python/3.11
# Optionnel si vous voulez forcer le GPU pour PyTorch :
# module load cuda/12.2

if [[ ! -d .venv ]]; then
    python -m venv .venv
    echo "→ venv créé"
fi

source .venv/bin/activate
pip install --upgrade pip wheel

# PyTorch CPU ou CUDA — sur Narval GPU, décommentez la ligne CUDA :
pip install torch --index-url https://download.pytorch.org/whl/cu121
# pip install torch --index-url https://download.pytorch.org/whl/cpu

# Dépendances légères pour Step B (sans TensorFlow si non nécessaire)
pip install numpy scipy scikit-learn matplotlib seaborn tslearn tqdm

echo ""
echo "=== Vérification ==="
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
python -c "import tslearn; print('tslearn OK')"

if [[ ! -f data/continuous_per_trial.pkl ]]; then
    echo ""
    echo "⚠️  data/continuous_per_trial.pkl manquant."
    echo "   Transférez-le depuis votre machine locale, ou générez-le :"
    echo "   python src/build_continuous_dataset.py"
fi

echo ""
echo "✅ Environnement prêt. Activez avec : source .venv/bin/activate"
