#!/bin/bash
# Gate déterminisme A0 — deux runs identiques seed 42 doivent matcher au 3e décimal.
# Lancer AVANT run_a0_canonical.sh. Ne pas lancer les 10 seeds si FAIL.
#
# Usage (depuis ~/icems/ICEMS-main, idéalement sur un nœud GPU) :
#   bash test_repro_seed42.sh
#   # ou interactif :
#   salloc --account=def-hgueziri --gres=gpu:a100:1 --cpus-per-task=4 --mem=32G --time=02:00:00
#   bash test_repro_seed42.sh

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
mkdir -p results/repro_check logs

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.12 cuda/12.2 2>/dev/null \
  || module load python/3.12 2>/dev/null || true
if [[ -f /home/malek1/icems/.venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source /home/malek1/icems/.venv/bin/activate
fi

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=42

PKL="${PKL:-data/trial_tensor_v2.pkl}"
SEED=42

echo "=== repro check seed $SEED ==="
echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "pkl=$PKL"
echo "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
echo "device probe:"
python - <<'PY'
import torch
print("  torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  gpu", torch.cuda.get_device_name(0))
PY

run_one () {
  local tag="$1"
  local out="results/repro_check/${tag}"
  rm -rf "$out"
  echo "--- train+eval -> $out ---"
  # train : pas de --agg (flag eval uniquement)
  python train_hybrid1.py \
      --pool-type gap \
      --seed "$SEED" \
      --pkl "$PKL" \
      --out "$out"
  python eval_hybrid1.py \
      --run "$out" \
      --seed "$SEED" \
      --pkl "$PKL" \
      --agg mean
}

run_one run_A
run_one run_B

python - <<'PY'
import json, sys
from pathlib import Path

SEED = 42
a_path = Path(f"results/repro_check/run_A/seed{SEED}/metrics_middle.json")
b_path = Path(f"results/repro_check/run_B/seed{SEED}/metrics_middle.json")

def load(p):
    if not p.exists():
        print(f"[!] fichier manquant: {p}")
        sys.exit(1)
    return json.loads(p.read_text(encoding="utf-8"))

def get_metric(d, key):
    # metrics_middle.json : nested primary/secondary + ols_bootstrap
    if key in d and isinstance(d[key], (int, float)):
        return float(d[key])
    pm = d.get("primary_metrics") or {}
    sm = d.get("secondary_metrics") or {}
    ols = d.get("ols_bootstrap") or {}
    if key == "slope":
        return float(pm.get("slope", ols.get("slope")))
    if key == "r2":
        return float(sm.get("r2", ols.get("r2")))
    if key == "rho_middle":
        return float(pm.get("rho_middle", (d.get("spearman") or {}).get("middle", {}).get("rho")))
    raise KeyError(key)

a = load(a_path)
b = load(b_path)
keys = ["slope", "r2", "rho_middle"]
ok = True
print("--- seed-42-twice ---")
for k in keys:
    try:
        va, vb = get_metric(a, k), get_metric(b, k)
    except Exception as e:
        print(f"[!] clé manquante: {k} ({e})")
        ok = False
        continue
    if va is None or vb is None:
        print(f"[!] clé manquante: {k}"); ok = False; continue
    d = abs(va - vb)
    flag = "OK" if d < 1e-3 else "FAIL"
    if d >= 1e-3:
        ok = False
    print(f"{k:12s} runA={va:+.4f} runB={vb:+.4f} |Δ|={d:.2e} {flag}")

for tag in ("run_A", "run_B"):
    cfg = Path(f"results/repro_check/{tag}/seed{SEED}/train_config.json")
    if cfg.exists():
        tc = json.loads(cfg.read_text(encoding="utf-8"))
        print(f"[{tag}] determinism={tc.get('determinism')} git_sha={tc.get('git_sha')}")

sys.exit(0 if ok else 1)
PY

echo "=== fin repro check ==="
