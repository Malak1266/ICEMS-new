#!/usr/bin/env python3
"""
Point d'entrée — expérience extreme-train / intermediate-test.

Usage :
  python run_experiment.py --seed 42
  python run_experiment.py --smoke
  python run_parallel.py --smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from exp_config import DEFAULT_CSV, DEFAULT_PKL, PARALLEL_SEEDS, REPO_ROOT  # noqa: E402
from train import run_single_experiment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Hybrid LSTM-Transformer : train extrêmes → test intermédiaires",
    )
    ap.add_argument("--seed", type=int, default=None,
                    help="Graine (defaut: variable d'environnement SEED, sinon 0)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--no-hoel",
        action="store_true",
        help="MSE plate (protocole papier). Defaut: HOEL (asymetrique).",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--pkl", type=Path, default=DEFAULT_PKL)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = ap.parse_args()

    if not args.pkl.exists():
        print(f"[build] Génération de {args.pkl} depuis filtered_data.json …")
        import subprocess
        subprocess.check_call([
            sys.executable,
            str(REPO_ROOT / "src" / "build_continuous_dataset.py"),
            "--output", str(args.pkl),
        ])

    result = run_single_experiment(
        seed=args.seed,
        smoke=args.smoke,
        out=args.out,
        pkl=args.pkl,
        csv=args.csv,
        use_hoel=(False if args.no_hoel else None),
    )
    print(f"[done] seed={args.seed} → {result['out_dir']}")


if __name__ == "__main__":
    main()
