#!/usr/bin/env python3
"""
Lance plusieurs seeds en parallèle (ProcessPoolExecutor).
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from exp_config import DEFAULT_CSV, DEFAULT_PKL, PARALLEL_SEEDS, REPO_ROOT  # noqa: E402


def _worker(seed: int, smoke: bool, pkl: str, csv: str, use_hoel: Optional[bool]) -> dict:
    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    from train import run_single_experiment
    return run_single_experiment(
        seed=seed,
        smoke=smoke,
        pkl=Path(pkl),
        csv=Path(csv),
        use_hoel=use_hoel,
    )


def _ensure_pkl(pkl: Path) -> None:
    if pkl.exists():
        return
    import subprocess
    subprocess.check_call([
        sys.executable,
        str(REPO_ROOT / "src" / "build_continuous_dataset.py"),
        "--output", str(pkl),
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel multi-seed experiment launcher")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(PARALLEL_SEEDS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--no-hoel",
        action="store_true",
        help="MSE plate (protocole papier). Defaut: HOEL (asymetrique).",
    )
    ap.add_argument("--workers", type=int, default=None,
                    help="Process pool size (default: min(n_seeds, cpu_count))")
    ap.add_argument("--pkl", type=Path, default=DEFAULT_PKL)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = ap.parse_args()

    _ensure_pkl(args.pkl)

    seeds = args.seeds
    workers = args.workers or min(len(seeds), max(1, __import__("os").cpu_count() or 1))
    use_hoel = False if args.no_hoel else None
    print(
        f"[parallel] seeds={seeds} workers={workers} smoke={args.smoke} "
        f"use_hoel={use_hoel if use_hoel is not None else 'default'}"
    )

    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_worker, s, args.smoke, str(args.pkl), str(args.csv), use_hoel): s
            for s in seeds
        }
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                res = fut.result()
                results.append(res)
                m = res["metrics"]["global"]
                print(
                    f"[ok] seed={seed} MAE={m['mae']:.4f} "
                    f"Spearman={m['spearman_rho']:.4f}"
                )
            except Exception as exc:
                print(f"[fail] seed={seed}: {exc}")

    agg_path = EXP_DIR / "results" / ("smoke_aggregate.json" if args.smoke else "aggregate.json")
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[done] {len(results)}/{len(seeds)} runs -> {agg_path}")


if __name__ == "__main__":
    main()
