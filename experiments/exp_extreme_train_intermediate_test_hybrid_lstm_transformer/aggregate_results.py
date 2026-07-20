#!/usr/bin/env python3
"""Agrège les metrics.json de tous les seeds après le array SLURM."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    results_dir = args.results_dir
    seed_dirs = sorted(results_dir.glob("seed_*"))
    if not seed_dirs:
        raise SystemExit(f"Aucun dossier seed_* dans {results_dir}")

    rows = []
    for d in seed_dirs:
        mpath = d / "metrics.json"
        spath = d / "experiment_summary.json"
        if not mpath.exists():
            print(f"[skip] {d.name} — metrics.json manquant")
            continue
        with open(mpath, encoding="utf-8") as f:
            metrics = json.load(f)
        seed = int(d.name.replace("seed_", ""))
        row = {"seed": seed, "dir": str(d), "metrics": metrics}
        if spath.exists():
            with open(spath, encoding="utf-8") as f:
                row["summary"] = json.load(f)
        rows.append(row)

    if not rows:
        raise SystemExit("Aucun résultat à agréger.")

    def _mean_std(key: str) -> dict:
        vals = [
            r["metrics"]["global"][key]
            for r in rows
            if key in r["metrics"].get("global", {})
        ]
        vals = [v for v in vals if v == v]  # drop nan
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }

    aggregate = {
        "n_seeds": len(rows),
        "seeds": [r["seed"] for r in rows],
        "mae": _mean_std("mae"),
        "mse": _mean_std("mse"),
        "spearman_rho": _mean_std("spearman_rho"),
        "per_seed": rows,
    }

    out = args.out or (results_dir / "aggregate.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print(f"[aggregate] {len(rows)} seeds -> {out}")
    print(
        f"  MAE      : {aggregate['mae']['mean']:.4f} "
        f"+/- {aggregate['mae']['std']:.4f}"
    )
    print(
        f"  Spearman : {aggregate['spearman_rho']['mean']:.4f} "
        f"+/- {aggregate['spearman_rho']['std']:.4f}"
    )


if __name__ == "__main__":
    main()
