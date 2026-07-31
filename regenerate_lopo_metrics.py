#!/usr/bin/env python3
"""Regénère metrics_summary.json depuis lopo_predictions.pkl (sans ré-entraîner)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from step_B_classification import (  # noqa: E402
    _sanitize_for_json,
    print_corrected_lopo_metrics,
    regenerate_metrics_from_predictions_pkl,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pkl",
        type=Path,
        default=Path("results/lopo_corrected/2026-06-18/lopo_predictions.pkl"),
    )
    ap.add_argument("--out", type=Path, default=None, help="Défaut : même dossier que --pkl")
    args = ap.parse_args()

    metrics, preds_df = regenerate_metrics_from_predictions_pkl(args.pkl)
    out_dir = args.out or args.pkl.parent
    out_path = out_dir / "metrics_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(metrics), f, indent=2, ensure_ascii=False)

    print(f"Regénéré : {out_path.resolve()}  ({len(preds_df)} trials)")
    print_corrected_lopo_metrics(metrics)


if __name__ == "__main__":
    main()
