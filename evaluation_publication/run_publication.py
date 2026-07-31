#!/usr/bin/env python3
"""
Publication evaluation for Hybrid 1 under Extreme Validation.

Reuses the existing extremes protocol without changing architecture, loss,
split logic, or checkpoints:

  results/hybrid_extremes/
      model_best.pt
      norm_stats.pkl
      predictions_middle.pkl
      predictions_extremes.pkl
      frame_predictions.pkl   (extracted if missing)

Outputs (default results/publication_figures/):
      metrics.json
      fig{1,2,3}_*.{pdf,svg,png}

Usage (Narval)::

    export PYTHONPATH="$PWD:$PWD/src:${PYTHONPATH:-}"
    python -m evaluation_publication.run_publication \\
        --run-dir results/hybrid_extremes \\
        --out results/publication_figures

If the checkpoint is absent, pass ``--train`` to launch the unchanged
``train_hybrid_extremes`` entry point first.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluation_publication.config import (  # noqa: E402
    DEFAULT_CSV,
    DEFAULT_OUT,
    DEFAULT_PKL,
    DEFAULT_RUN_DIR,
)
from evaluation_publication.metrics import (  # noqa: E402
    compute_all_metrics,
    load_frame_entries,
    load_trial_predictions,
)
from evaluation_publication.plots import generate_all_figures  # noqa: E402


def _read_seq_len(run_dir: Path, default: int = 4000) -> int:
    split_path = run_dir / "split_info.json"
    if split_path.exists():
        with open(split_path, encoding="utf-8") as f:
            info = json.load(f)
        if "seq_len" in info:
            return int(info["seq_len"])
    return default


def ensure_training(
    run_dir: Path,
    pkl: Path,
    csv: Path,
    seed: int,
    smoke: bool,
) -> None:
    """Delegate to the canonical extremes trainer — no methodology edits."""
    cmd = [
        sys.executable, "-u", str(SRC / "train" / "train_hybrid_extremes.py"),
        "--output", str(run_dir),
        "--pkl", str(pkl),
        "--csv", str(csv),
        "--seed", str(seed),
    ]
    if smoke:
        cmd.append("--smoke")
    print("[publication] launching Extreme Validation training …")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def ensure_frame_predictions(
    run_dir: Path,
    pkl: Path,
    device: str,
    force: bool = False,
) -> Path:
    out_path = run_dir / "frame_predictions.pkl"
    if out_path.exists() and not force:
        print(f"[publication] reusing {out_path}")
        return out_path

    seq_len = _read_seq_len(run_dir)
    cmd = [
        sys.executable, "-u", "-m", "eval.extract_frame_scores",
        "--run-dir", str(run_dir),
        "--pkl", str(pkl),
        "--protocol", "extremes",
        "--seq-len", str(seq_len),
        "--device", device,
        "--out", str(out_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(SRC), env.get("PYTHONPATH", "")]
    )
    print("[publication] extracting per-frame scores …")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)
    return out_path


def run(
    run_dir: Path,
    out_dir: Path,
    pkl: Path,
    csv: Path,
    seed: int,
    device: str,
    train: bool,
    smoke: bool,
    force_frames: bool,
) -> dict:
    base_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Canonical trainer writes under base_dir/smoke when --smoke is set
    active_dir = base_dir / "smoke" if smoke else base_dir
    ckpt = active_dir / "model_best.pt"

    if train or not ckpt.exists():
        if not ckpt.exists():
            print(f"[publication] checkpoint missing ({ckpt}) — launching training")
        ensure_training(base_dir, pkl, csv, seed, smoke)

    active_dir = base_dir / "smoke" if smoke else base_dir
    if not (active_dir / "model_best.pt").exists():
        # Fall back to non-smoke artefacts if smoke flag was only for intent
        if (base_dir / "model_best.pt").exists():
            active_dir = base_dir
        else:
            raise FileNotFoundError(
                f"No model_best.pt under {active_dir}. "
                "Run Extreme Validation first or pass --train."
            )

    if not (active_dir / "predictions_middle.pkl").exists():
        raise FileNotFoundError(f"Missing predictions_middle.pkl in {active_dir}")

    frame_path = ensure_frame_predictions(active_dir, pkl, device, force=force_frames)

    print("[publication] loading predictions …")
    trial_preds = load_trial_predictions(active_dir)
    frame_entries = load_frame_entries(frame_path)
    print(
        f"[publication] trials={len(trial_preds)}  "
        f"frame_trials={len(frame_entries)}"
    )

    report = compute_all_metrics(trial_preds, frame_entries)
    report["run_dir"] = str(active_dir.resolve())
    report["frame_predictions"] = str(frame_path.resolve())

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[publication] metrics → {metrics_path}")

    print("[publication] generating figures …")
    paths = generate_all_figures(trial_preds, frame_entries, out_dir / "figures")
    for name, plist in paths.items():
        print(f"  {name}:")
        for p in plist:
            print(f"    {p}")

    mono = report["ordinal_monotonicity"]
    cm = report["confusion_matrix"]
    print(
        f"[publication] Spearman ρ={mono['spearman_rho']:+.3f} "
        f"(p={mono['spearman_p']:.2e})  R²={mono['r2']:.3f}  "
        f"CM acc={100 * cm['accuracy']:.1f}%"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Publication figures — Hybrid 1 Extreme Validation",
    )
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR,
                    help="Existing extremes run directory (checkpoint + preds)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="Output directory for metrics + figures")
    ap.add_argument("--pkl", type=Path, default=DEFAULT_PKL)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda",
                    help="Device for frame-score extraction")
    ap.add_argument("--train", action="store_true",
                    help="Run train_hybrid_extremes before evaluation")
    ap.add_argument("--smoke", action="store_true",
                    help="Forwarded to trainer (short seq / few epochs)")
    ap.add_argument("--force-frames", action="store_true",
                    help="Recompute frame_predictions.pkl even if present")
    return ap


def main(argv: Optional[list] = None) -> None:
    args = build_parser().parse_args(argv)
    run(
        run_dir=args.run_dir,
        out_dir=args.out,
        pkl=args.pkl,
        csv=args.csv,
        seed=args.seed,
        device=args.device,
        train=args.train,
        smoke=args.smoke,
        force_frames=args.force_frames,
    )


if __name__ == "__main__":
    main()
