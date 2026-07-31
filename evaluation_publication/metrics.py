"""Metric computation from Hybrid extremes predictions (no re-training)."""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from evaluation_publication.config import (
    CLASS4_LABELS,
    CLASS4_ORDER,
    CLASS4_RANK,
    CLASS4_TARGETS,
    PHASE_BOUNDS,
    PHASE_LABELS,
    SUBLEVEL_TO_CLASS4,
)
from evaluation_publication.statistics import pearson_r2, spearman_with_p, summarise_groups


def resolve_class4(row: dict) -> str:
    c4 = row.get("class_4")
    if isinstance(c4, str) and c4.lower() in CLASS4_RANK:
        return c4.lower()
    if isinstance(c4, int) and 0 <= c4 < 4:
        return CLASS4_ORDER[c4]
    sub = str(row.get("sublevel", "")).strip().lower()
    return SUBLEVEL_TO_CLASS4.get(sub, "junior")


def score_to_class_idx(score: float) -> int:
    """Nearest continuous target among Student/Junior/Senior/Expert anchors."""
    targets = np.asarray(CLASS4_TARGETS, dtype=float)
    return int(np.argmin(np.abs(targets - float(score))))


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_trial_predictions(run_dir: Path) -> List[dict]:
    """Merge middle (test) + extremes (train) trial-level scores."""
    run_dir = Path(run_dir)
    middle_path = run_dir / "predictions_middle.pkl"
    extremes_path = run_dir / "predictions_extremes.pkl"
    if not middle_path.exists():
        raise FileNotFoundError(f"Missing {middle_path}")

    rows: List[dict] = []
    for path, split in ((middle_path, "test"), (extremes_path, "train")):
        if not path.exists():
            continue
        for row in load_pickle(path):
            item = dict(row)
            item["split"] = split
            item["class_4"] = resolve_class4(item)
            rows.append(item)
    return rows


def load_frame_entries(path: Path) -> List[dict]:
    payload = load_pickle(path)
    if isinstance(payload, dict) and "entries" in payload:
        entries = list(payload["entries"])
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError(f"Unrecognised frame_predictions format: {path}")
    for e in entries:
        e["class_4"] = resolve_class4(e)
    return entries


def phase_means(entries: Sequence[dict]) -> Dict[str, Dict[str, List[float]]]:
    """Mean predicted score per trial phase, grouped by expertise class."""
    out = {lab: {c: [] for c in CLASS4_ORDER} for lab in PHASE_LABELS}
    for e in entries:
        cls = e["class_4"]
        if cls not in CLASS4_RANK:
            continue
        t = np.asarray(e["time_norm"], dtype=float).ravel()
        s = np.asarray(e["frame_scores"], dtype=float).ravel()
        if t.size == 0 or s.size == 0:
            continue
        for (lo, hi), lab in zip(PHASE_BOUNDS, PHASE_LABELS):
            # Closed on the last bin so t=1.0 is retained
            if hi >= 1.0:
                mask = (t >= lo) & (t <= hi)
            else:
                mask = (t >= lo) & (t < hi)
            if mask.any():
                out[lab][cls].append(float(np.mean(s[mask])))
    return out


def ordinal_monotonicity(preds: Sequence[dict]) -> dict:
    """Class-level means + Spearman / R² on ordinal ranks (4 coarse levels)."""
    by_cls: Dict[str, List[float]] = defaultdict(list)
    for p in preds:
        by_cls[resolve_class4(p)].append(float(p["score_pred"]))

    group_stats = summarise_groups(by_cls, CLASS4_ORDER)
    ranks, means = [], []
    for cls in CLASS4_ORDER:
        if group_stats[cls]["n"] > 0:
            ranks.append(CLASS4_RANK[cls])
            means.append(group_stats[cls]["mean"])

    rho, p_rho = spearman_with_p(ranks, means)
    slope, intercept, r2 = pearson_r2(ranks, means)
    mono = all(means[i] <= means[i + 1] for i in range(len(means) - 1)) if len(means) >= 2 else True

    return {
        "by_class": {
            CLASS4_LABELS[c]: group_stats[c] for c in CLASS4_ORDER
        },
        "ranks": ranks,
        "means": means,
        "spearman_rho": rho,
        "spearman_p": p_rho,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "monotone": mono,
    }


def confusion_matrix_4(preds: Sequence[dict], normalise: str = "true") -> dict:
    """
    4×4 confusion from continuous scores via nearest-anchor mapping.

    normalise:
      'none' — raw counts
      'true' — row-normalised recall percentages (scientifically preferred)
    """
    cm = np.zeros((4, 4), dtype=np.float64)
    for p in preds:
        true_idx = CLASS4_ORDER.index(resolve_class4(p))
        pred_idx = score_to_class_idx(float(p["score_pred"]))
        cm[true_idx, pred_idx] += 1

    counts = cm.copy()
    if normalise == "true":
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            cm = np.where(row_sums > 0, 100.0 * cm / row_sums, 0.0)

    accuracy = float(np.trace(counts) / counts.sum()) if counts.sum() > 0 else float("nan")
    return {
        "matrix": cm.tolist(),
        "counts": counts.tolist(),
        "labels": [CLASS4_LABELS[c] for c in CLASS4_ORDER],
        "normalise": normalise,
        "accuracy": accuracy,
        "n": int(counts.sum()),
    }


def regression_summary(preds: Sequence[dict]) -> dict:
    y_true = np.asarray([p["score_true"] for p in preds], dtype=float)
    y_pred = np.asarray([p["score_pred"] for p in preds], dtype=float)
    if y_true.size == 0:
        return {}
    errors = y_pred - y_true
    rho, p_rho = spearman_with_p(y_true, y_pred)
    return {
        "n": int(y_true.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "spearman_rho": rho,
        "spearman_p": p_rho,
    }


def compute_all_metrics(
    trial_preds: Sequence[dict],
    frame_entries: Optional[Sequence[dict]] = None,
) -> dict:
    # Monotonicity / CM on the full labelled set (middle + extremes)
    mono = ordinal_monotonicity(trial_preds)
    cm = confusion_matrix_4(trial_preds, normalise="true")

    middle = [p for p in trial_preds if p.get("split") == "test"]
    extremes = [p for p in trial_preds if p.get("split") == "train"]

    report = {
        "protocol": "extreme_validation",
        "model": "HybridLSTMTransformer",
        "n_trials_total": len(trial_preds),
        "n_trials_middle": len(middle),
        "n_trials_extremes": len(extremes),
        "regression_middle": regression_summary(middle),
        "regression_extremes": regression_summary(extremes),
        "ordinal_monotonicity": mono,
        "confusion_matrix": cm,
    }

    if frame_entries:
        phases = phase_means(frame_entries)
        report["temporal_stability"] = {
            phase: summarise_groups(phases[phase], CLASS4_ORDER)
            for phase in PHASE_LABELS
        }
        report["n_frame_trials"] = len(frame_entries)

    return report
