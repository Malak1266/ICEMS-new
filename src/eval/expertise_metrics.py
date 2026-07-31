"""
expertise_metrics.py
====================
Métriques et figures pour l'évaluation LOPO hybrid + loss hiérarchique.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import kendalltau


TIER_NAMES = ("PGY", "Fellow", "Expert")


def _tier_from_score(score: float, thresholds: tuple[float, float] = (0.45, 0.75)) -> int:
    if score >= thresholds[1]:
        return 2
    if score >= thresholds[0]:
        return 1
    return 0


def compute_all(preds: Sequence[dict]) -> dict:
    """Agrège métriques trial-level et participant-level."""
    if not preds:
        return {}

    y_true = np.array([p["score_true"] for p in preds], dtype=np.float64)
    y_pred = np.array([p["score_pred"] for p in preds], dtype=np.float64)
    tiers_true = np.array([p["tier"] for p in preds], dtype=np.int64)
    tiers_pred = np.array([
        _tier_from_score(float(s)) for s in y_pred
    ], dtype=np.int64)

    r_trial, p_trial = stats.pearsonr(y_true, y_pred) if len(y_true) > 2 else (float("nan"), float("nan"))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rho, p_rho = stats.spearmanr(y_true, y_pred) if len(y_true) > 2 else (float("nan"), float("nan"))
    tau, tau_p = kendalltau(y_true, y_pred) if len(y_true) > 2 else (float("nan"), float("nan"))

    by_pid: Dict[str, List[float]] = {}
    by_pid_true: Dict[str, List[float]] = {}
    for p in preds:
        pid = str(p["participant"])
        by_pid.setdefault(pid, []).append(float(p["score_pred"]))
        by_pid_true.setdefault(pid, []).append(float(p["score_true"]))

    p_true = np.array([np.mean(by_pid_true[k]) for k in sorted(by_pid)])
    p_pred = np.array([np.mean(by_pid[k]) for k in sorted(by_pid)])
    r_part, _ = stats.pearsonr(p_true, p_pred) if len(p_true) > 2 else (float("nan"), float("nan"))

    tier_acc = float(np.mean(tiers_true == tiers_pred))

    return {
        "n_trials": len(preds),
        "n_participants": len(by_pid),
        "r_trial": float(r_trial),
        "p_trial": float(p_trial),
        "r_participant": float(r_part),
        "mae": mae,
        "rmse": rmse,
        "spearman_rho": float(rho),
        "p_spearman": float(p_rho),
        "kendall_tau": float(tau),
        "kendall_p": float(tau_p),
        "tier_accuracy": tier_acc,
    }


def plot_confusion(
    preds: Sequence[dict],
    out_path: Path,
    title: str = "Tier confusion (pred vs true)",
) -> None:
    tiers_true = [p["tier"] for p in preds]
    tiers_pred = [_tier_from_score(float(p["score_pred"])) for p in preds]
    cm = np.zeros((3, 3), dtype=int)
    for t, pr in zip(tiers_true, tiers_pred):
        cm[int(t), int(pr)] += 1

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=TIER_NAMES, yticklabels=TIER_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted tier")
    ax.set_ylabel("True tier")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(
    preds: Sequence[dict],
    out_path: Path,
    title: str = "LOPO predictions",
) -> None:
    y_true = [p["score_true"] for p in preds]
    y_pred = [p["score_pred"] for p in preds]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    c = [colors[int(p["tier"])] for p in preds]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, c=c, alpha=0.7, edgecolors="k", linewidths=0.3)
    ax.plot([-1, 1], [-1, 1], "k--", lw=1)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("True score")
    ax.set_ylabel("Predicted score")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_history(
    fold_histories: Sequence[List[dict]],
    out_path: Path,
    title: str = "Training history (mean over folds)",
) -> None:
    if not fold_histories:
        return
    max_ep = max(len(h) for h in fold_histories)
    train_curve = np.full(max_ep, np.nan)
    val_curve = np.full(max_ep, np.nan)
    counts = np.zeros(max_ep)

    for hist in fold_histories:
        for i, row in enumerate(hist):
            if np.isnan(train_curve[i]):
                train_curve[i] = row.get("train_loss", np.nan)
                val_curve[i] = row.get("val_loss", np.nan)
            else:
                train_curve[i] += row.get("train_loss", 0)
                val_curve[i] += row.get("val_loss", 0)
            counts[i] += 1

    valid = counts > 0
    train_curve[valid] /= counts[valid]
    val_curve[valid] /= counts[valid]
    epochs = np.arange(1, max_ep + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs[valid], train_curve[valid], label="train")
    ax.plot(epochs[valid], val_curve[valid], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics(metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
