"""
Évaluation sur intermédiaires uniquement : MAE, MSE, Spearman, erreurs par sous-groupe.
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


def compute_metrics(preds: Sequence[dict]) -> dict:
    if not preds:
        return {}

    y_true = np.array([p["score_true"] for p in preds], dtype=np.float64)
    y_pred = np.array([p["score_pred"] for p in preds], dtype=np.float64)
    errors = y_pred - y_true

    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))

    if len(y_true) > 2:
        rho, p_rho = stats.spearmanr(y_true, y_pred)
        r_pearson, p_pearson = stats.pearsonr(y_true, y_pred)
    else:
        rho, p_rho = float("nan"), float("nan")
        r_pearson, p_pearson = float("nan"), float("nan")

    return {
        "n_trials": len(preds),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "spearman_rho": float(rho),
        "p_spearman": float(p_rho),
        "pearson_r": float(r_pearson),
        "p_pearson": float(p_pearson),
        "error_mean": float(np.mean(errors)),
        "error_std": float(np.std(errors)),
    }


def metrics_by_subgroup(preds: Sequence[dict]) -> dict:
    by_group: Dict[str, List[dict]] = defaultdict(list)
    by_sublevel: Dict[str, List[dict]] = defaultdict(list)
    for p in preds:
        by_group[p.get("subgroup", "unknown")].append(p)
        by_sublevel[p["sublevel"]].append(p)

    out = {
        "by_subgroup": {k: compute_metrics(v) for k, v in sorted(by_group.items())},
        "by_sublevel": {k: compute_metrics(v) for k, v in sorted(by_sublevel.items())},
    }
    return out


def plot_error_distribution(preds: Sequence[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    errors = [p["score_pred"] - p["score_true"] for p in preds]
    subgroups = [p.get("subgroup", "?") for p in preds]
    sublevels = [p["sublevel"] for p in preds]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.boxplot(x=subgroups, y=errors, ax=axes[0], palette="Set2")
    axes[0].axhline(0, color="k", ls="--", lw=0.8)
    axes[0].set_title("Error distribution by subgroup (junior / senior)")
    axes[0].set_ylabel("Prediction error")
    axes[0].set_xlabel("Subgroup")

    order = sorted(set(sublevels), key=lambda s: (
        ["pgy1", "pgy2", "pgy3", "pgy4", "pgy5", "pgy6", "fellow"].index(s)
        if s in {"pgy1", "pgy2", "pgy3", "pgy4", "pgy5", "pgy6", "fellow"} else 99
    ))
    sns.boxplot(x=sublevels, y=errors, order=order, ax=axes[1], palette="viridis")
    axes[1].axhline(0, color="k", ls="--", lw=0.8)
    axes[1].set_title("Error distribution by PGY sublevel")
    axes[1].set_ylabel("Prediction error")
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(out_dir / "error_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    y_true = [p["score_true"] for p in preds]
    y_pred = [p["score_pred"] for p in preds]
    colors = {"junior": "#1f77b4", "senior": "#ff7f0e"}
    c = [colors.get(p.get("subgroup", ""), "#888888") for p in preds]
    ax2.scatter(y_true, y_pred, c=c, alpha=0.75, edgecolors="k", linewidths=0.3)
    ax2.plot([-1, 1], [-1, 1], "k--", lw=1)
    ax2.set_xlim(-1.05, 1.05)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_xlabel("True score (intermediate)")
    ax2.set_ylabel("Predicted score")
    ax2.set_title("Extreme-train → Intermediate-test")
    fig2.tight_layout()
    fig2.savefig(out_dir / "scatter_intermediate.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)


def run_evaluation(preds: Sequence[dict], out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_metrics = compute_metrics(preds)
    subgroup_metrics = metrics_by_subgroup(preds)
    full = {"global": global_metrics, **subgroup_metrics}

    with open(out_dir / "predictions.pkl", "wb") as f:
        pickle.dump(list(preds), f)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2)

    plot_error_distribution(preds, out_dir)
    print(f"[eval] MAE={global_metrics.get('mae', '?'):.4f} "
          f"MSE={global_metrics.get('mse', '?'):.4f} "
          f"Spearman={global_metrics.get('spearman_rho', '?'):.4f}")
    return full
