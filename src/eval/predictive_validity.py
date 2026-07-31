"""
predictive_validity.py
=======================
Validité prédictive du protocole extrêmes : régression OLS (pred_score ~ année de
formation) sur les niveaux intermédiaires jamais vus, avec intervalles de confiance
bootstrap PAR PARTICIPANT (unité statistique = participant, pas trial).

Sortie :
  - metrics.json : R², pente, intercept, MSE, MAE + IC95 (R², pente)
  - scatter.png  : pred vs année, points colorés par sublevel, droite OLS, bande IC95
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import linregress

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42

# Couleurs par sous-niveau intermédiaire (cohérentes avec le reste du projet).
SUBLEVEL_COLORS = {
    "pgy1": "#fee08b", "pgy2": "#fdae61", "pgy3": "#f46d43",
    "pgy4": "#d73027", "pgy5": "#4575b4", "pgy6": "#74add1",
    "fellow": "#1a9850",
}


def _ols_metrics(years: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """OLS pred ~ year + métriques d'erreur."""
    reg = linregress(years, preds)
    fitted = reg.intercept + reg.slope * years
    residuals = preds - fitted
    return {
        "r2": float(reg.rvalue ** 2),
        "slope": float(reg.slope),
        "intercept": float(reg.intercept),
        "p_value": float(reg.pvalue),
        "mse": float(np.mean(residuals ** 2)),
        "mae": float(np.mean(np.abs(residuals))),
    }


def _bootstrap_ci(
    years: np.ndarray, preds: np.ndarray, n_boot: int, seed: int
) -> Dict[str, List[float]]:
    """IC95 bootstrap (percentile) sur R² et pente — resampling PAR PARTICIPANT.

    Chaque point (year, pred) est déjà un participant (médiane par participant en
    amont). Le bootstrap rééchantillonne donc les participants avec remise.
    """
    rng = np.random.default_rng(seed)
    n = len(years)
    r2_samples, slope_samples = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # tirage avec remise des participants
        yb, pb = years[idx], preds[idx]
        if np.unique(yb).size < 2:  # régression impossible si une seule année
            continue
        reg = linregress(yb, pb)
        r2_samples.append(float(reg.rvalue ** 2))
        slope_samples.append(float(reg.slope))

    def ci95(vals: List[float]) -> List[float]:
        if not vals:
            return [float("nan"), float("nan")]
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    return {
        "r2_ci95": ci95(r2_samples),
        "slope_ci95": ci95(slope_samples),
        "n_valid_resamples": len(r2_samples),
    }


def _plot_scatter(
    results: List[dict],
    metrics: Dict,
    years: np.ndarray,
    preds: np.ndarray,
    out_path: Path,
    seed: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    # Points colorés par sublevel
    seen = set()
    for r in results:
        sl = r["sublevel"]
        ax.scatter(
            r["year_of_training"], r["pred_score"],
            color=SUBLEVEL_COLORS.get(sl, "#666666"),
            s=70, alpha=0.85, edgecolors="black", linewidths=0.5,
            label=sl if sl not in seen else None,
        )
        seen.add(sl)

    # Droite OLS
    xs = np.linspace(years.min() - 0.3, years.max() + 0.3, 100)
    ys = metrics["intercept"] + metrics["slope"] * xs
    ax.plot(xs, ys, color="black", lw=2,
            label=f"OLS (R²={metrics['r2']:.3f}, pente={metrics['slope']:.3f})")

    # Bande IC95 bootstrap (recalcul des droites sur resamples)
    rng = np.random.default_rng(seed)
    n = len(years)
    boot_lines = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        if np.unique(years[idx]).size < 2:
            continue
        reg = linregress(years[idx], preds[idx])
        boot_lines.append(reg.intercept + reg.slope * xs)
    if boot_lines:
        boot_arr = np.stack(boot_lines)
        lo = np.percentile(boot_arr, 2.5, axis=0)
        hi = np.percentile(boot_arr, 97.5, axis=0)
        ax.fill_between(xs, lo, hi, color="gray", alpha=0.25, label="IC95 bootstrap")

    ax.set_xlabel("Année de formation (PGY1=1 … PGY6=6, Fellow=7)")
    ax.set_ylabel("Score d'expertise prédit")
    ax.set_title("Validité prédictive — niveaux intermédiaires jamais vus")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_predictive_validity(results_json_path, output_dir) -> Dict:
    """Charge le JSON du protocole extrêmes, calcule OLS + IC95 bootstrap, sauvegarde.

    Parameters
    ----------
    results_json_path : chemin du JSON produit par run_extremes_protocol().
    output_dir : dossier de sortie (metrics.json + scatter.png).

    Returns
    -------
    dict : métriques OLS + IC95.
    """
    results_json_path = Path(results_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(results_json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    years = np.array([r["year_of_training"] for r in results], dtype=float)
    preds = np.array([r["pred_score"] for r in results], dtype=float)

    if len(results) < 2 or np.unique(years).size < 2:
        raise ValueError(
            f"Validité prédictive impossible : n={len(results)} participants, "
            f"{np.unique(years).size} année(s) distincte(s)."
        )

    metrics = _ols_metrics(years, preds)
    metrics.update(_bootstrap_ci(years, preds, N_BOOTSTRAP, BOOTSTRAP_SEED))
    metrics["n_participants"] = int(len(results))

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    scatter_path = output_dir / "scatter.png"
    _plot_scatter(results, metrics, years, preds, scatter_path, BOOTSTRAP_SEED)

    metrics["metrics_path"] = str(metrics_path)
    metrics["scatter_path"] = str(scatter_path)
    return metrics
