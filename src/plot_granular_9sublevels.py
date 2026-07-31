"""
plot_granular_9sublevels.py
============================
Visualisations granulaires 9 sous-niveaux (TABLE I) pour ICEMS V2.

Usage :
    python src/plot_granular_9sublevels.py --preds results/lopo_v2/2026-06-19/lopo_predictions.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

SUBLEVEL_MAP: Dict[str, dict] = {
    "Medical Student": {"group": "Student", "y4": 0, "y_reg": -1.00, "color": "#d32f2f"},
    "Resident PGY1": {"group": "Junior", "y4": 1, "y_reg": -0.33, "color": "#f57c00"},
    "Resident PGY2": {"group": "Junior", "y4": 1, "y_reg": -0.33, "color": "#ffa726"},
    "Resident PGY3": {"group": "Junior", "y4": 1, "y_reg": -0.33, "color": "#ffcc02"},
    "Resident PGY4": {"group": "Junior", "y4": 1, "y_reg": -0.33, "color": "#ffee58"},
    "Resident PGY5": {"group": "Junior", "y4": 1, "y_reg": -0.33, "color": "#d4e157"},
    "Resident PGY6": {"group": "Senior", "y4": 2, "y_reg": +0.33, "color": "#66bb6a"},
    "Fellow": {"group": "Senior", "y4": 2, "y_reg": +0.33, "color": "#26a69a"},
    "Neurosurgeon": {"group": "Expert", "y4": 3, "y_reg": +1.00, "color": "#1565c0"},
}

SUBLEVEL_N = {
    "Medical Student": 14,
    "Resident PGY1": 5,
    "Resident PGY2": 3,
    "Resident PGY3": 2,
    "Resident PGY4": 1,
    "Resident PGY5": 3,
    "Resident PGY6": 4,
    "Fellow": 7,
    "Neurosurgeon": 8,
}

MIN_N_FOR_STATS = 3


def _paper_y_reg(sublevel: str, fallback: float) -> float:
    return float(SUBLEVEL_MAP.get(sublevel, {}).get("y_reg", fallback))


def plot_scatter_9sublevels(preds: Sequence[dict], output_path: Path) -> None:
    """Scatter prédit vs réel, coloré par sous-niveau (version publiable)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    for sublevel, props in SUBLEVEL_MAP.items():
        trials = [p for p in preds if p.get("sublevel") == sublevel]
        n = len(trials)
        if n == 0:
            continue

        y_true = [_paper_y_reg(sublevel, float(p["y_reg"])) for p in trials]
        y_pred = [p["score"] for p in trials]

        marker = "*" if n < MIN_N_FOR_STATS else "o"
        label = f"{sublevel} (n={n})" if n < MIN_N_FOR_STATS else sublevel

        ax.scatter(
            y_true, y_pred,
            c=props["color"], marker=marker,
            s=120 if n < MIN_N_FOR_STATS else 60,
            alpha=0.8, label=label, zorder=3,
        )

    ax.plot([-1.1, 1.1], [-1.1, 1.1], "k--", alpha=0.3, lw=1)

    for score, name in [(-1.0, "Student"), (-0.33, "Junior"), (0.33, "Senior"), (1.0, "Expert")]:
        ax.axvline(score, color="gray", ls=":", lw=0.7, alpha=0.5)
        ax.axhline(score, color="gray", ls=":", lw=0.7, alpha=0.5)

    ax.set_xlabel("Score réel (ordinal)", fontsize=12)
    ax.set_ylabel("Score prédit", fontsize=12)
    ax.set_title("Score prédit vs réel — 9 sous-niveaux\nGRU Causal + LOPO", fontsize=13)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)

    low_n = [s for s in SUBLEVEL_MAP if SUBLEVEL_N[s] < MIN_N_FOR_STATS]
    if low_n:
        ax.text(
            0.02, 0.02, f"★ n<{MIN_N_FOR_STATS}: {', '.join(low_n)}",
            transform=ax.transAxes, fontsize=7, color="gray", va="bottom",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


def plot_violin_9sublevels(preds: Sequence[dict], output_path: Path) -> None:
    """Distribution des scores prédits par sous-niveau (diagnostique)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(SUBLEVEL_MAP.keys())

    data_plot = []
    for p in preds:
        data_plot.append({
            "sublevel": p.get("sublevel", ""),
            "score_predit": p["score"],
            "n": SUBLEVEL_N.get(p.get("sublevel", ""), 0),
        })
    df = pd.DataFrame(data_plot)

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, sublevel in enumerate(order):
        sub_df = df[df["sublevel"] == sublevel]
        n_part = SUBLEVEL_N.get(sublevel, 0)

        if n_part >= MIN_N_FOR_STATS and len(sub_df) > 0:
            parts = ax.violinplot(
                sub_df["score_predit"], positions=[i],
                showmedians=True, showextrema=True,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(SUBLEVEL_MAP[sublevel]["color"])
                pc.set_alpha(0.7)
        elif len(sub_df) > 0:
            ax.scatter(
                [i] * len(sub_df), sub_df["score_predit"],
                c=SUBLEVEL_MAP[sublevel]["color"],
                s=80, alpha=0.9, marker="D", zorder=5,
            )
            ax.text(i, -1.05, f"n={n_part}\n⚠", ha="center", fontsize=7, color="red")

    for score, name, color in [
        (-1.00, "Student", "#d32f2f"),
        (-0.33, "Junior", "#f57c00"),
        (+0.33, "Senior", "#66bb6a"),
        (+1.00, "Expert", "#1565c0"),
    ]:
        ax.axhline(score, color=color, ls="--", lw=1.0, alpha=0.5,
                  label=f"{name} ({score:+.2f})")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [f"{s}\n(n={SUBLEVEL_N.get(s, 0)})" for s in order],
        rotation=30, ha="right", fontsize=9,
    )
    ax.set_ylabel("Score prédit [-1, +1]", fontsize=11)
    ax.set_title(
        "Distribution des scores prédits par sous-niveau\n(⚠ = n<3, stats non fiables)",
        fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(-1.2, 1.2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


def plot_score_vs_time_9sublevels(curves_data: Sequence[dict], output_path: Path) -> None:
    """Grille 3×3 : courbe moyenne ± écart-type par sous-niveau."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(SUBLEVEL_MAP.keys())
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, sublevel in enumerate(order):
        ax = axes_flat[i]
        n_part = SUBLEVEL_N.get(sublevel, 0)
        props = SUBLEVEL_MAP[sublevel]

        sub_curves = [c for c in curves_data if c.get("sublevel") == sublevel]

        if n_part < MIN_N_FOR_STATS:
            ax.set_facecolor("#f5f5f5")
            for c in sub_curves:
                ax.plot(c["time"], c["scores"], color=props["color"], alpha=0.6, lw=1.5)
            ax.text(
                0.5, 0.92, f"⚠ n={n_part} (non statistique)",
                transform=ax.transAxes, ha="center", fontsize=7, color="red",
            )
        elif sub_curves:
            all_scores = np.array([c["scores_interp"] for c in sub_curves])
            time_grid = sub_curves[0]["time_grid"]
            mean = all_scores.mean(axis=0)
            std = all_scores.std(axis=0)

            ax.plot(time_grid, mean, color=props["color"], lw=2.5, label="Moyenne")
            ax.fill_between(time_grid, mean - std, mean + std,
                           color=props["color"], alpha=0.2, label="±1 std")

        target = props["y_reg"]
        ax.axhline(target, color=props["color"], ls="--", lw=1, alpha=0.6)

        ax.set_title(f"{sublevel}\n(n={n_part})", fontsize=9, fontweight="bold",
                    color=props["color"])
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel("Temps normalisé", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Score [-1, +1]", fontsize=9)

    fig.suptitle(
        "Progression du score d'expertise au cours du geste\n"
        "GRU Causal + LOPO — 9 sous-niveaux",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


def plot_ordinal_monotonicity(
    preds: Sequence[dict], output_path: Path,
) -> Tuple[float, float]:
    """Monotonicité ordinale — figure publiable pour l'encadrant."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    order = list(SUBLEVEL_MAP.keys())

    ranks, means, stds, ns, colors = [], [], [], [], []
    ranks_for_rho, means_for_rho = [], []

    for rank, sublevel in enumerate(order, 1):
        sub_preds = [p["score"] for p in preds if p.get("sublevel") == sublevel]
        n = SUBLEVEL_N.get(sublevel, 0)
        if len(sub_preds) == 0:
            continue

        m = float(np.mean(sub_preds))
        s = float(np.std(sub_preds))

        ranks.append(rank)
        means.append(m)
        stds.append(s)
        ns.append(n)
        colors.append(SUBLEVEL_MAP[sublevel]["color"])

        if n >= MIN_N_FOR_STATS:
            ranks_for_rho.append(rank)
            means_for_rho.append(m)

    if len(ranks_for_rho) >= 2:
        rho, p_val = spearmanr(ranks_for_rho, means_for_rho)
        rho, p_val = float(rho), float(p_val)
    else:
        rho, p_val = float("nan"), float("nan")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(ranks, means, yerr=stds, fmt="none",
               color="gray", alpha=0.4, capsize=4, zorder=2)

    ax.scatter(
        ranks, means,
        c=colors,
        s=[max(60, n * 20) for n in ns],
        zorder=3, edgecolors="white", linewidths=1.5,
    )

    for rank, mean, sublevel, n in zip(ranks, means, order[: len(ranks)], ns):
        suffix = " ⚠" if n < MIN_N_FOR_STATS else ""
        ax.annotate(
            f"{sublevel}{suffix}\n(n={n})",
            (rank, mean),
            textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=7,
            color="red" if n < MIN_N_FOR_STATS else "black",
        )

    if len(ranks_for_rho) >= 2:
        z = np.polyfit(ranks_for_rho, means_for_rho, 1)
        p_fit = np.poly1d(z)
        x_line = np.linspace(1, len(order), 100)
        ax.plot(x_line, p_fit(x_line), "k--", alpha=0.4, lw=1.5,
               label="Tendance linéaire")

    for score, _name in [(-1.0, "Student"), (-0.33, "Junior"),
                         (0.33, "Senior"), (1.0, "Expert")]:
        ax.axhline(score, color="gray", ls=":", lw=0.7, alpha=0.4)

    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([f"{i}" for i in range(1, len(order) + 1)])
    ax.set_xlabel("Rang d'expertise ordinal (1=débutant → 9=expert)", fontsize=11)
    ax.set_ylabel("Score prédit moyen", fontsize=11)
    ax.set_title(
        f"Monotonicité ordinale — Spearman ρ = {rho:.3f} (p = {p_val:.2e})\n"
        f"(⚠ = sous-niveaux n<{MIN_N_FOR_STATS} exclus du calcul ρ)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(-1.2, 1.2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")

    return rho, p_val


def run_granular_plots(
    preds: Sequence[dict],
    curves_data: Optional[Sequence[dict]],
    out_dir: Path,
) -> None:
    """Génère les 4 figures granulaires dans out_dir."""
    plot_scatter_9sublevels(preds, out_dir / "scatter_9sublevels.png")
    plot_violin_9sublevels(preds, out_dir / "violin_9sublevels.png")
    if curves_data:
        plot_score_vs_time_9sublevels(curves_data, out_dir / "score_vs_time_9sublevels.png")
    plot_ordinal_monotonicity(preds, out_dir / "ordinal_monotonicity.png")


def load_preds_from_pkl(path: Path) -> Tuple[List[dict], Optional[List[dict]]]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    preds = list(payload.get("preds", []))
    curves = payload.get("curves")
    return preds, curves


def main() -> None:
    ap = argparse.ArgumentParser(description="Figures granulaires 9 sous-niveaux ICEMS V2.")
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    preds, curves = load_preds_from_pkl(args.preds)
    out_dir = args.out or args.preds.parent

    try:
        from step_B_classification import enrich_preds_with_sublevel, prepare_curves_for_granular_plots
    except ImportError:
        from src.step_B_classification import enrich_preds_with_sublevel, prepare_curves_for_granular_plots

    preds = enrich_preds_with_sublevel(
        preds,
        participant_csv=Path("data/Exvivo_trial_Participants(Sheet1).csv"),
    )
    curves_data = prepare_curves_for_granular_plots(curves) if curves else None
    run_granular_plots(preds, curves_data, out_dir)
    print(f"\n✅ Figures granulaires dans {out_dir.resolve()}")


if __name__ == "__main__":
    main()
