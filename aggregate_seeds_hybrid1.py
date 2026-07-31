"""
aggregate_seeds_hybrid1.py
==========================
Agrege les rapports panneau E de plusieurs seeds -> moyenne +/- ecart-type.

Lit results/hybrid1_faithful/seed{S}/panelE_report.json pour chaque seed fournie,
et produit :
  * un tableau console (moyenne +/- std par groupe, E-vs-S p, Spearman rho, R2 middle)
  * results/hybrid1_faithful/aggregate_report.json
  * results/hybrid1_faithful/panelE_barplot_aggregate.png (barres = moyenne inter-seeds)

Usage :
    python aggregate_seeds_hybrid1.py --seeds 42 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_hybrid1 import GROUP4_DISPLAY, GROUP4_ORDER

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("aggregate")

PAPER_TARGET = {"expert": 0.75, "senior": 0.25, "junior": -0.09, "novice": -0.80}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("results/hybrid1_faithful"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    args = ap.parse_args()

    reports = []
    used_seeds = []
    for s in args.seeds:
        p = args.run / f"seed{s}" / "panelE_report.json"
        if p.exists():
            reports.append(json.loads(p.read_text()))
            used_seeds.append(s)
        else:
            logger.warning(f"[skip] rapport manquant pour seed {s} : {p}")

    if not reports:
        raise SystemExit("Aucun rapport trouve.")

    logger.info("=" * 70)
    logger.info(f"AGREGATION panneau E sur {len(reports)} seeds : {used_seeds}")
    logger.info("=" * 70)

    # Moyennes de groupe
    group_means = {g: np.array([r["group_means_ci95"][g]["mean"] for r in reports])
                   for g in GROUP4_ORDER}
    logger.info("\n[groupes] score participant (moyenne inter-seeds +/- std) vs cible papier :")
    agg_groups = {}
    for g in GROUP4_ORDER:
        vals = group_means[g]
        m, sd = float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        agg_groups[g] = {"mean": m, "std": sd, "per_seed": vals.tolist()}
        logger.info(f"   {GROUP4_DISPLAY[g]:9s} : {m:+.3f} +/- {sd:.3f}   "
                    f"(cible ~{PAPER_TARGET[g]:+.2f})")

    def agg(path_fn, label, fmt="{:+.4f}"):
        vals = np.array([path_fn(r) for r in reports], dtype=float)
        m, sd = float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        logger.info(f"   {label:28s} : {fmt.format(m)} +/- {fmt.format(sd).lstrip('+')}")
        return {"mean": m, "std": sd, "per_seed": vals.tolist()}

    logger.info("\n[stats globales] (moyenne inter-seeds +/- std) :")
    evs_diff = agg(lambda r: r["expert_vs_senior"]["diff"], "Expert-vs-Senior diff")
    evs_p = agg(lambda r: r["expert_vs_senior"]["p_adj"], "Expert-vs-Senior p_adj", "{:.4g}")
    n_sig = sum(1 for r in reports if r["expert_vs_senior"]["significant"])
    logger.info(f"   Expert-vs-Senior significatif : {n_sig}/{len(reports)} seeds")
    anova_p = agg(lambda r: r["anova"]["p"], "ANOVA p", "{:.4g}")
    rho = agg(lambda r: r["spearman_9sublevels"]["rho"], "Spearman rho (9 niveaux)")
    r2 = agg(lambda r: r["predictive_validity_middle"]["r2"], "OLS middle R2")
    slope = agg(lambda r: r["predictive_validity_middle"]["slope"], "OLS middle pente")

    out = {
        "seeds": used_seeds,
        "n_seeds": len(reports),
        "paper_target": PAPER_TARGET,
        "group_means": agg_groups,
        "expert_vs_senior_diff": evs_diff,
        "expert_vs_senior_p_adj": evs_p,
        "expert_vs_senior_n_significant": n_sig,
        "anova_p": anova_p,
        "spearman_rho": rho,
        "ols_middle_r2": r2,
        "ols_middle_slope": slope,
    }
    with open(args.run / "aggregate_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Barplot agrege (barres = moyenne inter-seeds, err = std inter-seeds)
    labels = [GROUP4_DISPLAY[g] for g in GROUP4_ORDER]
    means = [agg_groups[g]["mean"] for g in GROUP4_ORDER]
    stds = [agg_groups[g]["std"] for g in GROUP4_ORDER]
    targets = [PAPER_TARGET[g] for g in GROUP4_ORDER]
    colors = ["#1a6b3c", "#4a9e6f", "#e0a44c", "#c0392b"]

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    x = np.arange(len(GROUP4_ORDER))
    ax.bar(x, means, yerr=stds, capsize=6, color=colors, edgecolor="black",
           width=0.62, label="Reproduction (moyenne +/- std inter-seeds)")
    ax.scatter(x, targets, marker="D", s=70, color="black", zorder=5,
               label="Cible papier (panneau E)")
    ax.axhline(0.0, color="gray", lw=0.8, ls="--", zorder=0)
    for xi, m, sd in zip(x, means, stds):
        ax.text(xi, m + (0.04 if m >= 0 else -0.04), f"{m:+.2f}",
                ha="center", va="bottom" if m >= 0 else "top", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score d'expertise predit (mediane participant)", fontsize=11)
    ax.set_title(f"Panneau E — reproduction agregee ({len(reports)} seeds)", fontsize=12)
    ax.set_ylim(-1.15, 1.15)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.run / "panelE_barplot_aggregate.png", dpi=150)
    plt.close(fig)
    logger.info(f"\n[save] {args.run / 'aggregate_report.json'}")
    logger.info(f"[save] {args.run / 'panelE_barplot_aggregate.png'}")


if __name__ == "__main__":
    main()
