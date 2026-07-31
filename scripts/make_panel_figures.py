#!/usr/bin/env python3
r"""
ICEMS / Hybrid1 per-pair — Panel de figures publication (Nature/IEEE style).

Usage (sur Narval) :
    python scripts/make_panel_figures.py \
        --pred results/hybrid1_perpair_v1/predictions_composite.pkl \
        --out  results/figures/ \
        --tag  v1

Produit :
    panel_ABCD.pdf / .png   (figure 4 panneaux)
    panelA_scores.pdf, panelB_calibration.pdf, panelC_monotonicity.pdf,
    panelD_forest.pdf       (versions individuelles, pour insertion LaTeX)
    panel_stats.json        (toutes les valeurs numeriques -> \RES{} du memoire)

Design : palette monochrome sequentielle (encode l'ordinalite + lisible en
niveaux de gris), lignes fines, sans grille lourde, ratio data-to-ink maximal.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

# Affiche les lignes ±1 "cibles d'entrainement" (honnetete scientifique :
# Novice/Expert tombent sur ces lignes PAR CONSTRUCTION de la calibration).
SHOW_TRAIN_TARGETS = True

GROUPS = ["novice", "junior", "senior", "expert"]
GROUP_LABELS = {"novice": "Novice", "junior": "Junior",
                "senior": "Senior", "expert": "Expert"}

PALETTE = {
    "novice": "#C2CCD2",
    "junior": "#8AA2AE",
    "senior": "#4E7180",
    "expert": "#1E3A47",
}
INK = "#2B2B2B"
MUTED = "#9AA5AB"
REF = "#B3603F"

PAPER = {
    "expert": {"mean": 0.75, "ci95_half": 0.22, "n": 8},
    "senior": {"mean": 0.25, "ci95_half": None, "n": 11},
    "junior": {"mean": -0.09, "ci95_half": None, "n": 14},
    "novice": {"mean": -0.80, "ci95_half": None, "n": 14},
}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "axes.titleweight": "regular",
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.linewidth": 1.0,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load(pred_path: Path) -> dict:
    recs = pickle.load(open(pred_path, "rb"))
    s = np.array([r["score"] for r in recs], dtype=float)
    g = np.array([str(r["group4"]).lower() for r in recs])
    lv = np.array([str(r["level9"]).lower() for r in recs])
    yr = np.array([float(r["year"]) for r in recs])
    return {"raw": s, "group": g, "level9": lv, "year": yr, "n": len(s)}


def calibrate(s: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, float, float]:
    me, mn = s[g == "expert"].mean(), s[g == "novice"].mean()
    a = 2.0 / (me - mn)
    b = -a * (me + mn) / 2.0
    return a * s + b, a, b


def hedges_g(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    n1, n2 = len(x), len(y)
    sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    d = (x.mean() - y.mean()) / sp
    J = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    g_ = J * d
    se = np.sqrt((n1 + n2) / (n1 * n2) + g_ ** 2 / (2.0 * (n1 + n2)))
    return g_, g_ - 1.96 * se, g_ + 1.96 * se


def level_order(lv: np.ndarray, yr: np.ndarray) -> list[str]:
    uniq = sorted(set(lv))
    return sorted(uniq, key=lambda L: yr[lv == L].mean())


def panel_A(ax, d: dict, cal: np.ndarray) -> None:
    s, g = cal, d["group"]
    data = [s[g == k] for k in GROUPS]
    pos = np.arange(1, 5)

    if SHOW_TRAIN_TARGETS:
        for _t in (-1.0, 1.0):
            ax.axhline(_t, color=MUTED, lw=0.6, ls=(0, (4, 3)), zorder=0)
        ax.text(4.62, 1.0, "cibles\nd'entraînement", color=MUTED, fontsize=6,
                va="center", ha="left", linespacing=1.3)

    bp = ax.boxplot(data, positions=pos, widths=0.56, showfliers=False,
                    patch_artist=True, medianprops=dict(color=INK, lw=1.1),
                    whiskerprops=dict(color=INK, lw=0.6),
                    capprops=dict(color=INK, lw=0.6),
                    boxprops=dict(lw=0.6))
    for patch, k in zip(bp["boxes"], GROUPS):
        patch.set_facecolor(PALETTE[k])
        patch.set_edgecolor(INK)
        patch.set_alpha(0.85)

    rng = np.random.default_rng(0)
    for i, k in enumerate(GROUPS):
        v = s[g == k]
        x = pos[i] + rng.uniform(-0.13, 0.13, len(v))
        ax.scatter(x, v, s=7, facecolor="white", edgecolor=INK,
                   linewidth=0.45, zorder=3, alpha=0.9)

    ax.set_xticks(pos)
    ax.set_xticklabels([GROUP_LABELS[k] for k in GROUPS])
    ax.set_ylabel("Score d'expertise calibré")
    # Affichage [-1, 1] : l'echelle saturante du modele (tanh) + cibles train.
    # Quelques points calibres peuvent depasser ±1 → clippes visuellement.
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(0.45, 4.55)
    ax.axhline(0, color=MUTED, lw=0.4, zorder=0)
    ax.set_title("A", loc="left", fontweight="bold", fontsize=10, pad=6)


def panel_B(ax, d: dict, cal: np.ndarray, a: float) -> None:
    order = level_order(d["level9"], d["year"])
    rank = np.array([order.index(L) for L in d["level9"]], dtype=float)
    x = 2.0 * rank / (len(order) - 1) - 1.0

    ax.plot([-1, 1], [-1, 1], color=MUTED, lw=0.6, ls=(0, (4, 3)), zorder=0)
    ax.text(0.98, 0.90, "identité", color=MUTED, fontsize=6, ha="right")

    ax.scatter(x, d["raw"], s=9, facecolor="white", edgecolor=MUTED,
               linewidth=0.5, zorder=2)
    sl0, ic0, *_ = stats.linregress(x, d["raw"])
    xs = np.linspace(-1, 1, 50)
    ax.plot(xs, sl0 * xs + ic0, color=MUTED, lw=1.2, zorder=3)

    ax.scatter(x, cal, s=9, color=PALETTE["senior"], alpha=0.75,
               edgecolor="none", zorder=4)
    sl1, ic1, *_ = stats.linregress(x, cal)
    ax.plot(xs, sl1 * xs + ic1, color=PALETTE["expert"], lw=1.4, zorder=5)

    ax.annotate(f"brut  (pente {sl0:.3f})", xy=(-0.05, sl0 * -0.05 + ic0),
                xytext=(-0.95, -0.88), color=MUTED, fontsize=6.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5,
                                shrinkA=0, shrinkB=2))
    ax.annotate(f"calibré  (pente {sl1:.2f})", xy=(0.42, sl1 * 0.42 + ic1),
                xytext=(-0.15, 0.88), color=PALETTE["expert"], fontsize=6.5,
                arrowprops=dict(arrowstyle="-", color=PALETTE["expert"],
                                lw=0.5, shrinkA=0, shrinkB=2))
    ax.text(-0.97, -0.97, f"facteur de compression corrigé : ×{a:.1f}",
            fontsize=6.5, color=INK)

    ax.set_xlabel("Niveau d'expertise (rang ordinal normalisé)")
    ax.set_ylabel("Score du modèle")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.0, 1.0)
    ax.set_title("B", loc="left", fontweight="bold", fontsize=10, pad=6)


def panel_C(ax, d: dict, cal: np.ndarray, stats_out: dict) -> None:
    order = level_order(d["level9"], d["year"])
    lv = d["level9"]
    means = np.array([cal[lv == L].mean() for L in order])
    sems = np.array([stats.sem(cal[lv == L]) if (lv == L).sum() > 1 else 0.0
                     for L in order])
    ns = [int((lv == L).sum()) for L in order]
    xs = np.arange(len(order))

    cols = []
    for L in order:
        gg = d["group"][lv == L]
        cols.append(PALETTE[max(set(gg), key=list(gg).count)])

    ax.axhline(0, color=MUTED, lw=0.4, zorder=0)
    ax.plot(xs, means, color=INK, lw=0.9, zorder=2)
    ax.errorbar(xs, means, yerr=sems, fmt="none", ecolor=INK,
                elinewidth=0.6, capsize=2, capthick=0.6, zorder=3)
    ax.scatter(xs, means, s=34, c=cols, edgecolor=INK, linewidth=0.5, zorder=4)

    rho, p = stats_out["spearman_rho"], stats_out["spearman_p"]
    ptxt = "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"
    ax.text(0.03, 0.94, f"ρ = {rho:+.3f}   ({ptxt})", transform=ax.transAxes,
            fontsize=7.5, color=INK, va="top")

    ax.set_xticks(xs)
    ax.set_xticklabels([L.upper() if len(L) <= 4 else L.capitalize()
                        for L in order], rotation=45, ha="right")
    lo = ax.get_ylim()[0]
    ax.set_ylim(lo - 0.12, ax.get_ylim()[1])
    for i, n in enumerate(ns):
        ax.annotate(f"n={n}" if i == 0 else f"{n}", (xs[i], lo - 0.04),
                    fontsize=5.5, color=MUTED, ha="center", va="center")
    ax.set_ylabel("Score calibré  (moyenne ± SEM)")
    ax.set_xlabel("Sous-niveau clinique")
    ax.set_title("C", loc="left", fontweight="bold", fontsize=10, pad=6)


def panel_D(ax, d: dict, stats_out: dict) -> None:
    contrasts = stats_out["contrasts"]
    keys = list(contrasts.keys())[::-1]
    ys = np.arange(len(keys))

    ax.axvline(0, color=MUTED, lw=0.6, zorder=0)
    ax.axvline(0.8, color=MUTED, lw=0.4, ls=(0, (2, 3)), zorder=0)
    ax.text(0.8, len(keys) - 0.28, "grand effet", fontsize=5.8, color=MUTED,
            ha="center")

    for i, k in enumerate(keys):
        c = contrasts[k]
        hi_lvl = k.split("-")[0].strip().lower()
        col = PALETTE.get(hi_lvl, INK)
        ax.plot([c["ci_low"], c["ci_high"]], [ys[i], ys[i]],
                color=INK, lw=0.8, solid_capstyle="butt", zorder=2)
        ax.scatter([c["g"]], [ys[i]], s=32, color=col, edgecolor=INK,
                   linewidth=0.5, zorder=3)
        if c["paper_d"] is not None:
            ax.scatter([c["paper_d"]], [ys[i] + 0.26], s=26, marker="D",
                       facecolor="none", edgecolor=REF, linewidth=0.9, zorder=3)

    ax.set_yticks(ys)
    ax.set_yticklabels([k.replace("-", " vs ") for k in keys])
    ax.set_xlabel("Taille d'effet  (g de Hedges, IC 95 %)")
    ax.set_ylim(-0.6, len(keys) - 0.2)
    ax.set_title("D", loc="left", fontweight="bold", fontsize=10, pad=6)

    handles = [
        Line2D([], [], marker="o", ls="none", color=PALETTE["senior"],
               markeredgecolor=INK, markersize=5, label="ce travail"),
        Line2D([], [], marker="D", ls="none", markerfacecolor="none",
               markeredgecolor=REF, markersize=4.5, label="Hybrid 1 (publié)"),
    ]
    ax.legend(handles=handles, loc="lower right", handletextpad=0.4,
              borderpad=0.2)


def compute_stats(d: dict, cal: np.ndarray, a: float, b: float) -> dict:
    s, g, yr = d["raw"], d["group"], d["year"]
    out: dict = {
        "n_participants": int(d["n"]),
        "calibration": {"slope_a": float(a), "intercept_b": float(b),
                        "compression_factor": float(a)},
        "group_means_raw": {}, "group_means_calibrated": {},
    }
    for k in GROUPS:
        out["group_means_raw"][k] = {
            "mean": float(s[g == k].mean()), "sd": float(s[g == k].std(ddof=1)),
            "n": int((g == k).sum())}
        out["group_means_calibrated"][k] = {
            "mean": float(cal[g == k].mean()),
            "sd": float(cal[g == k].std(ddof=1))}

    rho, p = stats.spearmanr(yr, s)
    out["spearman_rho"], out["spearman_p"] = float(rho), float(p)

    mid = np.isin(g, ["junior", "senior"])
    sl, ic, r, pv, se = stats.linregress(yr[mid], cal[mid])
    out["predictive_validity_middle"] = {
        "n": int(mid.sum()), "R2": float(r ** 2), "slope": float(sl),
        "intercept": float(ic), "p": float(pv)}

    paper_sd = {}
    for k, v in PAPER.items():
        if v.get("ci95_half") and v.get("n"):
            paper_sd[k] = v["ci95_half"] * np.sqrt(v["n"]) / 1.96

    contrasts = {}
    pairs = [("expert", "senior"), ("expert", "junior"), ("expert", "novice"),
             ("senior", "junior"), ("senior", "novice"), ("junior", "novice")]
    for hi, lo in pairs:
        x, y = s[g == hi], s[g == lo]
        gg, lo_ci, hi_ci = hedges_g(x, y)
        t, pval = stats.ttest_ind(x, y, equal_var=False)
        pd_ = None
        if hi in paper_sd or lo in paper_sd:
            sds = [paper_sd[k] for k in (hi, lo) if k in paper_sd]
            pooled = float(np.mean(sds))
            pd_ = (PAPER[hi]["mean"] - PAPER[lo]["mean"]) / pooled
        contrasts[f"{hi.capitalize()}-{lo.capitalize()}"] = {
            "g": float(gg), "ci_low": float(lo_ci), "ci_high": float(hi_ci),
            "p_welch": float(pval),
            "diff_calibrated": float(cal[g == hi].mean() - cal[g == lo].mean()),
            "paper_d": None if pd_ is None else float(pd_),
        }
    out["contrasts"] = contrasts
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--out", default=Path("results/figures"), type=Path)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()

    set_style()
    args.out.mkdir(parents=True, exist_ok=True)

    d = load(args.pred)
    cal, a, b = calibrate(d["raw"], d["group"])
    st = compute_stats(d, cal, a, b)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4), constrained_layout=True)
    panel_A(axes[0, 0], d, cal)
    panel_B(axes[0, 1], d, cal, a)
    panel_C(axes[1, 0], d, cal, st)
    panel_D(axes[1, 1], d, st)

    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"panel_ABCD_{args.tag}.{ext}")
    plt.close(fig)

    for name, fn, size in [
        ("panelA_scores", panel_A, (3.4, 3.0)),
        ("panelB_calibration", panel_B, (3.4, 3.0)),
        ("panelC_monotonicity", panel_C, (3.4, 3.0)),
        ("panelD_forest", panel_D, (3.4, 3.0)),
    ]:
        f, ax = plt.subplots(figsize=size)
        if name == "panelA_scores":
            fn(ax, d, cal)
        elif name == "panelB_calibration":
            fn(ax, d, cal, a)
        elif name == "panelC_monotonicity":
            fn(ax, d, cal, st)
        else:
            fn(ax, d, st)
        ax.set_title("")
        for ext in ("pdf", "png"):
            f.savefig(args.out / f"{name}_{args.tag}.{ext}")
        plt.close(f)

    (args.out / f"panel_stats_{args.tag}.json").write_text(json.dumps(st, indent=2))

    print(f"[ok] figures -> {args.out}")
    print(f"  pente de calibration a = {a:.2f}  (compression ×{a:.1f})")
    print(f"  Spearman rho = {st['spearman_rho']:+.3f}")
    print(f"  R2 middle    = {st['predictive_validity_middle']['R2']:.3f}")
    for k, c in st["contrasts"].items():
        ref = "" if c["paper_d"] is None else f"   [papier d≈{c['paper_d']:.2f}]"
        print(f"  {k:16s} g={c['g']:+.2f} [{c['ci_low']:+.2f},{c['ci_high']:+.2f}]"
              f"  p={c['p_welch']:.3g}{ref}")


if __name__ == "__main__":
    main()
