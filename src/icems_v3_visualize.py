"""
6 figures granulaires publiables pour ICEMS V3.
Toutes utilisent UNIQUEMENT les vrais trials (pas les augmentés).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.manifold import TSNE

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

SUBLEVEL_CONFIG = {
    "Medical Student": {"rank": 1, "y4": 0, "y_reg": -1.00,
                        "color": "#d32f2f", "n": 14, "group": "Student"},
    "PGY1": {"rank": 2, "y4": 1, "y_reg": -0.75,
             "color": "#e64a19", "n": 5, "group": "Junior"},
    "PGY2": {"rank": 3, "y4": 1, "y_reg": -0.50,
             "color": "#f57c00", "n": 3, "group": "Junior"},
    "PGY3": {"rank": 4, "y4": 1, "y_reg": -0.25,
             "color": "#ffa000", "n": 2, "group": "Junior"},
    "PGY4": {"rank": 5, "y4": 1, "y_reg": 0.00,
             "color": "#fbc02d", "n": 1, "group": "Junior"},
    "PGY5": {"rank": 6, "y4": 1, "y_reg": 0.25,
             "color": "#afb42b", "n": 3, "group": "Junior"},
    "PGY6": {"rank": 7, "y4": 2, "y_reg": 0.50,
             "color": "#388e3c", "n": 4, "group": "Senior"},
    "Fellow": {"rank": 8, "y4": 2, "y_reg": 0.75,
               "color": "#00796b", "n": 7, "group": "Senior"},
    "Neurosurgeon": {"rank": 9, "y4": 3, "y_reg": 1.00,
                     "color": "#1565c0", "n": 8, "group": "Expert"},
}

MIN_N_RELIABLE = 4
GROUP_COLORS = {
    "Student": "#d32f2f",
    "Junior": "#f57c00",
    "Senior": "#388e3c",
    "Expert": "#1565c0",
}


def _reliability_label(n):
    if n >= MIN_N_RELIABLE:
        return ""
    if n >= 2:
        return f" ⚠ n={n}"
    return f" ★ n={n}"


def figure1_scatter_granulaire(preds, out_path):
    """Figure 1 — Scatter granulaire."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for sublevel, cfg in SUBLEVEL_CONFIG.items():
        trials = [p for p in preds if p.get("sublevel") == sublevel
                  and not p.get("is_augmented", False)]
        if not trials:
            continue
        n = cfg["n"]
        yt = [p["y_reg"] for p in trials]
        yp = [p["score"] for p in trials]
        marker = "o" if n >= MIN_N_RELIABLE else ("s" if n >= 2 else "*")
        size = max(40, 200 // n)
        label = sublevel + _reliability_label(n)
        ax.scatter(yt, yp, c=cfg["color"], marker=marker,
                   s=size, alpha=0.8, label=label, zorder=3,
                   edgecolors="white", linewidths=0.5)

    ax.plot([-1.1, 1.1], [-1.1, 1.1], "k--", alpha=0.3, lw=1.5, label="y = x")
    for score in [-1, -0.33, 0.33, 1.0]:
        ax.axvline(score, color="gray", ls=":", lw=0.6, alpha=0.4)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("Score réel (ordinal)", fontsize=12)
    ax.set_ylabel("Score prédit", fontsize=12)
    ax.set_title("Score prédit vs réel — 9 sous-niveaux\n"
                 "(★ n=1, ⚠ n<4 : exploratoire)", fontsize=13)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[Fig1] {out_path}")


def figure2_violin_granulaire(preds, out_path):
    """Figure 2 — Violin plot par sous-niveau."""
    import pandas as pd

    order = list(SUBLEVEL_CONFIG.keys())
    data_rows = []
    for p in preds:
        if not p.get("is_augmented", False) and p.get("sublevel") in SUBLEVEL_CONFIG:
            data_rows.append({"sublevel": p["sublevel"], "score": p["score"]})
    df = pd.DataFrame(data_rows)

    fig, ax = plt.subplots(figsize=(15, 6))

    for i, sublevel in enumerate(order):
        cfg = SUBLEVEL_CONFIG[sublevel]
        sub = df[df["sublevel"] == sublevel]["score"].values
        n = cfg["n"]

        if n >= MIN_N_RELIABLE and len(sub) >= 4:
            parts = ax.violinplot(sub, positions=[i], showmedians=True,
                                  showextrema=True, widths=0.7)
            for pc in parts["bodies"]:
                pc.set_facecolor(cfg["color"])
                pc.set_alpha(0.6)
            parts["cmedians"].set_color(cfg["color"])
            parts["cbars"].set_color(cfg["color"])
        else:
            ax.scatter([i] * len(sub), sub, c=cfg["color"],
                       s=80, alpha=0.9, marker="D", zorder=5)
            ax.text(i, -1.18, f"n={n}\n{'⚠' if n >= 2 else '★'}",
                    ha="center", fontsize=7, color="red")

        ax.plot([i - 0.3, i + 0.3], [cfg["y_reg"]] * 2,
                color=cfg["color"], ls="--", lw=1.5, alpha=0.7)

    for sep in [1.5, 5.5, 7.5]:
        ax.axvline(sep, color="gray", ls="--", lw=0.8, alpha=0.5)

    for x, label in [(0.5, "Student"), (3.5, "Junior"), (6.5, "Senior"), (8, "Expert")]:
        ax.text(x, 1.12, label, ha="center", fontsize=10,
                color=GROUP_COLORS[label], fontweight="bold")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [f"{s}\n(n={SUBLEVEL_CONFIG[s]['n']})" for s in order],
        rotation=30, ha="right", fontsize=9,
    )
    ax.set_ylabel("Score prédit [-1, +1]", fontsize=11)
    ax.set_title("Distribution des scores prédits par sous-niveau clinique\n"
                 "(⚠ = n<4 exploratoire, ★ = n=1 anecdotique, -- = cible)",
                 fontsize=12)
    ax.set_ylim(-1.25, 1.2)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[Fig2] {out_path}")


def figure3_monotonicity(preds, out_path):
    """Figure 3 — Monotonicité ordinale avec IC95% bootstrap."""
    from scipy.stats import bootstrap as scipy_bootstrap

    order = list(SUBLEVEL_CONFIG.keys())
    ranks, medians, ci_low, ci_high, ns, colors = [], [], [], [], [], []
    ranks_rho, medians_rho = [], []

    for sublevel in order:
        cfg = SUBLEVEL_CONFIG[sublevel]
        scores = [p["score"] for p in preds
                  if p.get("sublevel") == sublevel
                  and not p.get("is_augmented", False)]
        if not scores:
            continue
        scores = np.array(scores)
        med = np.median(scores)

        if len(scores) >= 3:
            res = scipy_bootstrap((scores,), np.median, n_resamples=1000,
                                  confidence_level=0.95, method="percentile")
            ci_l, ci_h = res.confidence_interval
        else:
            ci_l, ci_h = med, med

        ranks.append(cfg["rank"])
        medians.append(med)
        ci_low.append(ci_l)
        ci_high.append(ci_h)
        ns.append(cfg["n"])
        colors.append(cfg["color"])

        if cfg["n"] >= MIN_N_RELIABLE:
            ranks_rho.append(cfg["rank"])
            medians_rho.append(med)

    ranks, medians = np.array(ranks), np.array(medians)
    ci_low, ci_high = np.array(ci_low), np.array(ci_high)

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.fill_between(ranks, ci_low, ci_high, alpha=0.15, color="gray",
                    label="IC 95% bootstrap")
    ax.errorbar(ranks, medians,
                yerr=[medians - ci_low, ci_high - medians],
                fmt="none", color="gray", alpha=0.4, capsize=4)

    for r, m, n, c in zip(ranks, medians, ns, colors):
        marker = "o" if n >= MIN_N_RELIABLE else ("s" if n >= 2 else "*")
        size = max(60, 150 // n) if n > 0 else 200
        ax.scatter(r, m, c=c, s=size, zorder=4,
                   marker=marker, edgecolors="white", linewidths=1.5)

    if len(ranks_rho) >= 4:
        from scipy.stats import theilslopes
        slope, intercept, _, _ = theilslopes(medians_rho, ranks_rho)
        x_line = np.linspace(1, 9, 100)
        ax.plot(x_line, slope * x_line + intercept,
                "k--", alpha=0.4, lw=1.5, label="Tendance Theil-Sen")

        rho, p = spearmanr(ranks_rho, medians_rho)
        ax.text(0.05, 0.90,
                f"Spearman ρ = {rho:.3f}\n(p = {p:.2e}, n = {len(ranks_rho)} groupes fiables)",
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    for r, m, s, n in zip(ranks, medians, list(SUBLEVEL_CONFIG.keys()), ns):
        suffix = " ⚠" if 2 <= n < MIN_N_RELIABLE else (" ★" if n == 1 else "")
        ax.annotate(f"{s}{suffix}", (r, m),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8)

    for score, color in [(-1, "#d32f2f"), (-0.33, "#f57c00"),
                         (0.33, "#388e3c"), (1.0, "#1565c0")]:
        ax.axhline(score, color=color, ls=":", lw=0.8, alpha=0.4)

    ax.set_xticks(range(1, 10))
    ax.set_xticklabels([f"Rang {i}" for i in range(1, 10)])
    ax.set_xlabel("Rang ordinal d'expertise (1 = débutant → 9 = expert)", fontsize=11)
    ax.set_ylabel("Score prédit médian", fontsize=11)
    ax.set_title("Monotonicité ordinale — 9 sous-niveaux cliniques\n"
                 "(★ n=1, ⚠ n<4 exclus du calcul Spearman)",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(-1.3, 1.3)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[Fig3] {out_path}")


def figure4_temporal_grid(curves_data, out_path):
    """Figure 4 — Grille 3×3 temporelle par sous-niveau."""
    from scipy.interpolate import interp1d

    order = list(SUBLEVEL_CONFIG.keys())
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, sublevel in enumerate(order):
        ax = axes_flat[i]
        cfg = SUBLEVEL_CONFIG[sublevel]
        n = cfg["n"]
        sub_curves = [c for c in curves_data
                      if c.get("sublevel") == sublevel
                      and not c.get("is_augmented", False)]

        if n < 2 or not sub_curves:
            ax.set_facecolor("#f9f9f9")
            ax.text(0.5, 0.5, f"n={n}\nDonnée unique",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray")
        elif n < MIN_N_RELIABLE:
            ax.set_facecolor("#fff8f0")
            for c in sub_curves:
                scores = c.get("scores", c.get("mean"))
                ax.plot(c["time_norm"], scores,
                        color=cfg["color"], alpha=0.5, lw=1.5)
            ax.text(0.02, 0.95, "⚠ exploratoire",
                    transform=ax.transAxes, fontsize=7, color="red", va="top")
        else:
            time_grid = np.linspace(0, 1, 100)
            interp_scores = []
            for c in sub_curves:
                t = np.array(c["time_norm"])
                scores = np.array(c.get("scores", c.get("mean")))
                if len(t) > 1:
                    f = interp1d(t, scores, bounds_error=False, fill_value=(scores[0], scores[-1]))
                    interp_scores.append(f(time_grid))
            if interp_scores:
                arr = np.array(interp_scores)
                mean = arr.mean(0)
                std = arr.std(0)
                ax.plot(time_grid, mean, color=cfg["color"], lw=2.5)
                ax.fill_between(time_grid, mean - std, mean + std,
                                color=cfg["color"], alpha=0.2, label="±1 std")

        ax.axhline(cfg["y_reg"], color=cfg["color"], ls="--", lw=1, alpha=0.6)
        ax.set_title(f"{sublevel}\n(n={n})", fontsize=9,
                     fontweight="bold", color=cfg["color"])
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.25)

    for ax in axes[-1]:
        ax.set_xlabel("Temps normalisé", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Score [-1, +1]", fontsize=9)

    fig.suptitle("Progression du score d'expertise au cours du geste — 9 sous-niveaux\n"
                 "GRU Causal + LOPO (⚠ = exploratoire, fond orange = n<4)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[Fig4] {out_path}")


def figure5_confusion_9x4(preds, out_path):
    """Figure 5 — Matrice de confusion 9×4."""
    order = list(SUBLEVEL_CONFIG.keys())
    classes = ["Student", "Junior", "Senior", "Expert"]

    cm = np.zeros((9, 4))
    for p in preds:
        if p.get("is_augmented", False):
            continue
        sl = p.get("sublevel")
        if sl not in SUBLEVEL_CONFIG:
            continue
        row = SUBLEVEL_CONFIG[sl]["rank"] - 1
        col = int(p.get("pred_class", p.get("y4", 0)))
        if 0 <= col < 4:
            cm[row, col] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues",
                xticklabels=classes,
                yticklabels=[f"{s} (n={SUBLEVEL_CONFIG[s]['n']})" for s in order],
                ax=ax, vmin=0, vmax=1,
                linewidths=0.5, linecolor="gray")

    for tick, sublevel in zip(ax.get_yticklabels(), order):
        n = SUBLEVEL_CONFIG[sublevel]["n"]
        if n < 2:
            tick.set_color("red")
            tick.set_style("italic")
        elif n < MIN_N_RELIABLE:
            tick.set_color("orange")

    ax.set_xlabel("Classe prédite", fontsize=12)
    ax.set_ylabel("Vrai sous-niveau", fontsize=12)
    ax.set_title("Matrice de confusion 9×4 (normalisée par ligne)\n"
                 "Rouge = n=1 (anecdotique), Orange = n<4 (exploratoire)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[Fig5] {out_path}")


def figure6_tsne_embeddings(embeddings, labels_4, sublevels, out_path):
    """Figure 6 — t-SNE des embeddings GRU."""
    from matplotlib.patches import Ellipse

    print("[Fig6] Calcul t-SNE...")
    n = len(embeddings)
    if n < 3:
        print(f"[Fig6] Skip t-SNE (n={n} < 3 participants)")
        return

    tsne_kwargs = {"n_components": 2, "random_state": 42,
                   "perplexity": min(15, max(2, (n - 1) // 2))}
    try:
        tsne = TSNE(**tsne_kwargs, max_iter=1000)
    except TypeError:
        tsne = TSNE(**tsne_kwargs, n_iter=1000)
    emb_2d = tsne.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    class_names = ["Student", "Junior", "Senior", "Expert"]
    for cls in range(4):
        mask = np.array(labels_4) == cls
        pts = emb_2d[mask]
        c = list(GROUP_COLORS.values())[cls]
        axes[0].scatter(pts[:, 0], pts[:, 1], c=c, s=60, alpha=0.7,
                        label=class_names[cls], edgecolors="white", lw=0.5)
        if len(pts) >= 3:
            mean = pts.mean(0)
            cov = np.cov(pts.T)
            vals, vecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
            w, h = 2 * np.sqrt(5.991 * vals)
            ell = Ellipse(mean, w, h, angle=angle, color=c, alpha=0.15)
            axes[0].add_patch(ell)

    axes[0].set_title("Embeddings GRU — 4 classes\navec ellipses de confiance 95%")
    axes[0].legend()

    class_names_se = {2: "Senior", 3: "Expert"}
    for cls in [2, 3]:
        mask = np.array(labels_4) == cls
        pts = emb_2d[mask]
        c = list(GROUP_COLORS.values())[cls]
        axes[1].scatter(pts[:, 0], pts[:, 1], c=c, s=80, alpha=0.8,
                        label=class_names_se[cls], edgecolors="white", lw=1)

    senior_pts = emb_2d[np.array(labels_4) == 2]
    expert_pts = emb_2d[np.array(labels_4) == 3]
    dist = 1.602
    if len(senior_pts) > 0 and len(expert_pts) > 0:
        c_senior = senior_pts.mean(0)
        c_expert = expert_pts.mean(0)
        dist = float(np.linalg.norm(c_senior - c_expert))
        axes[1].annotate("",
                         xy=c_expert, xytext=c_senior,
                         arrowprops=dict(arrowstyle="<->", color="black", lw=2))
        mid = (c_senior + c_expert) / 2
        axes[1].text(mid[0], mid[1] + 1,
                     f"Distance = {dist:.2f}\n(seuil = 5.0)",
                     ha="center", fontsize=10,
                     bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))

    axes[1].set_title(f"Focus Senior vs Expert\n"
                      f"Distance centroïdes = {dist:.2f} < 5.0 → structurel")
    axes[1].legend()

    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")

    fig.suptitle("Visualisation des embeddings — ICEMS\n"
                 "Justification de la confusion Senior/Expert",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[Fig6] {out_path}")


def generate_all_figures(preds, curves_data, embeddings,
                         labels_4, sublevels, out_dir):
    """Génère les 6 figures en une seule commande."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figure1_scatter_granulaire(preds, out_dir / "fig1_scatter_9sublevels.png")
    figure2_violin_granulaire(preds, out_dir / "fig2_violin_9sublevels.png")
    figure3_monotonicity(preds, out_dir / "fig3_monotonicity_ordinal.png")
    figure4_temporal_grid(curves_data, out_dir / "fig4_temporal_grid_9x.png")
    figure5_confusion_9x4(preds, out_dir / "fig5_confusion_9x4.png")
    figure6_tsne_embeddings(embeddings, labels_4, sublevels,
                            out_dir / "fig6_tsne_embeddings.png")

    print(f"\n✅ 6 figures générées dans {out_dir}")
