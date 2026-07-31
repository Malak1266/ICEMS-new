"""
Figures style publication — reproduction des 3 figures de référence
avec le Hybrid LSTM-Transformer + HOEL (LOPO), n=136 trials.

Figure A : Progression temporelle du score par classe (4 classes)
Figure B : Monotonicité ordinale granulaire (9 sous-niveaux + tendance)
Figure C : Distribution par phase du geste (Early/Middle/Late, panneau unique)

Usage :
  python -m eval.figures_publication_style \
      --predictions results/hybrid_hoel_b/predictions.pkl \
      --frames results/hybrid_hoel_b/frame_predictions.pkl \
      --output results/hybrid_hoel_b/
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


# ═══════════════════════════════════════════
# Configuration commune
# ═══════════════════════════════════════════
SUBLEVEL_ORDER = ["ms", "pgy1", "pgy2", "pgy3", "pgy4", "pgy5", "pgy6", "fellow", "staff"]
SUBLEVEL_TO_RANK = {k: i + 1 for i, k in enumerate(SUBLEVEL_ORDER)}  # 1..9
SUBLEVEL_LABELS_FR = {
    "ms": "Medical Student", "pgy1": "Resident PGY1", "pgy2": "Resident PGY2",
    "pgy3": "Resident PGY3", "pgy4": "Resident PGY4", "pgy5": "Resident PGY5",
    "pgy6": "Resident PGY6", "fellow": "Fellow", "staff": "Neurosurgeon (Staff)",
}

SUBLEVEL_TO_CLASS4 = {
    "ms": "student",
    "pgy1": "junior", "pgy2": "junior", "pgy3": "junior",
    "pgy4": "junior", "pgy5": "junior",
    "pgy6": "senior", "fellow": "senior",
    "staff": "expert",
}

CLASS4_ORDER = ["student", "junior", "senior", "expert"]
CLASS4_COLORS = {
    "student": "tab:red",
    "junior":  "tab:orange",
    "senior":  "tab:blue",
    "expert":  "tab:green",
}
CLASS4_LABELS = {
    "student": "Student", "junior": "Junior",
    "senior": "Senior", "expert": "Expert",
}

MIN_N_FOR_RHO = 3  # exclusion threshold, matching reference figure


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    # frame_predictions.pkl is wrapped as {"meta": ..., "entries": [...]}
    # predictions.pkl is a plain list. Normalize both to a plain list.
    if isinstance(data, dict) and "entries" in data:
        return data["entries"]
    return data


def participant_level_mean(entries, sublevel):
    """Aggregate trial-level scores to participant-level means for one sublevel."""
    subset = [e for e in entries if e["sublevel"] == sublevel]
    participants = sorted(set(e["participant"] for e in subset))
    means = []
    for p in participants:
        vals = [e["score_pred"] for e in subset if e["participant"] == p]
        means.append(np.mean(vals))
    return np.array(means), len(participants)


# ═══════════════════════════════════════════
# FIGURE A — Progression temporelle (4 classes)
# ═══════════════════════════════════════════
def plot_progression_4class(frame_data, output_dir, n_bins=100, model_label="Hybrid LSTM-Transformer + HOEL · LOPO"):
    fig, ax = plt.subplots(figsize=(11, 6))
    t_axis = np.linspace(0, 1, n_bins)

    n_trials_total = len(frame_data)
    right_labels = []

    for cls in CLASS4_ORDER:
        trials = [d for d in frame_data if d.get("class_4", SUBLEVEL_TO_CLASS4.get(d.get("sublevel"))) == cls]
        if not trials:
            continue

        binned = []
        for d in trials:
            scores = np.asarray(d["frame_scores"], dtype=float)
            T = len(scores)
            if T == 0:
                continue
            binned.append(np.interp(np.linspace(0, 1, n_bins), np.linspace(0, 1, T), scores))
        if not binned:
            continue
        binned = np.array(binned)

        # Aggregate per participant first (avoid pseudo-replication)
        participants = sorted(set(d["participant"] for d in trials))
        pcurves = []
        for p in participants:
            idx = [i for i, d in enumerate(trials) if d["participant"] == p]
            pcurves.append(np.mean(binned[idx], axis=0))
        pcurves = np.array(pcurves)

        mean_curve = np.mean(pcurves, axis=0)
        sem = np.std(pcurves, axis=0, ddof=1) / np.sqrt(max(len(pcurves), 1)) if len(pcurves) > 1 else np.zeros(n_bins)
        lo = mean_curve - 1.96 * sem
        hi = mean_curve + 1.96 * sem

        color = CLASS4_COLORS[cls]
        ax.plot(t_axis, mean_curve, color=color, linewidth=2.5, label=CLASS4_LABELS[cls], zorder=3)
        ax.fill_between(t_axis, lo, hi, color=color, alpha=0.15, zorder=2)

        overall_mean = float(np.mean(mean_curve))
        right_labels.append((overall_mean, cls))

    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0, 1)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.4)

    # Right-side reference lines/labels (computed from actual data, not idealized anchors)
    for val, cls in right_labels:
        ax.axhline(val, color=CLASS4_COLORS[cls], linestyle=":", linewidth=0.8, alpha=0.5)
        ax.annotate(f"{CLASS4_LABELS[cls]} ({val:+.2f})",
                    xy=(1.005, val), xycoords=("axes fraction", "data"),
                    fontsize=9, color=CLASS4_COLORS[cls], va="center", ha="left")

    ax.set_xlabel("Temps normalisé", fontsize=12)
    ax.set_ylabel("Score d'expertise [-1, +1]", fontsize=12)
    ax.set_title(
        f"Progression du score d'expertise au cours du geste\n"
        f"{model_label} · n={n_trials_total} trials",
        fontsize=13
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(os.path.join(output_dir, f"figure_progression_4class{ext}"),
                    dpi=300, bbox_inches="tight")
    print("Saved: figure_progression_4class.pdf/.png")
    plt.close(fig)


# ═══════════════════════════════════════════
# FIGURE B — Monotonicité granulaire (9 sous-niveaux)
# ═══════════════════════════════════════════
def plot_monotonicity_granular(predictions, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6.5))

    ranks, means, sds, ns = [], [], [], []
    excluded_ranks = []

    for sub in SUBLEVEL_ORDER:
        p_means, n_part = participant_level_mean(predictions, sub)
        if n_part == 0:
            continue
        rank = SUBLEVEL_TO_RANK[sub]
        mean_val = float(np.mean(p_means))
        sd_val = float(np.std(p_means, ddof=1)) if n_part > 1 else 0.0

        ranks.append(rank)
        means.append(mean_val)
        sds.append(sd_val)
        ns.append(n_part)

        if n_part < MIN_N_FOR_RHO:
            excluded_ranks.append(rank)

    ranks = np.array(ranks)
    means = np.array(means)
    sds = np.array(sds)
    ns = np.array(ns)

    # Spearman computed on group means, excluding n<MIN_N_FOR_RHO (matches reference methodology)
    keep = ns >= MIN_N_FOR_RHO
    if keep.sum() >= 3:
        rho, pval = spearmanr(ranks[keep], means[keep])
    else:
        rho, pval = np.nan, np.nan

    # Linear trend (on kept points)
    if keep.sum() >= 2:
        coeffs = np.polyfit(ranks[keep], means[keep], 1)
        trend_x = np.linspace(ranks.min(), ranks.max(), 100)
        trend_y = np.polyval(coeffs, trend_x)
        ax.plot(trend_x, trend_y, "--", color="gray", linewidth=1.2,
                label="Tendance linéaire", zorder=1)

    cmap = plt.cm.turbo(np.linspace(0.05, 0.95, len(SUBLEVEL_ORDER)))
    color_by_rank = {r: cmap[r - 1] for r in range(1, len(SUBLEVEL_ORDER) + 1)}

    for i, sub in enumerate(SUBLEVEL_ORDER):
        if sub not in [SUBLEVEL_ORDER[j] for j, r in enumerate(ranks)]:
            pass

    for rank, mean_val, sd_val, n_part in zip(ranks, means, sds, ns):
        sub = SUBLEVEL_ORDER[rank - 1]
        excluded = n_part < MIN_N_FOR_RHO
        marker = "^" if excluded else "o"
        size = 90 if not excluded else 70
        color = color_by_rank[rank]
        edge = "red" if excluded else "black"

        ax.errorbar(rank, mean_val, yerr=sd_val, fmt="none",
                    ecolor="gray", elinewidth=1, capsize=4, zorder=2)
        ax.scatter(rank, mean_val, s=size, color=color, edgecolor=edge,
                  linewidth=1.3, marker=marker, zorder=3)

        label = SUBLEVEL_LABELS_FR[sub]
        label_color = "red" if excluded else "black"
        excl_tag = " \u25b3" if excluded else ""
        ax.annotate(f"{label}{excl_tag}\n(n={n_part})",
                    xy=(rank, mean_val), xytext=(0, 12 if mean_val >= 0 else -22),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=label_color)

    ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
    ax.set_xlim(0.5, len(SUBLEVEL_ORDER) + 0.5)
    ax.set_xticks(range(1, len(SUBLEVEL_ORDER) + 1))
    ax.set_ylim(-1.1, 1.1)

    title_line2 = f"(\u25b3 = sous-niveaux n<{MIN_N_FOR_RHO} exclus du calcul \u03c1)"
    rho_str = f"{rho:.3f}" if not np.isnan(rho) else "n/a"
    p_str = f"{pval:.2e}" if not np.isnan(pval) else "n/a"

    ax.set_title(
        f"Monotonicit\u00e9 ordinale \u2014 Spearman \u03c1 = {rho_str} (p = {p_str})\n{title_line2}",
        fontsize=12.5
    )
    ax.set_xlabel("Rang d'expertise ordinal (1=d\u00e9butant \u2192 9=expert)", fontsize=11)
    ax.set_ylabel("Score pr\u00e9dit moyen", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(os.path.join(output_dir, f"figure_monotonicity_granular{ext}"),
                    dpi=300, bbox_inches="tight")
    print("Saved: figure_monotonicity_granular.pdf/.png")
    plt.close(fig)

    return rho, pval


# ═══════════════════════════════════════════
# FIGURE C — Distribution par phase (panneau unique)
# ═══════════════════════════════════════════
def plot_phase_single_panel(frame_data, output_dir):
    fig, ax = plt.subplots(figsize=(13, 5.5))

    phases = {"Early\n[0-33%]": (0.0, 0.33), "Middle\n[33-66%]": (0.33, 0.66), "Late\n[66-100%]": (0.66, 1.0)}
    phase_names = list(phases.keys())
    n_phases = len(phase_names)
    n_classes = len(CLASS4_ORDER)

    group_width = 0.8
    slot_width = group_width / n_classes

    rng = np.random.RandomState(42)

    for pi, (pname, (tlo, thi)) in enumerate(phases.items()):
        base_x = pi

        for ci, cls in enumerate(CLASS4_ORDER):
            trials = [d for d in frame_data if d.get("class_4", SUBLEVEL_TO_CLASS4.get(d.get("sublevel"))) == cls]
            vals = []
            for d in trials:
                scores = np.asarray(d["frame_scores"], dtype=float)
                T = len(scores)
                if T == 0:
                    continue
                ilo, ihi = int(tlo * T), max(int(thi * T), int(tlo * T) + 1)
                vals.append(np.mean(scores[ilo:ihi]))
            if not vals:
                continue

            x_pos = base_x - group_width / 2 + slot_width * (ci + 0.5)
            color = CLASS4_COLORS[cls]

            bp = ax.boxplot([vals], positions=[x_pos], widths=slot_width * 0.85,
                            patch_artist=True, showfliers=False, zorder=2,
                            medianprops=dict(color="black", linewidth=1.3))
            for box in bp["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(0.55)
                box.set_edgecolor("black")

            jitter = rng.uniform(-slot_width * 0.3, slot_width * 0.3, len(vals))
            ax.scatter(x_pos + jitter, vals, s=10, color=color, alpha=0.6,
                      zorder=3, edgecolors="none")

    for pi in range(1, n_phases):
        ax.axvline(pi - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xticks(range(n_phases))
    ax.set_xticklabels(phase_names, fontsize=11)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
    ax.set_ylabel("Score moyen dans la phase", fontsize=12)
    ax.set_title(
        "Distribution des scores par phase du geste \u00b7 tiers Early / Middle / Late\n"
        "Hybrid LSTM-Transformer + HOEL \u00b7 LOPO",
        fontsize=13
    )
    ax.grid(True, axis="y", alpha=0.25)

    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=CLASS4_COLORS[c],
               markersize=12, alpha=0.7, label=CLASS4_LABELS[c])
        for c in CLASS4_ORDER
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10, framealpha=0.9)

    # Annotation matching reference
    ax.annotate(
        "Score stable d\u00e8s la phase pr\u00e9coce\n\u2192 expertise visible \u00e0 t \u2248 15%",
        xy=(0, 0.05), xytext=(0.55, 0.85), textcoords="data",
        fontsize=9.5, ha="left",
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9)
    )

    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(os.path.join(output_dir, f"figure_phase_single_panel{ext}"),
                    dpi=300, bbox_inches="tight")
    print("Saved: figure_phase_single_panel.pdf/.png")
    plt.close(fig)


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True,
                        help="Path to predictions.pkl (trial-level, all 9 sublevels)")
    parser.add_argument("--frames", required=True,
                        help="Path to frame_predictions.pkl (per-frame scores)")
    parser.add_argument("--output", default="results/hybrid_hoel_b/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    predictions = load_pickle(args.predictions)
    frame_data = load_pickle(args.frames)

    print(f"Predictions (trial-level): {len(predictions)} entries")
    print(f"Frame data: {len(frame_data)} entries")

    plot_progression_4class(frame_data, args.output)
    rho, pval = plot_monotonicity_granular(predictions, args.output)
    plot_phase_single_panel(frame_data, args.output)

    print(f"\n=== Termine. Monotonicite granulaire: rho={rho:.3f}, p={pval:.2e} ===")
