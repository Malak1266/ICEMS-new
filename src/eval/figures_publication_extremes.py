"""
Figures style publication pour l'approche TRAIN-EXTREMES -> TEST-MILIEU.

Distinction visuelle train (Student/Expert, appris) vs test (Junior/Senior, generalisation).
- Student + Expert : style hachure + label "(train)" -> ce sont des extremes appris
- Junior + Senior  : style plein -> vraie generalisation sur le milieu jamais vu

Figures :
  A : Progression temporelle 4 classes (train hachure, test plein)
  B : Monotonicite granulaire 9 sous-niveaux (extremes marques "train")
  C : Distribution par phase Early/Middle/Late (train hachure, test plein)

Usage :
  python -m eval.figures_publication_extremes \
      --predictions-middle results/hybrid_extremes/predictions_middle.pkl \
      --predictions-extremes results/hybrid_extremes/predictions_extremes.pkl \
      --frames results/hybrid_extremes/frame_predictions.pkl \
      --output results/hybrid_extremes/
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr


SUBLEVEL_ORDER = ["ms", "pgy1", "pgy2", "pgy3", "pgy4", "pgy5", "pgy6", "fellow", "staff"]
SUBLEVEL_TO_RANK = {k: i + 1 for i, k in enumerate(SUBLEVEL_ORDER)}
SUBLEVEL_LABELS_FR = {
    "ms": "Medical Student", "pgy1": "Resident PGY1", "pgy2": "Resident PGY2",
    "pgy3": "Resident PGY3", "pgy4": "Resident PGY4", "pgy5": "Resident PGY5",
    "pgy6": "Resident PGY6", "fellow": "Fellow", "staff": "Neurosurgeon (Staff)",
}
SUBLEVEL_TO_CLASS4 = {
    "ms": "student",
    "pgy1": "junior", "pgy2": "junior", "pgy3": "junior", "pgy4": "junior", "pgy5": "junior",
    "pgy6": "senior", "fellow": "senior",
    "staff": "expert",
}
# Which classes were in TRAIN (extremes) vs TEST (middle)
TRAIN_CLASSES = {"student", "expert"}
TEST_CLASSES = {"junior", "senior"}

CLASS4_ORDER = ["student", "junior", "senior", "expert"]
CLASS4_COLORS = {
    "student": "#C62828",
    "junior": "#EF6C00",
    "senior": "#6A1B9A",   # violet — bien séparé du vert Expert / Fellow
    "expert": "#1B5E20",
}
CLASS4_LABELS = {
    "student": "Student", "junior": "Junior",
    "senior": "Fellow / Senior", "expert": "Expert",
}
MIN_N_FOR_RHO = 3
PROGRESSION_N_BINS = 200
PROGRESSION_SMOOTH_SIGMA = 8


def _smooth_1d(y: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(y) < 3:
        return y
    return gaussian_filter1d(y, sigma=sigma, mode="nearest")


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "entries" in data:
        return data["entries"]
    return data


def load_all_entries(pred_middle_path, pred_extremes_path):
    """Merge middle (test) + extremes (train) predictions, tagging split."""
    entries = []
    middle = load_pickle(pred_middle_path)
    for e in middle:
        e = dict(e)
        e["split"] = "test"
        e.setdefault("class_4", SUBLEVEL_TO_CLASS4.get(e.get("sublevel")))
        entries.append(e)
    if pred_extremes_path and os.path.exists(pred_extremes_path):
        extremes = load_pickle(pred_extremes_path)
        for e in extremes:
            e = dict(e)
            e["split"] = "train"
            e.setdefault("class_4", SUBLEVEL_TO_CLASS4.get(e.get("sublevel")))
            entries.append(e)
    return entries


def participant_level_mean(entries, sublevel):
    subset = [e for e in entries if e["sublevel"] == sublevel]
    participants = sorted(set(e["participant"] for e in subset))
    means = []
    for p in participants:
        vals = [e["score_pred"] for e in subset if e["participant"] == p]
        means.append(np.mean(vals))
    return np.array(means), len(participants)


# ═══════════════════════════════════════════
# FIGURE A — Progression temporelle 4 classes (train hachure / test plein)
# ═══════════════════════════════════════════
def plot_progression_extremes(
    frame_data,
    output_dir,
    n_bins: int = PROGRESSION_N_BINS,
    smooth_sigma: float = PROGRESSION_SMOOTH_SIGMA,
):
    fig, ax = plt.subplots(figsize=(11, 6))
    t_axis = np.linspace(0, 1, n_bins)
    n_trials_total = len(frame_data)

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

        participants = sorted(set(d["participant"] for d in trials))
        pcurves = []
        for p in participants:
            idx = [i for i, d in enumerate(trials) if d["participant"] == p]
            pcurves.append(np.mean(binned[idx], axis=0))
        pcurves = np.array(pcurves)
        if smooth_sigma > 0:
            pcurves = np.array([_smooth_1d(c, smooth_sigma) for c in pcurves])

        mean_curve = _smooth_1d(np.mean(pcurves, axis=0), smooth_sigma)
        sem = np.std(pcurves, axis=0, ddof=1) / np.sqrt(max(len(pcurves), 1)) if len(pcurves) > 1 else np.zeros(n_bins)
        lo = _smooth_1d(mean_curve - 1.96 * sem, smooth_sigma)
        hi = _smooth_1d(mean_curve + 1.96 * sem, smooth_sigma)

        color = CLASS4_COLORS[cls]
        is_train = cls in TRAIN_CLASSES
        linestyle = "--" if is_train else "-"
        tag = " (train)" if is_train else " (test)"
        lw = 2.8 if cls == "expert" else 2.4
        z_line = 4 if cls in {"senior", "expert"} else 3
        alpha_fill = 0.08 if is_train else 0.14

        ax.plot(
            t_axis, mean_curve, color=color, linewidth=lw,
            linestyle=linestyle, label=CLASS4_LABELS[cls] + tag, zorder=z_line,
        )
        if is_train:
            ax.fill_between(
                t_axis, lo, hi, facecolor=color, edgecolor=color,
                hatch="///", linewidth=0.4, alpha=alpha_fill, zorder=z_line - 1,
            )
        else:
            ax.fill_between(t_axis, lo, hi, color=color, alpha=alpha_fill, zorder=z_line - 1)

    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0, 1)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
    ax.set_xlabel("Temps normalisé", fontsize=12)
    ax.set_ylabel("Score d'expertise [-1, +1]", fontsize=12)
    ax.set_title(
        "Progression du score d'expertise — TRAIN extrêmes → TEST milieu\n"
        f"Hybrid LSTM-Transformer + HOEL · n={n_trials_total} trials "
        "(--- = classes vues en train)",
        fontsize=12.5
    )
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(os.path.join(output_dir, f"figure_extremes_progression_4class{ext}"),
                    dpi=300, bbox_inches="tight")
    print("Saved: figure_extremes_progression_4class.pdf/.png")
    plt.close(fig)


# ═══════════════════════════════════════════
# FIGURE B — Monotonicite granulaire (extremes = "train")
# ═══════════════════════════════════════════
def plot_monotonicity_extremes(all_entries, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6.5))

    ranks, means, sds, ns, splits = [], [], [], [], []
    for sub in SUBLEVEL_ORDER:
        p_means, n_part = participant_level_mean(all_entries, sub)
        if n_part == 0:
            continue
        cls = SUBLEVEL_TO_CLASS4[sub]
        split = "train" if cls in TRAIN_CLASSES else "test"
        ranks.append(SUBLEVEL_TO_RANK[sub])
        means.append(float(np.mean(p_means)))
        sds.append(float(np.std(p_means, ddof=1)) if n_part > 1 else 0.0)
        ns.append(n_part)
        splits.append(split)

    ranks = np.array(ranks); means = np.array(means)
    sds = np.array(sds); ns = np.array(ns); splits = np.array(splits)

    # Spearman on TEST (middle) sublevels only, n>=MIN — that's the real generalization
    test_mask = (splits == "test") & (ns >= MIN_N_FOR_RHO)
    if test_mask.sum() >= 3:
        rho, pval = spearmanr(ranks[test_mask], means[test_mask])
    else:
        rho, pval = np.nan, np.nan

    # Trend on test points
    if test_mask.sum() >= 2:
        coeffs = np.polyfit(ranks[test_mask], means[test_mask], 1)
        tx = np.linspace(ranks.min(), ranks.max(), 100)
        ax.plot(tx, np.polyval(coeffs, tx), "--", color="gray", lw=1.2,
                label="Tendance (milieu, test)", zorder=1)

    cmap = plt.cm.turbo(np.linspace(0.05, 0.95, len(SUBLEVEL_ORDER)))
    for rank, mean_val, sd_val, n_part, split in zip(ranks, means, sds, ns, splits):
        sub = SUBLEVEL_ORDER[rank - 1]
        color = cmap[rank - 1]
        is_train = split == "train"
        excluded = (n_part < MIN_N_FOR_RHO) and not is_train

        ax.errorbar(rank, mean_val, yerr=sd_val, fmt="none",
                    ecolor="gray", elinewidth=1, capsize=4, zorder=2)

        if is_train:
            # train points: square marker + hatch style
            ax.scatter(rank, mean_val, s=130, color=color, edgecolor="black",
                       linewidth=1.5, marker="s", hatch="///", zorder=3)
            tag = " [train]"
            label_color = "dimgray"
        else:
            marker = "^" if excluded else "o"
            edge = "red" if excluded else "black"
            ax.scatter(rank, mean_val, s=90, color=color, edgecolor=edge,
                       linewidth=1.3, marker=marker, zorder=3)
            tag = " \u25b3" if excluded else ""
            label_color = "red" if excluded else "black"

        ax.annotate(f"{SUBLEVEL_LABELS_FR[sub]}{tag}\n(n={n_part})",
                    xy=(rank, mean_val), xytext=(0, 12 if mean_val >= 0 else -22),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=label_color)

    ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
    ax.set_xlim(0.5, len(SUBLEVEL_ORDER) + 0.5)
    ax.set_xticks(range(1, len(SUBLEVEL_ORDER) + 1))
    ax.set_ylim(-1.1, 1.1)

    rho_str = f"{rho:.3f}" if not np.isnan(rho) else "n/a"
    p_str = f"{pval:.2e}" if not np.isnan(pval) else "n/a"
    ax.set_title(
        f"Monotonicité ordinale — TRAIN extrêmes → TEST milieu\n"
        f"Spearman ρ (milieu jamais vu) = {rho_str} (p = {p_str})  ·  "
        "\u25a0 = extrêmes appris",
        fontsize=12
    )
    ax.set_xlabel("Rang d'expertise ordinal (1=débutant → 9=expert)", fontsize=11)
    ax.set_ylabel("Score prédit moyen", fontsize=11)

    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=11, label="Extrêmes (train)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=10, label="Milieu (test)"),
        Line2D([0], [0], linestyle="--", color="gray", label="Tendance (test)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(os.path.join(output_dir, f"figure_extremes_monotonicity_granular{ext}"),
                    dpi=300, bbox_inches="tight")
    print("Saved: figure_extremes_monotonicity_granular.pdf/.png")
    plt.close(fig)
    return rho, pval


# ═══════════════════════════════════════════
# FIGURE C — Distribution par phase (train hachure / test plein)
# ═══════════════════════════════════════════
def plot_phase_extremes(frame_data, output_dir):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    phases = {"Early\n[0-33%]": (0.0, 0.33), "Middle\n[33-66%]": (0.33, 0.66), "Late\n[66-100%]": (0.66, 1.0)}
    phase_names = list(phases.keys())
    n_classes = len(CLASS4_ORDER)
    group_width = 0.8
    slot_width = group_width / n_classes
    rng = np.random.RandomState(42)

    for pi, (pname, (tlo, thi)) in enumerate(phases.items()):
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

            x_pos = pi - group_width / 2 + slot_width * (ci + 0.5)
            color = CLASS4_COLORS[cls]
            is_train = cls in TRAIN_CLASSES

            bp = ax.boxplot([vals], positions=[x_pos], widths=slot_width * 0.85,
                            patch_artist=True, showfliers=False, zorder=2,
                            medianprops=dict(color="black", linewidth=1.3))
            for box in bp["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(0.55)
                box.set_edgecolor("black")
                if is_train:
                    box.set_hatch("///")

            jitter = rng.uniform(-slot_width * 0.3, slot_width * 0.3, len(vals))
            ax.scatter(x_pos + jitter, vals, s=10, color=color, alpha=0.6,
                       zorder=3, edgecolors="none")

    for pi in range(1, len(phase_names)):
        ax.axvline(pi - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xticks(range(len(phase_names)))
    ax.set_xticklabels(phase_names, fontsize=11)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
    ax.set_ylabel("Score moyen dans la phase", fontsize=12)
    ax.set_title(
        "Distribution par phase — TRAIN extrêmes → TEST milieu\n"
        "Hybrid LSTM-Transformer + HOEL  ·  hachures = classes vues en train",
        fontsize=12.5
    )
    ax.grid(True, axis="y", alpha=0.25)

    legend_elements = []
    for c in CLASS4_ORDER:
        is_train = c in TRAIN_CLASSES
        legend_elements.append(
            Patch(facecolor=CLASS4_COLORS[c], edgecolor="black", alpha=0.6,
                  hatch="///" if is_train else None,
                  label=CLASS4_LABELS[c] + (" (train)" if is_train else " (test)"))
        )
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9.5, framealpha=0.9)

    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(os.path.join(output_dir, f"figure_extremes_phase_single_panel{ext}"),
                    dpi=300, bbox_inches="tight")
    print("Saved: figure_extremes_phase_single_panel.pdf/.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-middle", required=True)
    parser.add_argument("--predictions-extremes", default=None,
                        help="Optional: extremes (train) predictions for monotonicity anchors")
    parser.add_argument("--frames", required=True,
                        help="frame_predictions.pkl for the extremes run (all classes)")
    parser.add_argument("--output", default="results/hybrid_extremes/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    all_entries = load_all_entries(args.predictions_middle, args.predictions_extremes)
    frame_data = load_pickle(args.frames)

    print(f"Entries (middle+extremes): {len(all_entries)}")
    print(f"Frame data: {len(frame_data)}")

    plot_progression_extremes(frame_data, args.output)
    rho, pval = plot_monotonicity_extremes(all_entries, args.output)
    plot_phase_extremes(frame_data, args.output)
    print(f"\n=== Termine. Monotonicite milieu: rho={rho}, p={pval} ===")
