"""Publication-ready figures for Hybrid extremes evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from evaluation_publication.config import (
    CLASS4_LABELS,
    CLASS4_ORDER,
    CLASS4_RANK,
    EXPORT_FORMATS,
    FIGURE_DPI,
    FIGURE_WIDTH,
    PALETTE,
    PHASE_LABELS,
    STYLE,
)
from evaluation_publication.metrics import phase_means, resolve_class4, score_to_class_idx
from evaluation_publication.statistics import pearson_r2, spearman_with_p


def apply_style() -> None:
    plt.rcParams.update(STYLE)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for ext in EXPORT_FORMATS:
        path = out_dir / f"{stem}.{ext}"
        kwargs = {"dpi": FIGURE_DPI} if ext == "png" else {}
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


def _legend_handles() -> List[Patch]:
    return [
        Patch(facecolor=PALETTE[c], edgecolor="#222", linewidth=0.6, label=CLASS4_LABELS[c])
        for c in CLASS4_ORDER
    ]


# ─── Figure 1 — Temporal stability ─────────────────────────────────────────────

def plot_temporal_stability(frame_entries: Sequence[dict], out_dir: Path) -> List[Path]:
    """Grouped boxplots of mean phase scores by expertise level."""
    apply_style()
    phases = phase_means(frame_entries)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 4.2))
    n_cls = len(CLASS4_ORDER)
    group_w = 0.78
    slot_w = group_w / n_cls
    rng = np.random.default_rng(42)

    for pi, phase in enumerate(PHASE_LABELS):
        for ci, cls in enumerate(CLASS4_ORDER):
            vals = phases[phase][cls]
            if not vals:
                continue
            x = pi - group_w / 2 + slot_w * (ci + 0.5)
            color = PALETTE[cls]

            bp = ax.boxplot(
                [vals],
                positions=[x],
                widths=slot_w * 0.82,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="#1a1a1a", linewidth=1.2),
                whiskerprops=dict(color="#444", linewidth=0.9),
                capprops=dict(color="#444", linewidth=0.9),
                boxprops=dict(linewidth=0.8),
                zorder=2,
            )
            for box in bp["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(0.72)
                box.set_edgecolor("#222")

            jitter = rng.uniform(-slot_w * 0.28, slot_w * 0.28, size=len(vals))
            ax.scatter(
                x + jitter, vals,
                s=11, c=color, alpha=0.55, edgecolors="none", zorder=3,
            )

    for boundary in range(1, len(PHASE_LABELS)):
        ax.axvline(boundary - 0.5, color="#bbbbbb", lw=0.7, ls="--", zorder=1)

    ax.axhline(0.0, color="#888888", lw=0.7, ls=":", zorder=1)
    ax.set_xticks(range(len(PHASE_LABELS)))
    ax.set_xticklabels(PHASE_LABELS)
    ax.set_ylabel("Mean score in phase")
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks(np.arange(-1.0, 1.01, 0.25))
    ax.set_title("Temporal Stability of Expertise Prediction")
    ax.legend(handles=_legend_handles(), frameon=False, loc="lower right")
    ax.set_xlim(-0.55, len(PHASE_LABELS) - 0.45)

    fig.tight_layout()
    paths = save_figure(fig, out_dir, "fig1_temporal_stability")
    plt.close(fig)
    return paths


# ─── Figure 2 — Ordinal monotonicity ───────────────────────────────────────────

def plot_ordinal_monotonicity(trial_preds: Sequence[dict], out_dir: Path) -> List[Path]:
    """Mean ± SD by rank. Spearman/R² are computed on ALL points (not 4 means)."""
    apply_style()

    by_cls: Dict[str, List[float]] = {c: [] for c in CLASS4_ORDER}
    ranks_pts: List[float] = []
    scores_pts: List[float] = []
    for p in trial_preds:
        cls = resolve_class4(p)
        s = float(p["score_pred"])
        by_cls[cls].append(s)
        ranks_pts.append(float(CLASS4_RANK[cls]))
        scores_pts.append(s)

    ranks, means, errs, colors, ns = [], [], [], [], []
    for cls in CLASS4_ORDER:
        vals = np.asarray(by_cls[cls], dtype=float)
        if vals.size == 0:
            continue
        ranks.append(CLASS4_RANK[cls])
        means.append(float(vals.mean()))
        errs.append(float(vals.std(ddof=1)) if vals.size > 1 else 0.0)
        colors.append(PALETTE[cls])
        ns.append(int(vals.size))

    ranks_arr = np.asarray(ranks, dtype=float)
    means_arr = np.asarray(means, dtype=float)
    # Participant/trial-level (not the 4 group means — that would force ρ≈1)
    rho, p_rho = spearman_with_p(
        np.asarray(ranks_pts, dtype=float),
        np.asarray(scores_pts, dtype=float),
    )
    slope, intercept, r2 = pearson_r2(
        np.asarray(ranks_pts, dtype=float),
        np.asarray(scores_pts, dtype=float),
    )

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 4.2))

    if len(ranks) >= 2:
        x_line = np.linspace(min(ranks) - 0.15, max(ranks) + 0.15, 80)
        ax.plot(
            x_line, slope * x_line + intercept,
            ls="--", color="#666666", lw=1.2, zorder=1, label="Linear trend",
        )

    ax.errorbar(
        ranks_arr, means_arr, yerr=errs,
        fmt="none", ecolor="#555555", elinewidth=1.0, capsize=4, zorder=2,
    )
    sizes = 55 + 18 * np.asarray(ns, dtype=float) / max(max(ns), 1)
    ax.scatter(
        ranks_arr, means_arr,
        s=sizes, c=colors, edgecolors="#1a1a1a", linewidths=0.8, zorder=3,
    )

    ax.axhline(0.0, color="#888888", lw=0.7, ls=":", zorder=1)
    ax.set_xticks(list(CLASS4_RANK.values()))
    ax.set_xticklabels([CLASS4_LABELS[c] for c in CLASS4_ORDER])
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks(np.arange(-1.0, 1.01, 0.25))
    ax.set_xlabel("Clinical expertise rank")
    ax.set_ylabel("Mean predicted score")

    rho_s = f"{rho:.3f}" if np.isfinite(rho) else "n/a"
    p_s = f"{p_rho:.2e}" if np.isfinite(p_rho) else "n/a"
    r2_s = f"{r2:.3f}" if np.isfinite(r2) else "n/a"
    ax.set_title(
        f"Ordinal Monotonicity  |  Spearman ρ = {rho_s}, p = {p_s}, R² = {r2_s}"
        f"  (n={len(scores_pts)} trials)"
    )
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    paths = save_figure(fig, out_dir, "fig2_ordinal_monotonicity")
    plt.close(fig)
    return paths


# ─── Figure 3 — Confusion matrix ───────────────────────────────────────────────

def plot_confusion_matrix(trial_preds: Sequence[dict], out_dir: Path) -> List[Path]:
    """Row-normalised (%) confusion matrix; absolute counts annotated beneath %."""
    apply_style()

    counts = np.zeros((4, 4), dtype=np.int64)
    for p in trial_preds:
        ti = CLASS4_ORDER.index(resolve_class4(p))
        pi = score_to_class_idx(float(p["score_pred"]))
        counts[ti, pi] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(
            counts.sum(axis=1, keepdims=True) > 0,
            100.0 * counts / counts.sum(axis=1, keepdims=True),
            0.0,
        )

    labels = [CLASS4_LABELS[c] for c in CLASS4_ORDER]
    acc = float(np.trace(counts) / counts.sum()) if counts.sum() else float("nan")

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH * 0.72, FIGURE_WIDTH * 0.62))
    im = ax.imshow(pct, cmap="Greys", vmin=0, vmax=100, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (%)")
    cbar.outline.set_visible(False)

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(4):
        for j in range(4):
            val = pct[i, j]
            n = counts[i, j]
            color = "white" if val >= 55 else "#1a1a1a"
            ax.text(
                j, i - 0.12, f"{val:.0f}%",
                ha="center", va="center", fontsize=10, color=color, fontweight="medium",
            )
            ax.text(
                j, i + 0.22, f"n={n}",
                ha="center", va="center", fontsize=7, color=color, alpha=0.85,
            )

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion Matrix  |  Accuracy = {100 * acc:.1f}%")
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    paths = save_figure(fig, out_dir, "fig3_confusion_matrix")
    plt.close(fig)
    return paths


def generate_all_figures(
    trial_preds: Sequence[dict],
    frame_entries: Sequence[dict],
    out_dir: Path,
) -> Dict[str, List[Path]]:
    out_dir = Path(out_dir)
    return {
        "fig1_temporal_stability": plot_temporal_stability(frame_entries, out_dir),
        "fig2_ordinal_monotonicity": plot_ordinal_monotonicity(trial_preds, out_dir),
        "fig3_confusion_matrix": plot_confusion_matrix(trial_preds, out_dir),
    }
