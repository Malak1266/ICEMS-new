"""
plot_expertise_progression.py
=============================
Figure de progression ordonnée du score d'expertise prédit (6 groupes).

Usage :
    from plot_expertise_progression import plot_expertise_progression
    stats = plot_expertise_progression(preds, participant_metadata, out_path="results/progression")
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import kruskal, mannwhitneyu, spearmanr

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

GROUP_LABELS = [
    "Student",
    "Junior\nprécoce",
    "Junior\ntardif",
    "Senior\nrésident",
    "Senior\nFellow",
    "Expert",
]

GROUP_COLORS = [
    "#08306b",
    "#2171b5",
    "#6baed6",
    "#fdae6b",
    "#e6550d",
    "#a50f15",
]

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
N_ADJACENT_TESTS = 5


def _normalize_pgy(pgy_level: Optional[str]) -> Optional[str]:
    if pgy_level is None:
        return None
    s = str(pgy_level).strip().lower()
    if not s or s in {"none", "nan", "null", "-"}:
        return None
    if "fellow" in s:
        return "fellow"
    m = re.search(r"pgy\s*(\d)", s)
    if m:
        return f"pgy{m.group(1)}"
    if s.isdigit() and len(s) == 1:
        return f"pgy{s}"
    return s


def assign_progression_group(
    group_coarse: str,
    pgy_level: Optional[str],
) -> int:
    """Retourne l'index ordinal 0–5 pour un participant."""
    gc = str(group_coarse).strip()
    pgy = _normalize_pgy(pgy_level)

    if gc == "Student":
        return 0
    if gc == "Expert":
        return 5
    if gc == "Junior":
        if pgy in {"pgy1", "pgy2"}:
            return 1
        if pgy in {"pgy3", "pgy4", "pgy5"}:
            return 2
        raise ValueError(
            f"Junior sans PGY reconnu : group_coarse={gc!r}, pgy_level={pgy_level!r}"
        )
    if gc == "Senior":
        if pgy == "pgy6":
            return 3
        if pgy == "fellow":
            return 4
        raise ValueError(
            f"Senior sans sous-niveau reconnu : group_coarse={gc!r}, pgy_level={pgy_level!r}"
        )
    raise ValueError(f"group_coarse inconnu : {group_coarse!r}")


def _resolve_participant_meta(
    pid: str,
    participant_metadata: Dict[str, dict],
    trial_rows: List[dict],
) -> dict:
    if pid in participant_metadata:
        return participant_metadata[pid]
    if trial_rows:
        row = trial_rows[0]
        return {
            "group_coarse": row.get("group_coarse"),
            "pgy_level": row.get("pgy_level"),
            "dominant_hand": row.get("dominant_hand"),
        }
    raise KeyError(f"Participant {pid!r} absent de participant_metadata et de preds.")


def _participant_medians(preds: Sequence[dict]) -> Dict[str, List[float]]:
    by_pid: Dict[str, List[float]] = defaultdict(list)
    for row in preds:
        by_pid[str(row["pid"])].append(float(row["score"]))
    return {pid: scores for pid, scores in by_pid.items()}


def _bootstrap_median_ci(
    values: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:
    """Médiane + IC 95% bootstrap sur les médianes participants."""
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        v = float(values[0])
        return v, v, v

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots[i] = np.median(sample)
    return (
        float(np.median(values)),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )


def _print_statistics(
    group_scores: List[List[float]],
    participant_ranks: np.ndarray,
    participant_medians: np.ndarray,
) -> dict:
    rho, p_spearman = spearmanr(participant_ranks, participant_medians)

    non_empty = [g for g in group_scores if len(g) > 0]
    if len(non_empty) >= 2:
        h_stat, p_kw = kruskal(*non_empty)
    else:
        h_stat, p_kw = float("nan"), float("nan")

    print("\n" + "=" * 60)
    print(" Statistiques — progression d'expertise prédite")
    print("=" * 60)
    print(f"  Spearman ρ (rang 0–5 vs score médian/participant) : {rho:+.4f}  (p = {p_spearman:.4e})")
    print(f"  Kruskal-Wallis (6 groupes)                         : H = {h_stat:.4f}  (p = {p_kw:.4e})")

    print("\n  Mann-Whitney post-hoc (paires adjacentes, Bonferroni ×5) :")
    mw_rows = []
    for i in range(len(group_scores) - 1):
        a, b = group_scores[i], group_scores[i + 1]
        label = f"{GROUP_LABELS[i].replace(chr(10), ' ')} vs {GROUP_LABELS[i + 1].replace(chr(10), ' ')}"
        if len(a) == 0 or len(b) == 0:
            print(f"    {label:<45}  n/a (groupe vide)")
            mw_rows.append({"pair": label, "u": float("nan"), "p_raw": float("nan"), "p_adj": float("nan")})
            continue
        res = mannwhitneyu(a, b, alternative="two-sided")
        p_adj = min(float(res.pvalue) * N_ADJACENT_TESTS, 1.0)
        print(
            f"    {label:<45}  U = {res.statistic:.1f}  "
            f"p = {res.pvalue:.4e}  p_adj = {p_adj:.4e}"
        )
        mw_rows.append(
            {
                "pair": label,
                "u": float(res.statistic),
                "p_raw": float(res.pvalue),
                "p_adj": p_adj,
            }
        )

    return {
        "spearman_rho": float(rho),
        "spearman_p": float(p_spearman),
        "kruskal_h": float(h_stat),
        "kruskal_p": float(p_kw),
        "mann_whitney_adjacent": mw_rows,
    }


def plot_expertise_progression(
    preds: Sequence[dict],
    participant_metadata: Dict[str, dict],
    out_path: Optional[Union[str, Path]] = None,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """
    Violin plot de la progression ordonnée du score d'expertise prédit.

    Parameters
    ----------
    preds
        Liste de dicts avec clés ``pid``, ``trial``, ``score``,
        ``group_coarse``, ``pgy_level``.
    participant_metadata
        Mapping pid → {group_coarse, pgy_level, dominant_hand}.
    out_path
        Préfixe de sortie (sans extension) pour PDF et PNG.
        Si None, sauvegarde ``expertise_progression.{pdf,png}`` dans le cwd.

    Returns
    -------
    dict
        Résumé numérique (médianes par groupe, IC bootstrap, tests).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials_by_pid: Dict[str, List[dict]] = defaultdict(list)
    for row in preds:
        trials_by_pid[str(row["pid"])].append(row)

    participant_rows: List[dict] = []
    for pid, trial_rows in sorted(trials_by_pid.items()):
        meta = _resolve_participant_meta(pid, participant_metadata, trial_rows)
        scores = [float(r["score"]) for r in trial_rows]
        group_idx = assign_progression_group(meta["group_coarse"], meta.get("pgy_level"))
        participant_rows.append(
            {
                "pid": pid,
                "median_score": float(np.median(scores)),
                "group_idx": group_idx,
                "n_trials": len(scores),
            }
        )

    group_scores: List[List[float]] = [[] for _ in range(6)]
    for row in participant_rows:
        group_scores[row["group_idx"]].append(row["median_score"])

    group_summaries = []
    for idx, scores in enumerate(group_scores):
        arr = np.asarray(scores, dtype=float)
        med, lo, hi = _bootstrap_median_ci(arr, seed=seed + idx)
        group_summaries.append(
            {
                "group_idx": idx,
                "label": GROUP_LABELS[idx],
                "n_participants": len(scores),
                "median": med,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "scores": scores,
            }
        )

    participant_ranks = np.array([r["group_idx"] for r in participant_rows], dtype=float)
    participant_medians = np.array([r["median_score"] for r in participant_rows], dtype=float)
    stats = _print_statistics(group_scores, participant_ranks, participant_medians)

    rng = np.random.default_rng(seed)
    positions = np.arange(6)

    fig, ax = plt.subplots(figsize=(8, 5))

    violin_data = [np.asarray(g, dtype=float) for g in group_scores if len(g) > 0]
    violin_pos = [i for i, g in enumerate(group_scores) if len(g) > 0]
    if violin_data:
        parts = ax.violinplot(
            violin_data,
            positions=violin_pos,
            widths=0.55,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for i, body in enumerate(parts["bodies"]):
            gidx = violin_pos[i]
            body.set_facecolor(GROUP_COLORS[gidx])
            body.set_edgecolor("#333333")
            body.set_alpha(0.55)
        parts["cmedians"].set_color("#111111")
        parts["cmedians"].set_linewidth(1.5)

    for gidx, scores in enumerate(group_scores):
        if not scores:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(scores))
        ax.scatter(
            np.full(len(scores), gidx) + jitter,
            scores,
            s=36,
            c=GROUP_COLORS[gidx],
            edgecolors="#222222",
            linewidths=0.6,
            alpha=0.9,
            zorder=3,
        )

    x_trend = []
    y_trend = []
    y_err_lo = []
    y_err_hi = []
    for summary in group_summaries:
        if summary["n_participants"] == 0:
            continue
        x_trend.append(summary["group_idx"])
        y_trend.append(summary["median"])
        y_err_lo.append(summary["median"] - summary["ci95_lo"])
        y_err_hi.append(summary["ci95_hi"] - summary["median"])

    if len(x_trend) >= 2:
        x_arr = np.asarray(x_trend, dtype=float)
        y_arr = np.asarray(y_trend, dtype=float)
        coef = np.polyfit(x_arr, y_arr, deg=1)
        x_line = np.linspace(0, 5, 100)
        y_line = np.polyval(coef, x_line)
        ax.plot(
            x_line,
            y_line,
            color="#444444",
            linestyle="-",
            linewidth=1.8,
            alpha=0.85,
            label="Tendance linéaire",
            zorder=2,
        )
        ax.errorbar(
            x_arr,
            y_arr,
            yerr=[y_err_lo, y_err_hi],
            fmt="D",
            color="#111111",
            markersize=5,
            capsize=3,
            elinewidth=1.2,
            zorder=4,
            label="Médiane groupe (IC 95%)",
        )

    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.2, alpha=0.8, zorder=1)

    ax.set_xticks(positions)
    ax.set_xticklabels(GROUP_LABELS, rotation=30, ha="right")
    ax.set_xlim(-0.6, 5.6)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Predicted Expertise Score")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)

    text = (
        f"Spearman ρ = {stats['spearman_rho']:+.3f}\n"
        f"p = {stats['spearman_p']:.3g}"
    )
    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#888888", "alpha": 0.95},
    )

    fig.tight_layout()

    stem = Path(out_path) if out_path is not None else Path("expertise_progression")
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Figure sauvegardée : {pdf_path}")
    print(f"                       {png_path}")

    stats["group_summaries"] = group_summaries
    stats["participant_rows"] = participant_rows
    stats["pdf_path"] = str(pdf_path)
    stats["png_path"] = str(png_path)
    return stats


if __name__ == "__main__":
    demo_preds = []
    demo_meta = {}
    demo_groups = [
        (0, "Student", None, 14),
        (1, "Junior", "PGY 1", 5),
        (1, "Junior", "PGY 2", 3),
        (2, "Junior", "PGY 3", 2),
        (2, "Junior", "PGY 5", 3),
        (3, "Senior", "PGY 6", 4),
        (4, "Senior", "Fellow Spine", 7),
        (5, "Expert", None, 8),
    ]
    pid_counter = 0
    rng = np.random.default_rng(0)
    for gidx, gc, pgy, n in demo_groups:
        for _ in range(n):
            pid = f"P{pid_counter:03d}"
            pid_counter += 1
            demo_meta[pid] = {
                "group_coarse": gc if gidx != 5 else "Expert",
                "pgy_level": pgy,
                "dominant_hand": "Right",
            }
            base = -0.85 + (gidx / 5.0) * 1.7 + rng.normal(0, 0.08)
            for t in range(3):
                demo_preds.append(
                    {
                        "pid": pid,
                        "trial": str(t + 1),
                        "score": float(np.clip(base + rng.normal(0, 0.12), -1, 1)),
                        "group_coarse": demo_meta[pid]["group_coarse"],
                        "pgy_level": pgy,
                    }
                )

    plot_expertise_progression(demo_preds, demo_meta, out_path="results/expertise_progression_demo")
