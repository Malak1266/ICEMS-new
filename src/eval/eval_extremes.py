"""
Évaluation de l'expérience train-extrêmes.
Génère toutes les figures + tableau de métriques.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import kendalltau, spearmanr

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from train.train_hybrid_extremes import SUBLEVEL_TO_CLASS4_EXTREMES, sublevel_to_class4

CLASS4_ORDER = ("student", "junior", "senior", "expert")
CLASS_LABELS_FR = ("Étudiant", "Junior", "Senior", "Expert")
CLASS_COLORS = ("#2166ac", "#67a9cf", "#e08214", "#b2182b")

MIDDLE_ORDER = ("pgy1", "pgy2", "pgy3", "pgy4", "pgy5", "pgy6", "fellow")


def _setup_style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
    })


def resolve_class4(pred: dict) -> str:
    c4 = pred.get("class_4")
    if isinstance(c4, str) and c4 in CLASS4_ORDER:
        return c4
    if isinstance(c4, int) and 0 <= c4 < 4:
        return CLASS4_ORDER[c4]
    return sublevel_to_class4(pred["sublevel"])


def class4_index(class_4: str) -> int:
    return CLASS4_ORDER.index(class_4)


def load_predictions(path: Path) -> List[dict]:
    with open(path, "rb") as f:
        return list(pickle.load(f))


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {}
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    out = {"n": len(y_true), "mae": mae, "mse": mse, "rmse": rmse}
    if len(y_true) > 2:
        rho, p_rho = spearmanr(y_true, y_pred)
        tau, p_tau = kendalltau(y_true, y_pred)
        out.update({
            "spearman_rho": float(rho),
            "p_spearman": float(p_rho),
            "kendall_tau": float(tau),
            "p_kendall": float(p_tau),
        })
    return out


def compute_global_metrics(preds: Sequence[dict]) -> dict:
    y_true = np.array([p["score_true"] for p in preds], dtype=np.float64)
    y_pred = np.array([p["score_pred"] for p in preds], dtype=np.float64)
    return _regression_metrics(y_true, y_pred)


def metrics_by_sublevel(preds: Sequence[dict]) -> Dict[str, dict]:
    by_sl: Dict[str, List[dict]] = defaultdict(list)
    for p in preds:
        by_sl[p["sublevel"]].append(p)
    return {
        sl: compute_global_metrics(rows)
        for sl, rows in sorted(by_sl.items())
    }


def metrics_by_class4(preds: Sequence[dict]) -> Dict[str, dict]:
    by_c4: Dict[str, List[dict]] = defaultdict(list)
    for p in preds:
        by_c4[resolve_class4(p)].append(p)
    return {
        c4: compute_global_metrics(rows)
        for c4, rows in sorted(by_c4.items(), key=lambda x: class4_index(x[0]))
    }


def ordinal_class_means(preds: Sequence[dict]) -> Tuple[List[float], List[int], bool]:
    """Moyennes par classe ordinale (student / junior / senior / expert)."""
    by_class: Dict[str, List[float]] = {c: [] for c in CLASS4_ORDER}
    for p in preds:
        by_class[resolve_class4(p)].append(float(p["score_pred"]))
    means = [
        float(np.mean(by_class[c])) if by_class[c] else float("nan")
        for c in CLASS4_ORDER
    ]
    counts = [len(by_class[c]) for c in CLASS4_ORDER]
    present = [i for i, n in enumerate(counts) if n > 0]
    mono_ok = all(
        means[present[i]] <= means[present[i + 1]]
        for i in range(len(present) - 1)
        if not (np.isnan(means[present[i]]) or np.isnan(means[present[i + 1]]))
    ) if len(present) >= 2 else True
    return means, counts, mono_ok


def generalization_gap(
    preds_extremes: Sequence[dict],
    preds_middle: Sequence[dict],
) -> dict:
    m_ext = compute_global_metrics(preds_extremes)
    m_mid = compute_global_metrics(preds_middle)
    gap = {}
    for key in ("mae", "mse", "rmse"):
        if key in m_ext and key in m_mid:
            gap[f"delta_{key}"] = m_mid[key] - m_ext[key]
    return {"extremes": m_ext, "middle": m_mid, "gap": gap}


def figure_monotonie(preds: Sequence[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    means, counts, mono_ok = ordinal_class_means(preds)
    by_class: Dict[str, List[float]] = {c: [] for c in CLASS4_ORDER}
    for p in preds:
        by_class[resolve_class4(p)].append(float(p["score_pred"]))

    present_idx = [i for i, n in enumerate(counts) if n > 0]
    valid_means = [means[i] for i in present_idx]
    rho, _ = (
        spearmanr(range(len(valid_means)), valid_means)
        if len(valid_means) >= 2 else (float("nan"), float("nan"))
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(4)
    ax.plot(x, means, "o-", color="#333333", markersize=8, linewidth=1.8, label="Moyenne prédite")
    for i, c4 in enumerate(CLASS4_ORDER):
        if by_class[c4]:
            jitter = np.random.default_rng(42 + i).uniform(-0.08, 0.08, size=len(by_class[c4]))
            ax.scatter(
                np.full(len(by_class[c4]), i) + jitter, by_class[c4],
                c=CLASS_COLORS[i], s=35, alpha=0.7, edgecolors="#333", linewidths=0.4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS_FR)
    ax.set_ylabel("Score d'expertise prédit")
    ax.set_xlabel("Niveau d'expertise (ordinal)")
    ax.set_ylim(-1.08, 1.08)
    ax.axhline(0, color="#888888", linestyle="--", linewidth=0.8)
    status = "respectée" if mono_ok else "non respectée"
    ax.set_title(
        f"Monotonie ordinale des prédictions (milieu)\n"
        f"Ordre attendu : Étudiant < Junior < Senior < Expert — {status} | "
        f"Spearman rho = {rho:+.3f}",
    )
    ax.legend(loc="lower right")

    path = out_dir / "figure_extremes_monotonie.pdf"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_distribution(preds: Sequence[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    by_pid: Dict[str, List[dict]] = defaultdict(list)
    for p in preds:
        by_pid[p["participant"]].append(p)

    rows = []
    for pid, trials in sorted(by_pid.items()):
        scores = [t["score_pred"] for t in trials]
        c4 = resolve_class4(trials[0])
        rows.append({
            "participant": pid,
            "sublevel": trials[0]["sublevel"],
            "class_4": c4,
            "class_idx": class4_index(c4),
            "median": float(np.median(scores)),
            "scores": scores,
        })
    rows.sort(key=lambda r: (r["class_idx"], r["sublevel"], r["participant"]))

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.35), 5))
    x = np.arange(len(rows))
    colors = [CLASS_COLORS[r["class_idx"]] for r in rows]
    ax.scatter(x, [r["median"] for r in rows], c=colors, s=55, edgecolors="#333", linewidths=0.4)

    for i, r in enumerate(rows):
        if len(r["scores"]) > 1:
            jitter = np.linspace(-0.12, 0.12, len(r["scores"]))
            ax.scatter(
                np.full(len(r["scores"]), i) + jitter, r["scores"],
                c=colors[i], s=22, alpha=0.45, edgecolors="none",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['participant'][-4:]}\n({r['sublevel']})" for r in rows],
        rotation=90, fontsize=7,
    )
    ax.set_ylabel("Score d'expertise prédit")
    ax.set_title("Distribution des scores prédits par participant (niveaux intermédiaires)")
    ax.set_ylim(-1.08, 1.08)
    ax.legend(
        handles=[Patch(facecolor=CLASS_COLORS[i], label=CLASS_LABELS_FR[i]) for i in range(4)],
        loc="upper left", fontsize=8,
    )

    path = out_dir / "figure_extremes_distribution.pdf"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_dispersion_sousniveaux(preds: Sequence[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    by_sl: Dict[str, List[float]] = defaultdict(list)
    for p in preds:
        by_sl[p["sublevel"]].append(float(p["score_pred"]))

    order = [s for s in MIDDLE_ORDER if s in by_sl]
    means, stds, ns = [], [], []
    for sl in order:
        arr = np.asarray(by_sl[sl], dtype=float)
        means.append(float(arr.mean()))
        stds.append(float(arr.std()))
        ns.append(len(arr))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(order))
    ax.bar(x, means, yerr=stds, capsize=4, color="#67a9cf", edgecolor="#333", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in order])
    ax.set_ylabel("Score prédit moyen ± écart-type")
    ax.set_xlabel("Sous-niveau de formation")
    ax.set_title("Score moyen prédit par sous-niveau (PGY1–Fellow)")
    ax.set_ylim(-1.08, 1.08)

    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(i, m + 0.05, f"n={n}", ha="center", fontsize=8)

    path = out_dir / "figure_extremes_dispersion_sousniveaux.pdf"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_erreurs(preds: Sequence[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    errors_by_sl: Dict[str, List[float]] = defaultdict(list)
    for p in preds:
        errors_by_sl[p["sublevel"]].append(float(p["score_pred"] - p["score_true"]))

    order = [s for s in MIDDLE_ORDER if s in errors_by_sl]
    data, labels = [], []
    for sl in order:
        data.append(errors_by_sl[sl])
        labels.append(sl.upper())

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#8da0cb")
        patch.set_alpha(0.7)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Erreur de prédiction (prédit − vrai)")
    ax.set_xlabel("Sous-niveau")
    ax.set_title("Distribution des erreurs par sous-niveau")

    path = out_dir / "figure_extremes_erreurs.pdf"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_metrics_table(
    global_metrics: dict,
    by_sublevel: Dict[str, dict],
    by_class4: Dict[str, dict],
    class_means: List[float],
    class_counts: List[int],
    mono_ok: bool,
    gap_info: dict,
    out_dir: Path,
) -> Path:
    path = out_dir / "table_extremes_metriques.csv"
    rows: List[dict] = []

    for key in ("mae", "mse", "rmse", "spearman_rho", "kendall_tau", "n"):
        rows.append({"section": "global_middle", "metric": key, "value": global_metrics.get(key, "")})

    for i, (c4, label) in enumerate(zip(CLASS4_ORDER, CLASS_LABELS_FR)):
        rows.append({
            "section": "ordinal_monotonicity",
            "metric": f"mean_pred_{label}",
            "value": class_means[i] if i < len(class_means) else "",
        })
        rows.append({
            "section": "ordinal_monotonicity",
            "metric": f"n_{c4}",
            "value": class_counts[i] if i < len(class_counts) else "",
        })
    rows.append({
        "section": "ordinal_monotonicity",
        "metric": "monotone_ok",
        "value": int(mono_ok),
    })

    for sl, m in by_sublevel.items():
        for key, val in m.items():
            rows.append({"section": f"sublevel_{sl}", "metric": key, "value": val})

    for c4, m in by_class4.items():
        for key, val in m.items():
            rows.append({"section": f"class4_{c4}", "metric": key, "value": val})

    for split_name in ("extremes", "middle"):
        m = gap_info.get(split_name, {})
        for key, val in m.items():
            rows.append({"section": f"generalization_{split_name}", "metric": key, "value": val})
    for key, val in gap_info.get("gap", {}).items():
        rows.append({"section": "generalization_gap", "metric": key, "value": val})

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_evaluation(
    results_dir: Path,
    out_dir: Optional[Path] = None,
) -> dict:
    _setup_style()
    results_dir = Path(results_dir)
    out_dir = out_dir or results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_middle = load_predictions(results_dir / "predictions_middle.pkl")
    extremes_path = results_dir / "predictions_extremes.pkl"
    preds_extremes = load_predictions(extremes_path) if extremes_path.exists() else []

    global_metrics = compute_global_metrics(preds_middle)
    by_sublevel = metrics_by_sublevel(preds_middle)
    by_class4 = metrics_by_class4(preds_middle)
    class_means, class_counts, mono_ok = ordinal_class_means(preds_middle)
    gap_info = generalization_gap(preds_extremes, preds_middle) if preds_extremes else {}

    figure_monotonie(preds_middle, out_dir)
    figure_distribution(preds_middle, out_dir)
    figure_dispersion_sousniveaux(preds_middle, out_dir)
    figure_erreurs(preds_middle, out_dir)
    write_metrics_table(
        global_metrics, by_sublevel, by_class4, class_means, class_counts,
        mono_ok, gap_info, out_dir,
    )

    full = {
        "global_middle": global_metrics,
        "by_sublevel": by_sublevel,
        "by_class4": by_class4,
        "class4_mapping": SUBLEVEL_TO_CLASS4_EXTREMES,
        "ordinal_monotonicity": {
            "class_means": dict(zip(CLASS_LABELS_FR, class_means)),
            "class_counts": dict(zip(CLASS4_ORDER, class_counts)),
            "monotone_ok": mono_ok,
        },
        "generalization": gap_info,
    }
    with open(out_dir / "metrics_extremes.json", "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2)

    print(
        f"[eval] milieu: MAE={global_metrics.get('mae', float('nan')):.4f} "
        f"RMSE={global_metrics.get('rmse', float('nan')):.4f} "
        f"spearman={global_metrics.get('spearman_rho', float('nan')):.4f} "
        f"kendall={global_metrics.get('kendall_tau', float('nan')):.4f}"
    )
    if gap_info:
        print(
            f"[eval] gap MAE (milieu-extremes)="
            f"{gap_info.get('gap', {}).get('delta_mae', float('nan')):+.4f}"
        )
    print(f"[eval] figures -> {out_dir}")
    return full


def main() -> None:
    ap = argparse.ArgumentParser(description="Évaluation train-extrêmes → test milieu")
    ap.add_argument(
        "--results", type=Path, default=Path("results/hybrid_extremes"),
        help="Répertoire contenant predictions_middle.pkl",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Répertoire de sortie figures (défaut: --results)",
    )
    args = ap.parse_args()
    run_evaluation(args.results, out_dir=args.out)


if __name__ == "__main__":
    main()
