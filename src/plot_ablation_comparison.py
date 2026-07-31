"""
plot_ablation_comparison.py
===========================
Figure comparative A vs B pour slide / rapport (ablation MAE).

Usage :
    python src/plot_ablation_comparison.py --ablation-dir results/ablation_aug
    python src/plot_ablation_comparison.py --ablation-dir "<chemin-local>/results/ablation_aug"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_B_classification import (  # noqa: E402
    CLASS4_COLORS,
    CLASS4_NAMES,
    load_lopo_results,
    plot_confusion,
    plot_scatter,
    plot_score_vs_time,
)


def _global_metrics(preds: List[dict]) -> Dict[str, float]:
    y_true = np.array([p["y_reg"] for p in preds], dtype=float)
    y_pred = np.array([p["score"] for p in preds], dtype=float)
    y4 = np.array([p["y4"] for p in preds])
    pred_cls = np.array([p["pred_class"] for p in preds])
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    return {
        "pearson_r": float(pr),
        "spearman_r": float(sr),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "accuracy": float((pred_cls == y4).mean()),
        "n_trials": len(preds),
    }


def _load_condition(ablation_dir: Path, tag: str) -> Tuple[List[dict], dict, Dict[str, float]]:
    pred_path = ablation_dir / tag / "lopo_predictions.pkl"
    if not pred_path.exists():
        raise FileNotFoundError(f"{pred_path} introuvable.")
    preds, aux = load_lopo_results(pred_path)
    return preds, aux, _global_metrics(preds)


def _side_by_side_images(
    left: Path,
    right: Path,
    out_path: Path,
    titles: Tuple[str, str],
    suptitle: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, img_path, title in zip(axes, (left, right), titles):
        ax.imshow(mpimg.imread(str(img_path)))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def _slide_figure(
    preds_a: List[dict],
    preds_b: List[dict],
    aux_a: dict,
    aux_b: dict,
    met_a: Dict[str, float],
    met_b: Dict[str, float],
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.32, wspace=0.28)

    # Barres métriques
    ax_bar = fig.add_subplot(gs[0, :])
    metrics = ["Pearson r", "Spearman ρ", "MAE", "Accuracy"]
    vals_a = [met_a["pearson_r"], met_a["spearman_r"], met_a["mae"], met_a["accuracy"] * 100]
    vals_b = [met_b["pearson_r"], met_b["spearman_r"], met_b["mae"], met_b["accuracy"] * 100]
    x = np.arange(len(metrics))
    w = 0.35
    bars_a = ax_bar.bar(x - w / 2, vals_a, w, label="A — MAE pré-entraîné", color="#2E86AB")
    bars_b = ax_bar.bar(x + w / 2, vals_b, w, label="B — Init. aléatoire", color="#A23B72")
    ax_bar.set_xticks(x, metrics)
    ax_bar.set_ylabel("Valeur")
    ax_bar.set_title(
        f"Ablation MAE augmenté — Δr = {met_a['pearson_r'] - met_b['pearson_r']:+.3f}  "
        f"(A={met_a['pearson_r']:.3f}, B={met_b['pearson_r']:.3f})",
        fontweight="bold",
    )
    ax_bar.legend(loc="upper right")
    ax_bar.grid(axis="y", alpha=0.3)
    for bars in (bars_a, bars_b):
        for bar in bars:
            h = bar.get_height()
            fmt = f"{h:.3f}" if bar.get_x() < 2.5 else f"{h:.1f}%"
            ax_bar.text(bar.get_x() + bar.get_width() / 2, h, fmt, ha="center", va="bottom", fontsize=8)

    # Scatter A / B
    for col, (tag, preds, met) in enumerate(
        (("A", preds_a, met_a), ("B", preds_b, met_b))
    ):
        ax = fig.add_subplot(gs[1, col])
        y_true = np.array([p["y_reg"] for p in preds])
        y_pred = np.array([p["score"] for p in preds])
        y4 = np.array([p["y4"] for p in preds])
        for c in range(4):
            m = y4 == c
            ax.scatter(y_true[m], y_pred[m], c=CLASS4_COLORS[c], alpha=0.65, s=22)
        lims = [-1.1, 1.1]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.4)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel("Score réel")
        ax.set_ylabel("Score prédit")
        ax.set_title(f"Condition {tag} — r = {met['pearson_r']:+.3f}", fontweight="bold")
        ax.grid(alpha=0.25)

    # Courbe score vs temps (médiane A)
    ax_time = fig.add_subplot(gs[1, 2])
    from scipy.ndimage import gaussian_filter1d

    t_common = np.linspace(0.0, 1.0, 500)
    sigma = 30
    for tag, aux, color, ls in (
        ("A", aux_a, "#2E86AB", "-"),
        ("B", aux_b, "#A23B72", "--"),
    ):
        curves = aux["curves"]
        stack = []
        for c in curves:
            stack.append(np.interp(t_common, c["t_norm"], c["mean"]))
        if not stack:
            continue
        med = gaussian_filter1d(np.median(np.stack(stack), axis=0), sigma=sigma)
        ax_time.plot(t_common, med, color=color, lw=2.5, ls=ls, label=f"Condition {tag}")
    for y_val, lbl in [(-1, "Student"), (-0.33, "Junior"), (0.33, "Senior"), (1, "Expert")]:
        ax_time.axhline(y_val, color="gray", ls=":", lw=0.7, alpha=0.5)
    ax_time.set_xlim(0, 1)
    ax_time.set_ylim(-1.05, 1.05)
    ax_time.set_xlabel("Temps normalisé")
    ax_time.set_ylabel("Score")
    ax_time.set_title("Progression temporelle (médiane)", fontweight="bold")
    ax_time.legend()
    ax_time.grid(alpha=0.25)

    fig.suptitle(
        "Transfert MAE Atracsys → ICEMS · LOPO 47 participants · encodeur augmenté (mae_aug_run1)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Figures comparatives ablation A vs B.")
    ap.add_argument(
        "--ablation-dir",
        type=Path,
        required=True,
        help="Dossier ablation (contient A/ et B/ avec lopo_predictions.pkl).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Dossier de sortie (défaut : <ablation-dir>/comparison).",
    )
    args = ap.parse_args()

    ablation_dir = args.ablation_dir.resolve()
    out_dir = (args.out or ablation_dir / "comparison").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PLOT ABLATION COMPARISON — A vs B")
    print("=" * 60)
    print(f"  Source : {ablation_dir}")
    print(f"  Sortie : {out_dir}")

    preds_a, aux_a, met_a = _load_condition(ablation_dir, "A")
    preds_b, aux_b, met_b = _load_condition(ablation_dir, "B")

    tmp_a = out_dir / "_tmp_A"
    tmp_b = out_dir / "_tmp_B"
    tmp_a.mkdir(exist_ok=True)
    tmp_b.mkdir(exist_ok=True)

    plot_scatter(preds_a, tmp_a / "scatter.png")
    plot_scatter(preds_b, tmp_b / "scatter.png")
    plot_score_vs_time(aux_a["curves"], tmp_a / "score_vs_time.png", n_trials=len(preds_a))
    plot_score_vs_time(aux_b["curves"], tmp_b / "score_vs_time.png", n_trials=len(preds_b))
    plot_confusion(preds_a, tmp_a / "confusion_matrix.png")
    plot_confusion(preds_b, tmp_b / "confusion_matrix.png")

    _side_by_side_images(
        tmp_a / "scatter.png",
        tmp_b / "scatter.png",
        out_dir / "comparison_scatter.png",
        ("A — MAE pré-entraîné", "B — Init. aléatoire"),
        "Score prédit vs réel",
    )
    _side_by_side_images(
        tmp_a / "score_vs_time.png",
        tmp_b / "score_vs_time.png",
        out_dir / "comparison_score_vs_time.png",
        ("A — MAE pré-entraîné", "B — Init. aléatoire"),
        "Progression du score au cours du geste",
    )
    _side_by_side_images(
        tmp_a / "confusion_matrix.png",
        tmp_b / "confusion_matrix.png",
        out_dir / "comparison_confusion.png",
        ("A — MAE pré-entraîné", "B — Init. aléatoire"),
        "Matrice de confusion 4×4",
    )
    _slide_figure(preds_a, preds_b, aux_a, aux_b, met_a, met_b, out_dir / "ablation_slide.png")

    summary = {
        "A": met_a,
        "B": met_b,
        "delta_pearson": met_a["pearson_r"] - met_b["pearson_r"],
        "delta_spearman": met_a["spearman_r"] - met_b["spearman_r"],
        "delta_mae": met_a["mae"] - met_b["mae"],
        "delta_accuracy": met_a["accuracy"] - met_b["accuracy"],
    }
    summary_path = out_dir / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  → {summary_path}")

    print("\n  Métriques globales :")
    print(f"    A  Pearson={met_a['pearson_r']:.3f}  Spearman={met_a['spearman_r']:.3f}  "
          f"MAE={met_a['mae']:.3f}  Acc={met_a['accuracy']*100:.1f}%")
    print(f"    B  Pearson={met_b['pearson_r']:.3f}  Spearman={met_b['spearman_r']:.3f}  "
          f"MAE={met_b['mae']:.3f}  Acc={met_b['accuracy']*100:.1f}%")
    print(f"    Δr = {summary['delta_pearson']:+.3f}")
    print("\n  Figures prêtes pour le slide :")
    print(f"    {out_dir / 'ablation_slide.png'}  ← figure tout-en-un")
    print(f"    {out_dir / 'comparison_scatter.png'}")


if __name__ == "__main__":
    main()
