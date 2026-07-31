"""
plot_score_vs_time.py
=====================
P3 — Graphe score d'expertise vs temps normalisé (la visualisation demandée par le prof).

Pour chaque trial : on fait glisser la fenêtre (N_CONTEXT) et on prédit un score par
position. L'axe X est normalisé [0, 1] (centre de fenêtre / durée du trial), l'axe Y
est le score d'expertise [-1, +1]. Chaque courbe est colorée par le niveau RÉEL.

Objectif visuel attendu : Staff (vert) globalement vers +1, Students (rouge) vers -1,
classes intermédiaires entre les deux, avec des variations visibles dans le temps.

Usage (depuis ICEMS-main, après train_long.py) :
    python plot_score_vs_time.py
    python plot_score_vs_time.py --model results/train_long --per-class 3
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from kfold_cv import extract_sliding_windows, N_CONTEXT, HOP  # noqa: E402
from continuous_scorer import apply_norm  # noqa: E402

LEVEL_NAMES = {0: "Student", 1: "PGY1", 2: "PGY2", 3: "PGY3", 4: "PGY4",
               5: "PGY5", 6: "PGY6", 7: "Fellow", 8: "Staff"}


def _rolling_median(scores, k):
    """Lissage par médiane glissante (réduit le bruit du score par fenêtre)."""
    if k <= 1 or len(scores) < k:
        return scores
    out = np.empty_like(scores)
    half = k // 2
    for i in range(len(scores)):
        lo, hi = max(0, i - half), min(len(scores), i + half + 1)
        out[i] = np.median(scores[lo:hi])
    return out


def trial_score_curve(model, X_raw, mean, std, smooth=1):
    """Retourne (x_norm[0..1], scores[-1..1]) pour un trial via fenêtre glissante."""
    X = apply_norm(X_raw, mean, std)
    windows, starts = extract_sliding_windows(X)
    if windows is None:
        return None, None
    scores = model.predict(windows, batch_size=64, verbose=0).flatten()
    scores = _rolling_median(scores, smooth)
    T = X.shape[0]
    centers = np.array([s + N_CONTEXT / 2.0 for s in starts])
    x_norm = centers / T
    return x_norm, scores


def main():
    ap = argparse.ArgumentParser(description="P3 — score vs temps normalisé.")
    ap.add_argument("--dataset", type=Path, default=Path("data/continuous_per_trial.pkl"))
    ap.add_argument("--model", type=Path, default=Path("results/train_long"),
                    help="Dossier contenant scorer.keras + norm_mean/std.npy")
    ap.add_argument("--out-dir", type=Path, default=Path("results/score_curves"))
    ap.add_argument("--per-class", type=int, default=3,
                    help="Nombre de trials à tracer par classe affichée.")
    ap.add_argument("--classes", type=int, nargs="+", default=[0, 4, 7, 8],
                    help="Classes à afficher (défaut : Student, PGY4, Fellow, Staff).")
    ap.add_argument("--smooth", type=int, default=9,
                    help="Fenêtre de médiane glissante sur le score (1 = aucun lissage).")
    args = ap.parse_args()

    from tensorflow import keras
    model = keras.models.load_model(args.model / "scorer.keras", compile=False)
    mean = np.load(args.model / "norm_mean.npy")
    std = np.load(args.model / "norm_std.npy")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset, "rb") as f:
        dataset = pickle.load(f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("RdYlGn")

    fig, ax = plt.subplots(figsize=(11, 6))
    for c in args.classes:
        keys = [k for k, v in dataset.items() if v["y9"] == c][: args.per_class]
        color = cmap(c / 8.0)
        for i, k in enumerate(keys):
            x, s = trial_score_curve(model, dataset[k]["X"], mean, std, smooth=args.smooth)
            if x is None:
                continue
            ax.plot(x, s, color=color, lw=1.3, alpha=0.8,
                    label=f"{LEVEL_NAMES[c]} (classe {c})" if i == 0 else None)

    ax.axhline(+1, color="green", ls=":", lw=0.6, alpha=0.5)
    ax.axhline(-1, color="red", ls=":", lw=0.6, alpha=0.5)
    ax.axhline(0, color="gray", ls="--", lw=0.6, alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Temps normalisé (0 = début du trial, 1 = fin)")
    ax.set_ylabel("Score d'expertise prédit")
    ax.set_title("Évolution du score d'expertise au cours du trial — par niveau")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = args.out_dir / "score_vs_time.png"
    pdf = args.out_dir / "score_vs_time.pdf"
    fig.savefig(png, dpi=130)
    fig.savefig(pdf)
    print(f"✅ Figures sauvegardées :\n   {png}\n   {pdf}")


if __name__ == "__main__":
    main()
