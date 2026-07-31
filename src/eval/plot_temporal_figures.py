"""
plot_temporal_figures.py
========================
Figures temporelles à partir de frame_predictions.pkl (sortie de extract_frame_scores).

Produit :
  - figure_temporal_progression.pdf  — courbes moyennes + IC 95 % par classe (4 groupes)
  - figure_phase_distribution.pdf    — violons Early / Middle / Late par classe

Usage :
    python -m eval.plot_temporal_figures \\
        --input results/hybrid_hoel_b/frame_predictions.pkl \\
        --output results/hybrid_hoel_b/
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import SUBLEVEL_ORDER  # noqa: E402

SUBLEVEL_TO_CLASS4: Dict[str, str] = {
    "ms": "student",
    "pgy1": "junior", "pgy2": "junior", "pgy3": "junior",
    "pgy4": "junior", "pgy5": "junior",
    "pgy6": "senior", "fellow": "senior",
    "staff": "expert",
}

CLASS4_ORDER = ("student", "junior", "senior", "expert")
CLASS4_LABELS = {
    "student": "Student",
    "junior": "Junior",
    "senior": "Senior",
    "expert": "Expert",
}
CLASS4_COLORS = {
    "student": "#2166ac",
    "junior": "#67a9cf",
    "senior": "#e08214",
    "expert": "#b2182b",
}

N_GRID = 100
N_BOOT = 2000
BOOT_SEED = 42
PHASE_BOUNDS = ((0.0, 0.33), (0.33, 0.66), (0.66, 1.01))
PHASE_LABELS = ("Early", "Middle", "Late")


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


def _class4(sublevel: str) -> str:
    key = sublevel.strip().lower()
    if key in SUBLEVEL_TO_CLASS4:
        return SUBLEVEL_TO_CLASS4[key]
    if key in SUBLEVEL_ORDER:
        y9 = SUBLEVEL_ORDER.index(key)
        return SUBLEVEL_TO_CLASS4[SUBLEVEL_ORDER[y9]]
    return "junior"


def load_entries(path: Path) -> List[dict]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "entries" in payload:
        return list(payload["entries"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Format non reconnu : {path}")


def _resample_curve(time_norm: np.ndarray, frame_scores: np.ndarray, grid: np.ndarray) -> np.ndarray:
    t = np.asarray(time_norm, dtype=float).ravel()
    s = np.asarray(frame_scores, dtype=float).ravel()
    if t.size < 2:
        return np.full(grid.size, float(s[0]) if s.size else np.nan)
    order = np.argsort(t)
    t, s = t[order], s[order]
    return np.interp(grid, t, s)


def _bootstrap_ci_band(curves: Sequence[np.ndarray], grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retourne (mean, ci_lo, ci_hi) de taille len(grid)."""
    arr = np.stack(curves, axis=0)
    mean = arr.mean(axis=0)
    if arr.shape[0] < 2:
        return mean, mean.copy(), mean.copy()

    rng = np.random.default_rng(BOOT_SEED)
    boots = np.empty((N_BOOT, grid.size), dtype=float)
    n = arr.shape[0]
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boots[i] = arr[idx].mean(axis=0)
    ci_lo = np.percentile(boots, 2.5, axis=0)
    ci_hi = np.percentile(boots, 97.5, axis=0)
    return mean, ci_lo, ci_hi


def plot_temporal_progression(entries: Sequence[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    grid = np.linspace(0.0, 1.0, N_GRID)
    by_class: Dict[str, List[np.ndarray]] = {c: [] for c in CLASS4_ORDER}

    for e in entries:
        cls = e.get("class_4") or _class4(str(e.get("sublevel", "")))
        if cls not in by_class:
            continue
        curve = _resample_curve(e["time_norm"], e["frame_scores"], grid)
        by_class[cls].append(curve)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cls in CLASS4_ORDER:
        curves = by_class[cls]
        if not curves:
            continue
        mean, lo, hi = _bootstrap_ci_band(curves, grid)
        color = CLASS4_COLORS[cls]
        ax.plot(grid, mean, color=color, lw=2.2, label=f"{CLASS4_LABELS[cls]} (n={len(curves)})")
        ax.fill_between(grid, lo, hi, color=color, alpha=0.20)

    ax.axhline(0.0, color="#888", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Normalized trial time")
    ax.set_ylabel("Predicted expertise score")
    ax.set_title("Temporal progression of per-frame expertise scores\nHybrid LSTM-Transformer · LOPO")
    ax.legend(loc="best")

    out_path = out_dir / "figure_temporal_progression.pdf"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_phase_distribution(entries: Sequence[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    phase_data: Dict[int, Dict[str, List[float]]] = {
        ph: {c: [] for c in CLASS4_ORDER} for ph in range(3)
    }

    for e in entries:
        cls = e.get("class_4") or _class4(str(e.get("sublevel", "")))
        if cls not in CLASS4_ORDER:
            continue
        t = np.asarray(e["time_norm"], dtype=float).ravel()
        s = np.asarray(e["frame_scores"], dtype=float).ravel()
        for ph, (lo, hi) in enumerate(PHASE_BOUNDS):
            mask = (t >= lo) & (t < hi)
            if mask.any():
                phase_data[ph][cls].append(float(np.mean(s[mask])))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)

    for ph, ax in enumerate(axes):
        data, positions, colors, labels = [], [], [], []
        pos = 1
        for cls in CLASS4_ORDER:
            vals = phase_data[ph][cls]
            if not vals:
                pos += 1
                continue
            data.append(vals)
            positions.append(pos)
            colors.append(CLASS4_COLORS[cls])
            labels.append(CLASS4_LABELS[cls])
            pos += 1

        if data:
            parts = ax.violinplot(data, positions=positions, widths=0.7,
                                  showmeans=True, showmedians=False, showextrema=True)
            for body, col in zip(parts["bodies"], colors):
                body.set_facecolor(col)
                body.set_alpha(0.65)
                body.set_edgecolor(col)
            parts["cmeans"].set_color("#222")
            parts["cmeans"].set_linewidth(1.5)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(PHASE_LABELS[ph])
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0.0, color="#888", linestyle=":", linewidth=0.7)
        if ph == 0:
            ax.set_ylabel("Mean per-frame score in phase")

    fig.suptitle(
        "Per-frame score distribution by trial phase\nHybrid LSTM-Transformer · LOPO",
        y=1.02,
    )
    fig.tight_layout()

    out_path = out_dir / "figure_phase_distribution.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Figures temporelles depuis frame_predictions.pkl")
    ap.add_argument("--input", type=Path, required=True, help="frame_predictions.pkl")
    ap.add_argument("--output", type=Path, required=True, help="Dossier de sortie des figures")
    args = ap.parse_args()

    _setup_style()
    entries = load_entries(args.input)
    if not entries:
        raise SystemExit(f"Aucune entrée dans {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)

    p1 = plot_temporal_progression(entries, args.output)
    p2 = plot_phase_distribution(entries, args.output)
    print(f"[done] {len(entries)} trials")
    print(f"       -> {p1}")
    print(f"       -> {p2}")


if __name__ == "__main__":
    main()
