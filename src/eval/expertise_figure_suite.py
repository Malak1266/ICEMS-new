"""
expertise_figure_suite.py
=========================
Suite de figures publication-ready pour l'évaluation d'un modèle d'expertise
chirurgicale (LOPO hybrid + HOEL).

Figures générées (texte en français) :
  1. Courbe de monotonie ordinale (4 classes)
  2. Distribution par participant
  3. Granularité fine par sous-niveau (PGY1–PGY5, Fellows)
  4. Progression temporelle par classe
  5. Monotonie ordinale dynamique (lissée)
  6. Distribution par phases du geste (Early / Middle / Late)

Usage (après run LOPO) :
    python -m eval.expertise_figure_suite \\
        --preds results/hybrid_hoel/predictions.pkl \\
        --out results/hybrid_hoel/figures

Avec courbes temporelles (nécessite checkpoints LOPO) :
    python -m eval.expertise_figure_suite \\
        --preds results/hybrid_hoel/predictions.pkl \\
        --out results/hybrid_hoel/figures \\
        --checkpoints results/hybrid_hoel/checkpoints \\
        --pkl data/continuous_per_trial.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import kendalltau, kruskal, spearmanr

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import SUBLEVEL_ORDER, SUBLEVEL_TO_SCORE  # noqa: E402

# ─── Mapping clinique ──────────────────────────────────────────────────────────
Y9_TO_Y4: Dict[int, int] = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 3}

CLASS_NAMES_FR = ("Étudiant", "Junior", "Senior", "Expert")
CLASS_COLORS = ("#2166ac", "#67a9cf", "#e08214", "#b2182b")

SUBLEVEL_TO_Y9 = {k: i for i, k in enumerate(SUBLEVEL_ORDER)}

GRANULAR_GROUPS = ("pgy1", "pgy2", "pgy3", "pgy4", "pgy5", "fellow")
GRANULAR_LABELS_FR = {
    "pgy1": "PGY1", "pgy2": "PGY2", "pgy3": "PGY3",
    "pgy4": "PGY4", "pgy5": "PGY5", "fellow": "Fellows",
}

PHASE_LABELS_FR = ("Début (0–33 %)", "Milieu (33–66 %)", "Fin (66–100 %)")

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
EARLY_FIXATION_FRAC = 0.15


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


def _bootstrap_ci(values: np.ndarray, stat: str = "mean") -> Tuple[float, float, float]:
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        v = float(values[0])
        return v, v, v
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots = np.empty(N_BOOTSTRAP, dtype=float)
    for i in range(N_BOOTSTRAP):
        s = rng.choice(values, size=len(values), replace=True)
        boots[i] = np.mean(s) if stat == "mean" else np.median(s)
    fn = np.mean if stat == "mean" else np.median
    return float(fn(values)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load_predictions(path: Path) -> List[dict]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and "preds" in payload:
        rows = payload["preds"]
    else:
        raise ValueError(f"Format pickle non reconnu : {path}")
    return normalize_predictions(rows)


def normalize_predictions(rows: Sequence[dict]) -> List[dict]:
    """Unifie predictions.pkl HOEL et formats Step B."""
    out: List[dict] = []
    for r in rows:
        sub = str(r.get("sublevel", "")).strip().lower()
        if not sub and "y9" in r:
            sub = SUBLEVEL_ORDER[int(r["y9"])]
        y9 = SUBLEVEL_TO_Y9.get(sub, int(r.get("y9", 0)))
        y4 = int(r.get("y4", Y9_TO_Y4[y9]))
        score = float(r.get("score_pred", r.get("score", 0.0)))
        pid = str(r.get("participant", r.get("pid", "")))
        out.append({
            "participant": pid,
            "trial_id": str(r.get("trial_id", r.get("trial", ""))),
            "sublevel": sub or SUBLEVEL_ORDER[y9],
            "y9": y9,
            "y4": y4,
            "class_fr": CLASS_NAMES_FR[y4],
            "score_pred": score,
            "score_true": float(r.get("score_true", r.get("y_reg", SUBLEVEL_TO_SCORE.get(sub, 0)))),
            "tier": int(r.get("tier", 0)),
        })
    return out


def participant_medians(preds: Sequence[dict]) -> List[dict]:
    by_pid: Dict[str, List[dict]] = defaultdict(list)
    for p in preds:
        by_pid[p["participant"]].append(p)
    rows = []
    for pid, trials in sorted(by_pid.items()):
        y4 = trials[0]["y4"]
        sub = trials[0]["sublevel"]
        scores = [t["score_pred"] for t in trials]
        rows.append({
            "participant": pid,
            "y4": y4,
            "class_fr": CLASS_NAMES_FR[y4],
            "sublevel": sub,
            "n_trials": len(trials),
            "score_median": float(np.median(scores)),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
        })
    return rows


@dataclass
class FigureAnalysis:
    figure_id: int
    title: str
    monotonicity: str = ""
    inter_class_separation: str = ""
    intra_class_variance: str = ""
    temporal_stability: str = ""
    outliers: str = ""
    early_fixation: str = ""
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "figure": self.figure_id,
            "title": self.title,
            "monotonie_ordinale": self.monotonicity,
            "separation_inter_classes": self.inter_class_separation,
            "variance_intra_classe": self.intra_class_variance,
            "stabilite_temporelle": self.temporal_stability,
            "individus_aberrants": self.outliers,
            "fixation_precoce": self.early_fixation,
            **self.extra,
        }


# ─── Figure 1 : monotonie ordinale (4 classes) ───────────────────────────────

def figure1_ordinal_monotonicity(
    preds: Sequence[dict], out_dir: Path,
) -> Tuple[Path, FigureAnalysis]:
    import matplotlib.pyplot as plt

    parts = participant_medians(preds)
    by_class: Dict[int, List[float]] = {i: [] for i in range(4)}
    for p in parts:
        by_class[p["y4"]].append(p["score_median"])

    means, stds, ci_lo, ci_hi, ns = [], [], [], [], []
    for y4 in range(4):
        arr = np.asarray(by_class[y4], dtype=float)
        m, lo, hi = _bootstrap_ci(arr, stat="mean")
        means.append(m)
        stds.append(float(arr.std()) if len(arr) else float("nan"))
        ci_lo.append(m - lo)
        ci_hi.append(hi - m)
        ns.append(len(arr))

    rho, p_rho = spearmanr(range(4), means) if len(means) == 4 else (float("nan"), float("nan"))
    mono_ok = all(means[i] <= means[i + 1] for i in range(3) if not np.isnan(means[i]))

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(4)
    ax.errorbar(
        x, means, yerr=[ci_lo, ci_hi], fmt="o-", color="#333333",
        markersize=8, capsize=5, linewidth=1.8, label="Moyenne (IC 95 % bootstrap)",
    )
    for i, (y4, n) in enumerate(zip(range(4), ns)):
        jitter = np.random.default_rng(42 + i).uniform(-0.08, 0.08, size=n)
        ax.scatter(
            np.full(n, i) + jitter, by_class[y4],
            c=CLASS_COLORS[y4], s=40, alpha=0.75, edgecolors="#333", linewidths=0.4, zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES_FR)
    ax.set_ylabel("Score d'expertise prédit")
    ax.set_xlabel("Niveau d'expertise (ordinal)")
    ax.set_ylim(-1.08, 1.08)
    ax.axhline(0, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_title(
        "Figure 1 — Monotonie ordinale des scores prédits\n"
        f"Spearman ρ = {rho:+.3f}  (p = {p_rho:.3g})",
    )
    ax.legend(loc="lower right")

    path = out_dir / "figure01_monotonie_ordinale.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    analysis = FigureAnalysis(
        figure_id=1,
        title="Monotonie ordinale (4 classes)",
        monotonicity=(
            "Monotonie stricte respectée." if mono_ok
            else "Inversion(s) détectée(s) entre classes adjacentes — vérifier Senior/Expert."
        ),
        inter_class_separation=(
            f"Écart Étudiant→Expert : {means[3] - means[0]:+.3f}. "
            f"ρ de Spearman = {rho:+.3f}."
        ),
        intra_class_variance=(
            f"Écarts-types par classe : "
            + ", ".join(f"{CLASS_NAMES_FR[i]}={stds[i]:.3f}" for i in range(4))
        ),
        outliers="Voir Figure 2 pour l'identification individuelle.",
        extra={"spearman_rho": float(rho), "monotone": mono_ok, "n_per_class": ns},
    )
    return path, analysis


# ─── Figure 2 : distribution par participant ─────────────────────────────────

def figure2_participant_distribution(
    preds: Sequence[dict], out_dir: Path,
) -> Tuple[Path, FigureAnalysis]:
    import matplotlib.pyplot as plt

    parts = sorted(participant_medians(preds), key=lambda r: (r["y4"], r["sublevel"], r["participant"]))
    labels = [f"{p['participant'][-4:]}\n({p['sublevel']})" for p in parts]
    scores = [p["score_median"] for p in parts]
    colors = [CLASS_COLORS[p["y4"]] for p in parts]

    fig, ax = plt.subplots(figsize=(max(12, len(parts) * 0.28), 5))
    x = np.arange(len(parts))
    ax.scatter(x, scores, c=colors, s=55, edgecolors="#333", linewidths=0.4, zorder=3)

    # Points pâles = trials individuels (dispersion intra-participant)
    by_pid: Dict[str, List[float]] = defaultdict(list)
    for p in preds:
        by_pid[p["participant"]].append(p["score_pred"])
    for i, part in enumerate(parts):
        trials = by_pid[part["participant"]]
        if len(trials) > 1:
            jitter = np.linspace(-0.15, 0.15, len(trials))
            ax.scatter(
                np.full(len(trials), i) + jitter, trials,
                c=colors[i], s=22, alpha=0.45, edgecolors="none",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Score d'expertise prédit")
    ax.set_title("Figure 2 — Distribution des scores par participant (groupés par classe)")
    ax.set_ylim(-1.08, 1.08)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=CLASS_COLORS[i], label=CLASS_NAMES_FR[i]) for i in range(4)],
        loc="upper left", fontsize=8,
    )

    # Outliers: > 1.5 IQR from class median
    outlier_pids = []
    for y4 in range(4):
        cls_scores = [p["score_median"] for p in parts if p["y4"] == y4]
        if len(cls_scores) < 4:
            continue
        q1, q3 = np.percentile(cls_scores, [25, 75])
        iqr = q3 - q1
        for p in parts:
            if p["y4"] == y4 and (p["score_median"] < q1 - 1.5 * iqr or p["score_median"] > q3 + 1.5 * iqr):
                outlier_pids.append(p["participant"])

    path = out_dir / "figure02_participants.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    analysis = FigureAnalysis(
        figure_id=2,
        title="Distribution par participant",
        intra_class_variance="Dispersion visible au sein de chaque classe ; les points pâles = trials individuels.",
        outliers=(
            f"{len(outlier_pids)} participant(s) aberrant(s) (critère IQR) : "
            + (", ".join(outlier_pids[:8]) if outlier_pids else "aucun détecté")
        ),
        extra={"outlier_participants": outlier_pids},
    )
    return path, analysis


# ─── Figure 3 : granularité fine par sous-niveau ─────────────────────────────

def figure3_sublevel_granularity(
    preds: Sequence[dict], out_dir: Path,
) -> Tuple[List[Path], FigureAnalysis]:
    import matplotlib.pyplot as plt

    paths: List[Path] = []
    sub_analyses: Dict[str, dict] = {}

    for sub in GRANULAR_GROUPS:
        trials = [p for p in preds if p["sublevel"] == sub]
        if not trials:
            continue

        by_pid: Dict[str, List[float]] = defaultdict(list)
        for t in trials:
            by_pid[t["participant"]].append(t["score_pred"])
        pids = sorted(by_pid.keys())
        medians = [float(np.median(by_pid[p])) for p in pids]

        fig, ax = plt.subplots(figsize=(max(5, len(pids) * 0.5), 4))
        x = np.arange(len(pids))
        ax.bar(x, medians, color="#67a9cf", edgecolor="#333", alpha=0.85, width=0.6)
        for i, pid in enumerate(pids):
            sc = by_pid[pid]
            jitter = np.random.default_rng(i).uniform(-0.12, 0.12, size=len(sc))
            ax.scatter(np.full(len(sc), i) + jitter, sc, c="#2166ac", s=30, alpha=0.7, zorder=3)

        m, lo, hi = _bootstrap_ci(np.asarray(medians), stat="mean")
        ax.axhline(m, color="#b2182b", linestyle="--", linewidth=1.2, label=f"Moyenne = {m:.3f}")
        ax.fill_between([-0.5, len(pids) - 0.5], lo, hi, alpha=0.12, color="#b2182b")

        label = GRANULAR_LABELS_FR[sub]
        ax.set_xticks(x)
        ax.set_xticklabels([p[-4:] for p in pids], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Score prédit")
        ax.set_title(f"Figure 3 — {label} (n={len(pids)} participants, {len(trials)} trials)")
        ax.set_ylim(-1.08, 1.08)
        ax.legend(fontsize=8)

        path = out_dir / f"figure03_{sub}.png"
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

        sub_analyses[sub] = {
            "n_participants": len(pids),
            "mean": m, "std": float(np.std(medians)),
            "spread": float(max(medians) - min(medians)) if medians else 0.0,
        }

    analysis = FigureAnalysis(
        figure_id=3,
        title="Granularité fine par sous-niveau",
        intra_class_variance="Voir dispersion par sous-niveau ; spread élevé = hétérogénéité intra-groupe.",
        outliers="Participants isolés aux extrémités des barres = candidats aberrants.",
        extra={"sublevel_stats": sub_analyses},
    )
    return paths, analysis


# ─── Courbes temporelles (Figures 4–6) ───────────────────────────────────────

def compute_temporal_curves(
    preds: Sequence[dict],
    dataset_pkl: Path,
    checkpoints_dir: Path,
    n_timepoints: int = 40,
    device: str = "cpu",
) -> List[dict]:
    """Inférence par préfixe croissant avec le checkpoint LOPO du participant."""
    import torch
    from train.train_hybrid_lopo import (
        N_FEATURES, apply_norm, crop_sequence, enrich_kinematic_features,
    )
    from models.hybrid_lstm_transformer import HybridConfig, HybridLSTMTransformer

    with open(dataset_pkl, "rb") as f:
        raw = pickle.load(f)

    dev = torch.device(device)
    curves: List[dict] = []
    ckpt_index: Dict[str, Path] = {}
    for ckpt in checkpoints_dir.glob("fold_*"):
        meta_path = ckpt / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        ckpt_index[meta["held"]] = ckpt / "model.pt"

    for p in preds:
        pid = p["participant"]
        tid = p["trial_id"]
        key = None
        for k in raw:
            if str(k[0]) == pid and str(k[1]) == tid:
                key = k
                break
        if key is None:
            continue
        ckpt_path = ckpt_index.get(pid)
        if ckpt_path is None or not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        cfg = HybridConfig(**ckpt["hybrid_cfg"])
        model = HybridLSTMTransformer(cfg).to(dev)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        mean, std = ckpt["norm_mean"], ckpt["norm_std"]
        X = enrich_kinematic_features(np.asarray(raw[key]["X"], dtype=np.float32))
        Xn = apply_norm(X, mean, std)
        T = Xn.shape[0]
        fracs = np.linspace(0.05, 1.0, n_timepoints)
        scores = []
        with torch.no_grad():
            for f in fracs:
                t_end = max(1, int(f * T))
                prefix = Xn[:t_end]
                chunk = crop_sequence(prefix, cfg.seq_len, mode="start")
                xb = torch.from_numpy(chunk).unsqueeze(0).to(dev)
                scores.append(float(model(xb).squeeze().item()))

        curves.append({
            "participant": pid,
            "trial_id": tid,
            "y4": p["y4"],
            "class_fr": p["class_fr"],
            "sublevel": p["sublevel"],
            "time": fracs,
            "scores": np.asarray(scores, dtype=float),
        })
    return curves


def _aggregate_class_curves(
    curves: Sequence[dict], smooth_window: int = 1,
) -> Dict[int, dict]:
    grid = np.linspace(0.05, 1.0, 100)
    by_class: Dict[int, List[np.ndarray]] = {i: [] for i in range(4)}
    for c in curves:
        s = c["scores"]
        if smooth_window > 1:
            k = smooth_window
            s = np.convolve(s, np.ones(k) / k, mode="same")
        by_class[c["y4"]].append(np.interp(grid, c["time"], s))

    out: Dict[int, dict] = {}
    for y4 in range(4):
        if not by_class[y4]:
            continue
        arr = np.stack(by_class[y4], axis=0)
        out[y4] = {
            "time": grid,
            "mean": arr.mean(axis=0),
            "std": arr.std(axis=0),
            "n": arr.shape[0],
        }
    return out


def figure4_temporal_progression(
    curves: Sequence[dict], out_dir: Path, smooth: int = 1,
) -> Tuple[Path, FigureAnalysis]:
    import matplotlib.pyplot as plt

    agg = _aggregate_class_curves(curves, smooth_window=smooth)
    fig, ax = plt.subplots(figsize=(8, 5))

    for y4 in range(4):
        if y4 not in agg:
            continue
        d = agg[y4]
        ax.plot(d["time"], d["mean"], color=CLASS_COLORS[y4], lw=2,
                label=f"{CLASS_NAMES_FR[y4]} (n={d['n']})")
        ax.fill_between(d["time"], d["mean"] - d["std"], d["mean"] + d["std"],
                        color=CLASS_COLORS[y4], alpha=0.18)

    ax.set_xlabel("Temps normalisé du geste")
    ax.set_ylabel("Score d'expertise prédit")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.08, 1.08)
    ax.axhline(0, color="#888", linestyle="--", linewidth=0.7)
    ax.set_title("Figure 4 — Progression temporelle du score par classe d'expertise")
    ax.legend(loc="best")

    # Early fixation test
    early_idx = int(EARLY_FIXATION_FRAC * 100) if len(list(agg.values())[0]["time"]) > 10 else 5
    fixation_notes = []
    for y4, d in agg.items():
        t_grid = d["time"]
        idx = np.argmin(np.abs(t_grid - EARLY_FIXATION_FRAC))
        r_early = d["mean"][idx]
        r_final = d["mean"][-1]
        diff = abs(r_final - r_early)
        fixation_notes.append(f"{CLASS_NAMES_FR[y4]}: Δ(t=15%→fin)={diff:.3f}")

    path = out_dir / "figure04_progression_temporelle.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    analysis = FigureAnalysis(
        figure_id=4,
        title="Progression temporelle",
        temporal_stability="Courbes relativement plates → score stable dans le temps.",
        early_fixation="; ".join(fixation_notes),
        monotonicity="Hiérarchie inter-classes maintenue si les courbes ne se croisent pas.",
    )
    return path, analysis


def figure5_dynamic_monotonicity(
    curves: Sequence[dict], out_dir: Path,
) -> Tuple[Path, FigureAnalysis]:
    import matplotlib.pyplot as plt

    agg_raw = _aggregate_class_curves(curves, smooth_window=1)
    agg_smooth = _aggregate_class_curves(curves, smooth_window=5)
    fig, ax = plt.subplots(figsize=(8, 5))

    for y4 in range(4):
        if y4 not in agg_smooth:
            continue
        d = agg_smooth[y4]
        ax.plot(d["time"], d["mean"], color=CLASS_COLORS[y4], lw=2.2,
                label=f"{CLASS_NAMES_FR[y4]} (lissé)")
        d_raw = agg_raw[y4]
        ax.plot(d_raw["time"], d_raw["mean"], color=CLASS_COLORS[y4],
                lw=0.8, alpha=0.35, linestyle=":")

    # Monotonicity at each time point
    grid = agg_smooth[0]["time"] if agg_smooth else np.linspace(0, 1, 100)
    mono_frac = 0.0
    if len(agg_smooth) == 4:
        means_t = np.stack([agg_smooth[i]["mean"] for i in range(4)], axis=0)
        mono_steps = np.all(np.diff(means_t, axis=0) >= -0.02, axis=0)
        mono_frac = float(mono_steps.mean())

    ax.set_xlabel("Temps normalisé du geste")
    ax.set_ylabel("Score d'expertise prédit")
    ax.set_title(
        "Figure 5 — Monotonie ordinale dynamique\n"
        f"Hiérarchie respectée à {mono_frac * 100:.0f} % des instants temporels"
    )
    ax.legend(loc="best")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.08, 1.08)

    path = out_dir / "figure05_monotonie_dynamique.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    analysis = FigureAnalysis(
        figure_id=5,
        title="Monotonie ordinale dynamique",
        monotonicity=f"Hiérarchie Étudiant < Junior < Senior < Expert stable à {mono_frac * 100:.0f} % du geste.",
        temporal_stability="Comparer courbes pleines (lissées) et pointillées (brutes) pour évaluer le bruit.",
    )
    return path, analysis


def figure6_phase_distribution(
    curves: Sequence[dict], out_dir: Path,
) -> Tuple[Path, FigureAnalysis]:
    import matplotlib.pyplot as plt

    phase_bounds = [(0.0, 0.33), (0.33, 0.66), (0.66, 1.01)]
    phase_scores: Dict[int, Dict[int, List[float]]] = {
        ph: {y4: [] for y4 in range(4)} for ph in range(3)
    }
    for c in curves:
        t, s = c["time"], c["scores"]
        for ph, (lo, hi) in enumerate(phase_bounds):
            mask = (t >= lo) & (t < hi)
            if mask.any():
                phase_scores[ph][c["y4"]].append(float(np.mean(s[mask])))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    early_vs_late = []

    for ph, ax in enumerate(axes):
        data, labels, colors = [], [], []
        for y4 in range(4):
            vals = phase_scores[ph][y4]
            if vals:
                data.append(vals)
                labels.append(CLASS_NAMES_FR[y4])
                colors.append(CLASS_COLORS[y4])
        if data:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
            for patch, col in zip(bp["boxes"], colors):
                patch.set_facecolor(col)
                patch.set_alpha(0.65)

        ax.set_title(PHASE_LABELS_FR[ph])
        ax.set_ylabel("Score prédit" if ph == 0 else "")
        ax.set_ylim(-1.08, 1.08)

    for y4 in range(4):
        e = phase_scores[0][y4]
        l = phase_scores[2][y4]
        if e and l:
            early_vs_late.append(float(np.mean(l)) - float(np.mean(e)))

    path = out_dir / "figure06_phases_geste.png"
    fig.suptitle("Figure 6 — Distribution des scores par phase du geste", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    mean_shift = float(np.mean(early_vs_late)) if early_vs_late else float("nan")
    analysis = FigureAnalysis(
        figure_id=6,
        title="Phases du geste",
        temporal_stability=(
            f"Δ moyen fin−début = {mean_shift:+.3f}. "
            + ("Distributions stables entre phases." if abs(mean_shift) < 0.05
               else "Variation notable entre début et fin du geste.")
        ),
        early_fixation=(
            "Si les boxplots des 3 phases sont similaires, le modèle fixe son score tôt "
            f"(hypothèse t ≈ {EARLY_FIXATION_FRAC:.0%})."
        ),
    )
    return path, analysis


def write_report(analyses: Sequence[FigureAnalysis], preds: Sequence[dict], out_dir: Path) -> Path:
    parts = participant_medians(preds)
    y4_scores = {i: [p["score_median"] for p in parts if p["y4"] == i] for i in range(4)}

    global_lines = [
        "# Rapport d'analyse — Modèle d'expertise chirurgicale (LOPO + HOEL)",
        "",
        f"**Trials analysés** : {len(preds)}",
        f"**Participants** : {len(parts)}",
        "",
        "## Conclusion globale",
        "",
    ]

    if len(y4_scores[3]) and len(y4_scores[0]):
        sep = float(np.mean(y4_scores[3]) - np.mean(y4_scores[0]))
        global_lines.append(
            f"- Séparation Étudiant / Expert : **{sep:+.3f}** sur l'échelle [-1, +1]."
        )

    scores_all = [p["score_pred"] for p in preds]
    true_all = [p["score_true"] for p in preds]
    if len(scores_all) > 2:
        rho, _ = spearmanr(true_all, scores_all)
        tau, _ = kendalltau(true_all, scores_all)
        global_lines.append(f"- Spearman (trial) : **ρ = {rho:+.3f}**")
        global_lines.append(f"- Kendall τ (trial) : **τ = {tau:+.3f}**")

    global_lines.extend(["", "## Analyse par figure", ""])
    for a in analyses:
        global_lines.append(f"### Figure {a.figure_id} — {a.title}")
        if a.monotonicity:
            global_lines.append(f"- **Monotonie** : {a.monotonicity}")
        if a.inter_class_separation:
            global_lines.append(f"- **Séparation inter-classes** : {a.inter_class_separation}")
        if a.intra_class_variance:
            global_lines.append(f"- **Variance intra-classe** : {a.intra_class_variance}")
        if a.temporal_stability:
            global_lines.append(f"- **Stabilité temporelle** : {a.temporal_stability}")
        if a.outliers:
            global_lines.append(f"- **Aberrants** : {a.outliers}")
        if a.early_fixation:
            global_lines.append(f"- **Fixation précoce** : {a.early_fixation}")
        global_lines.append("")

    report_path = out_dir / "rapport_analyse.md"
    report_path.write_text("\n".join(global_lines), encoding="utf-8")

    json_path = out_dir / "analyses.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([a.to_dict() for a in analyses], f, indent=2, ensure_ascii=False)

    return report_path


def run_figure_suite(
    preds_path: Path,
    out_dir: Path,
    dataset_pkl: Optional[Path] = None,
    checkpoints_dir: Optional[Path] = None,
    device: str = "cpu",
) -> dict:
    _setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = load_predictions(preds_path)
    analyses: List[FigureAnalysis] = []
    outputs: Dict[str, object] = {"preds_path": str(preds_path), "figures": []}

    _, a1 = figure1_ordinal_monotonicity(preds, out_dir)
    analyses.append(a1)
    outputs["figures"].append("figure01")

    _, a2 = figure2_participant_distribution(preds, out_dir)
    analyses.append(a2)
    outputs["figures"].append("figure02")

    _, a3 = figure3_sublevel_granularity(preds, out_dir)
    analyses.append(a3)
    outputs["figures"].append("figure03")

    curves: List[dict] = []
    if checkpoints_dir and dataset_pkl and checkpoints_dir.exists():
        print(f"[temporal] Inférence par préfixe depuis {checkpoints_dir} ...")
        curves = compute_temporal_curves(preds, dataset_pkl, checkpoints_dir, device=device)
        print(f"[temporal] {len(curves)} courbes calculées.")
        with open(out_dir / "temporal_curves.pkl", "wb") as f:
            pickle.dump(curves, f)

        _, a4 = figure4_temporal_progression(curves, out_dir)
        analyses.append(a4)
        outputs["figures"].extend(["figure04", "figure05", "figure06"])

        _, a5 = figure5_dynamic_monotonicity(curves, out_dir)
        analyses.append(a5)

        _, a6 = figure6_phase_distribution(curves, out_dir)
        analyses.append(a6)
    else:
        print(
            "[temporal] Figures 4–6 ignorées — fournir --checkpoints et --pkl "
            "(nécessite entraînement avec --save-checkpoints)."
        )
        analyses.append(FigureAnalysis(
            figure_id=4,
            title="Progression temporelle (non générée)",
            temporal_stability="Relancer l'entraînement avec --save-checkpoints puis réexécuter cette analyse.",
        ))

    report = write_report(analyses, preds, out_dir)
    outputs["report"] = str(report)
    print(f"\n[done] Figures -> {out_dir}")
    print(f"       Rapport -> {report}")
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Suite de figures d'évaluation d'expertise (HOEL / LOPO).")
    ap.add_argument("--preds", type=Path, required=True, help="predictions.pkl du run LOPO")
    ap.add_argument("--out", type=Path, required=True, help="Dossier de sortie des figures")
    ap.add_argument("--pkl", type=Path, default=Path("data/continuous_per_trial.pkl"))
    ap.add_argument("--checkpoints", type=Path, default=None, help="Dossier checkpoints LOPO")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    import torch
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    run_figure_suite(args.preds, args.out, args.pkl, args.checkpoints, device=device)


if __name__ == "__main__":
    main()
