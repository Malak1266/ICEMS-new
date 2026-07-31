"""
sublevel_analysis.py
====================
Analyse publication par 9 sous-niveaux cliniques (scores prédits LOPO).

Produit :
  - Violin plot (médiane participant par sous-niveau)
  - Scatter score prédit vs cible ordinale y9
  - Rapport console : monotonie, Kruskal-Wallis, Spearman, Mann-Whitney adjacents

Usage :
    python src/sublevel_analysis.py --preds results/lopo_sublevels_full/lopo_predictions.pkl
    python src/sublevel_analysis.py --preds results/run/lopo_predictions.pkl --out results/sublevel --prefix icems
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import kruskal, mannwhitneyu, pearsonr, spearmanr

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

SUBLEVEL_NAMES = [
    "Medical student", "PGY1", "PGY2", "PGY3", "PGY4",
    "PGY5", "PGY6", "Fellow", "Staff",
]
SUBLEVEL_SHORT = ["MS", "PGY1", "PGY2", "PGY3", "PGY4", "PGY5", "PGY6", "Fellow", "Staff"]
Y9_TO_REG = np.array(
    [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.86, 1.0], dtype=np.float64,
)
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
N_ADJACENT = 8


def _sublevel_color(y9: int):
    import matplotlib.pyplot as plt
    return plt.get_cmap("RdYlGn")(y9 / 8.0)


def load_preds(source: Union[Path, str, Sequence[dict]]) -> List[dict]:
    """Charge depuis lopo_predictions.pkl ou une liste de dicts Step B."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "preds" in payload:
            return list(payload["preds"])
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Format pickle non reconnu : {path}")
    return list(source)


def participant_table(preds: Sequence[dict]) -> List[dict]:
    """Une ligne par participant : y9 clinique + médiane des scores trial LOPO."""
    by_pid: Dict[str, List[dict]] = defaultdict(list)
    for row in preds:
        pid = str(row.get("participant") or row.get("pid", ""))
        if not pid:
            key = row.get("key")
            if key:
                pid = str(key[0])
        by_pid[pid].append(row)

    rows: List[dict] = []
    for pid, trials in sorted(by_pid.items()):
        y9_vals = [int(t["y9"]) for t in trials if "y9" in t]
        if not y9_vals:
            raise ValueError(
                f"Participant {pid!r} sans champ y9 — relancez LOPO avec step_B à jour."
            )
        y9 = Counter(y9_vals).most_common(1)[0][0]
        scores = [float(t["score"]) for t in trials]
        rows.append({
            "participant": pid,
            "y9": y9,
            "sublevel": SUBLEVEL_NAMES[y9],
            "n_trials": len(trials),
            "score_median": float(np.median(scores)),
            "score_mean": float(np.mean(scores)),
        })
    return rows


def _bootstrap_median_ci(values: np.ndarray) -> Tuple[float, float, float]:
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        v = float(values[0])
        return v, v, v
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots = np.array([
        np.median(rng.choice(values, size=len(values), replace=True))
        for _ in range(N_BOOTSTRAP)
    ])
    return float(np.median(values)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def compute_statistics(part_rows: Sequence[dict]) -> dict:
    """Stats participant-level (n = nombre de participants)."""
    by_y9: Dict[int, List[float]] = {i: [] for i in range(9)}
    for row in part_rows:
        by_y9[int(row["y9"])].append(float(row["score_median"]))

    group_medians = []
    for y9 in range(9):
        vals = by_y9[y9]
        med, lo, hi = _bootstrap_median_ci(np.array(vals, dtype=float))
        group_medians.append(med)
        print(f"  {SUBLEVEL_SHORT[y9]:>6}  n_part={len(vals):3d}  n_trials={sum(r['n_trials'] for r in part_rows if r['y9']==y9):3d}  "
              f"médiane={med:+.3f}  IC95%=[{lo:+.3f}, {hi:+.3f}]")

    # Monotonie : 8 paires consécutives (médianes de groupe)
    increasing = 0
    comparable = 0
    inversions = []
    for i in range(8):
        a, b = group_medians[i], group_medians[i + 1]
        if np.isnan(a) or np.isnan(b):
            continue
        comparable += 1
        if b > a:
            increasing += 1
        else:
            inversions.append(f"{SUBLEVEL_SHORT[i]}→{SUBLEVEL_SHORT[i+1]} ({a:+.3f} vs {b:+.3f})")

    ranks = np.array([int(r["y9"]) for r in part_rows], dtype=float)
    scores = np.array([float(r["score_median"]) for r in part_rows], dtype=float)
    if len(part_rows) >= 2:
        rho_s, p_s = spearmanr(ranks, scores)
        rho_p, p_p = pearsonr(Y9_TO_REG[ranks.astype(int)], scores)
    else:
        rho_s, p_s, rho_p, p_p = float("nan"), float("nan"), float("nan"), float("nan")

    non_empty = [by_y9[i] for i in range(9) if len(by_y9[i]) > 0]
    if len(non_empty) >= 2:
        h_kw, p_kw = kruskal(*non_empty)
    else:
        h_kw, p_kw = float("nan"), float("nan")

    print("\n" + "=" * 60)
    print(" Monotonie (médianes par sous-niveau)")
    print("=" * 60)
    print(f"  Paires consécutives croissantes : {increasing}/{comparable or N_ADJACENT}")
    if inversions:
        print(f"  Inversions : {', '.join(inversions)}")
    pub = comparable >= 4 and increasing >= max(6, int(0.75 * comparable))
    print(f"  Seuil publication (≥6/8)        : {'OK' if pub else 'NON ATTEINT (attendre LOPO full)'}")

    print("\n" + "=" * 60)
    print(" Tests globaux (niveau participant)")
    print("=" * 60)
    print(f"  Spearman ρ (rang y9 vs score médian)  : {rho_s:+.4f}  (p = {p_s:.4e}, n = {len(part_rows)})")
    print(f"  Pearson  r (y9_reg vs score médian)   : {rho_p:+.4f}  (p = {p_p:.4e})")
    print(f"  Kruskal-Wallis (9 sous-niveaux)       : H = {h_kw:.4f}  (p = {p_kw:.4e})")

    print("\n  Mann-Whitney paires adjacentes (Bonferroni ×8) :")
    mw_rows = []
    for i in range(8):
        a, b = by_y9[i], by_y9[i + 1]
        label = f"{SUBLEVEL_SHORT[i]} vs {SUBLEVEL_SHORT[i + 1]}"
        if len(a) == 0 or len(b) == 0:
            print(f"    {label:<22}  n/a (groupe vide)")
            mw_rows.append({"pair": label, "p_adj": None})
            continue
        res = mannwhitneyu(a, b, alternative="two-sided")
        p_adj = min(float(res.pvalue) * N_ADJACENT, 1.0)
        sig = "*" if p_adj < 0.05 else ""
        print(f"    {label:<22}  U={res.statistic:.0f}  p={res.pvalue:.4e}  p_adj={p_adj:.4e}{sig}")
        mw_rows.append({"pair": label, "u": float(res.statistic), "p_raw": float(res.pvalue), "p_adj": p_adj})

    return {
        "n_participants": len(part_rows),
        "n_trials": sum(r["n_trials"] for r in part_rows),
        "group_medians": group_medians,
        "monotonicity_increasing": increasing,
        "monotonicity_comparable": comparable,
        "monotonicity_total": N_ADJACENT,
        "monotonicity_publishable": pub,
        "inversions": inversions,
        "spearman_rho": float(rho_s),
        "spearman_p": float(p_s),
        "pearson_r": float(rho_p),
        "pearson_p": float(p_p),
        "kruskal_h": float(h_kw),
        "kruskal_p": float(p_kw),
        "mann_whitney": mw_rows,
        "participant_rows": list(part_rows),
    }


def plot_violins(part_rows: Sequence[dict], out_path: Path, stats: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_y9: Dict[int, List[float]] = {i: [] for i in range(9)}
    for row in part_rows:
        by_y9[int(row["y9"])].append(float(row["score_median"]))

    positions = list(range(9))
    colors = [_sublevel_color(i) for i in range(9)]

    fig, ax = plt.subplots(figsize=(13, 6))

    violin_data = []
    violin_pos = []
    for y9 in range(9):
        if len(by_y9[y9]) >= 2:
            violin_data.append(by_y9[y9])
            violin_pos.append(y9)
    if violin_data:
        parts = ax.violinplot(
            violin_data,
            positions=violin_pos,
            widths=0.75,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for i, body in enumerate(parts["bodies"]):
            y9 = violin_pos[i]
            body.set_facecolor(colors[y9])
            body.set_alpha(0.55)
            body.set_edgecolor("black")
            body.set_linewidth(0.6)

    for y9 in range(9):
        vals = by_y9[y9]
        if not vals:
            continue
        med = float(np.median(vals))
        ax.scatter(y9, med, marker="D", s=45, color="black", zorder=5)
        if len(vals) >= 2:
            _, lo, hi = _bootstrap_median_ci(np.array(vals))
            ax.errorbar(y9, med, yerr=[[med - lo], [hi - med]], fmt="none", color="black", capsize=4, lw=1.2)
        elif len(vals) == 1:
            ax.scatter(y9, med, marker="o", s=80, facecolors="none", edgecolors="black", zorder=4)

    for y9 in range(9):
        ax.axhline(float(Y9_TO_REG[y9]), xmin=(y9 - 0.4) / 9, xmax=(y9 + 0.4) / 9,
                   color="gray", ls=":", lw=0.8, alpha=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{SUBLEVEL_SHORT[i]}\n(n={len(by_y9[i])})" for i in range(9)],
        fontsize=9,
    )
    ax.set_ylabel("Score prédit médian (par participant)")
    ax.set_xlabel("Sous-niveau clinique")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(
        "Distribution du score LOPO par sous-niveau\n"
        f"losanges = médianes · barres = IC 95% bootstrap · "
        f"Spearman ρ = {stats['spearman_rho']:+.3f}"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def plot_scatter_sublevels(part_rows: Sequence[dict], out_path: Path, stats: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    for row in part_rows:
        y9 = int(row["y9"])
        ax.scatter(
            Y9_TO_REG[y9], row["score_median"],
            c=[_sublevel_color(y9)], s=50, alpha=0.75, edgecolors="k", linewidths=0.4,
        )
    lims = [-1.1, 1.1]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.4)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Cible ordinale y9 (dataset)")
    ax.set_ylabel("Score prédit médian (LOPO)")
    ax.set_title(
        f"Calibration ordinale — Pearson r = {stats['pearson_r']:+.3f}\n"
        f"(n = {stats['n_participants']} participants)"
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def run_sublevel_analysis(
    source: Union[Path, str, Sequence[dict]],
    out_dir: Union[Path, str],
    prefix: str = "icems",
) -> Tuple[List[dict], dict]:
    """Point d'entrée programmatique (Option B depuis step_B)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = load_preds(source)
    if not preds:
        raise ValueError("Aucune prédiction fournie.")

    print("=" * 60)
    print(" Analyse sous-niveaux (9 groupes cliniques)")
    print("=" * 60)
    print(f"\n  Trials LOPO : {len(preds)}")

    part_rows = participant_table(preds)
    print(f"  Participants : {len(part_rows)}")
    print("\n  Médianes par sous-niveau :")

    stats = compute_statistics(part_rows)

    print("\n[Figures]")
    plot_violins(part_rows, out_dir / f"{prefix}_violins_9sublevels.png", stats)
    plot_scatter_sublevels(part_rows, out_dir / f"{prefix}_scatter_9sublevels.png", stats)

    stats_path = out_dir / f"{prefix}_sublevel_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in stats.items() if k != "participant_rows"}, f, indent=2)
    print(f"  → {stats_path}")

    return part_rows, stats


def main():
    ap = argparse.ArgumentParser(description="Analyse LOPO par 9 sous-niveaux cliniques.")
    ap.add_argument(
        "--preds", type=Path, required=True,
        help="lopo_predictions.pkl ou chemin vers le dossier --out Step B",
    )
    ap.add_argument("--out", type=Path, default=None, help="Dossier de sortie (défaut : parent de --preds).")
    ap.add_argument("--prefix", type=str, default="icems_lopo")
    args = ap.parse_args()

    pred_path = args.preds
    if pred_path.is_dir():
        pred_path = pred_path / "lopo_predictions.pkl"
    out_dir = args.out or pred_path.parent

    run_sublevel_analysis(pred_path, out_dir, prefix=args.prefix)
    print(f"\n✅ Analyse terminée — {out_dir.resolve()}")


if __name__ == "__main__":
    main()
