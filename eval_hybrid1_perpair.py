"""
eval_hybrid1_perpair.py
=======================
Scoring composite per-paire (eApp 2) + validation panneau E.

  1) Charger les 10 modeles entraines (1 par paire).
  2) Scorer chaque trial -> score_k ; composite = moyenne des 10 score_k.
  3) score participant = MOYENNE des scores composite d'essais (eApp 2 : averaging).
  4) Validite de construit : ANOVA + Tukey, barplot panneau E.
  5) Validite predictive : OLS score~year sur 25 middle.
  6) Spearman rho 9 sous-niveaux.

Sorties (results/hybrid1_perpair/) :
  predictions_composite.pkl, per_pair_scores.pkl, panelE_barplot.png,
  tukey_contrasts.csv, predictive_validity.json, spearman_9levels.json
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from data_hybrid1 import (
    GROUP4_DISPLAY,
    GROUP4_ORDER,
    PAIR_NAMES,
    SUBLEVEL_ORDER,
    SUBLEVEL_RANK,
    Trial,
    apply_norm,
    build_trials,
    iter_non_overlapping_windows,
    load_raw_trials,
    log_gpu_memory,
    resolve_hybrid1_paths,
    to_pair_trials,
)
from models_hybrid1 import Hybrid1Config, Hybrid1ModelA

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("eval_hybrid1_perpair")

PAPER_TARGET = {"expert": 0.75, "senior": 0.25, "junior": -0.09, "novice": -0.80}


def load_pair_model(run_dir: Path, pair_idx: int, device: torch.device):
    pair_name = PAIR_NAMES[pair_idx]
    pair_dir = run_dir / f"pair{pair_idx}_{pair_name}"
    cfg_path = pair_dir / "train_config.json"
    max_len, dropout = 5000, 0.30
    if cfg_path.exists():
        tc = json.loads(cfg_path.read_text())
        max_len = tc.get("max_len", max_len)
        dropout = tc.get("dropout", dropout)

    with open(pair_dir / "norm_stats.pkl", "rb") as f:
        ns = pickle.load(f)
    mean, std = ns["mean"], ns["std"]

    cfg = Hybrid1Config(n_features=1, max_len=max_len, dropout=dropout)
    model = Hybrid1ModelA(cfg).to(device)
    try:
        state = torch.load(pair_dir / "model_A_best.pt", map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(pair_dir / "model_A_best.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, mean, std, max_len


@torch.no_grad()
def score_window(model, X_norm: np.ndarray, device: torch.device) -> float:
    L = X_norm.shape[0]
    xb = torch.from_numpy(X_norm).float().unsqueeze(0).to(device)
    mask = torch.zeros(1, L, dtype=torch.bool, device=device)
    out = model(xb, key_padding_mask=mask)
    return float(out.squeeze().cpu().item())


def score_trial_windows(
    model, X: np.ndarray, mean: np.ndarray, std: np.ndarray,
    device: torch.device, window_len: int,
) -> float:
    """Score un trial : moyenne des scores de fenetres non chevauchantes."""
    scores = []
    for chunk, _ in iter_non_overlapping_windows(X, window_len):
        Xn = apply_norm(chunk, mean, std)
        scores.append(score_window(model, Xn, device))
    return float(np.mean(scores))


def score_all_pairs(
    pair_trials: list[list[Trial]],
    run_dir: Path,
    device: torch.device,
) -> np.ndarray:
    """Renvoie (n_trials, 10) scores par paire."""
    n_trials = len(pair_trials[0])
    scores = np.zeros((n_trials, 10), dtype=np.float64)

    for k in range(10):
        model, mean, std, max_len = load_pair_model(run_dir, k, device)
        logger.info(f"[score] paire {k} ({PAIR_NAMES[k]}) max_len={max_len}")
        for i, t in enumerate(pair_trials[k]):
            scores[i, k] = score_trial_windows(model, t.X, mean, std, device, max_len)
        del model
        torch.cuda.empty_cache()
        log_gpu_memory(f"after pair {k}")

    return scores


def ci95(vals: np.ndarray):
    vals = np.asarray(vals, dtype=np.float64)
    n = len(vals)
    m = float(vals.mean())
    if n < 2:
        return m, 0.0
    sem = float(vals.std(ddof=1) / np.sqrt(n))
    h = sem * float(stats.t.ppf(0.975, n - 1))
    return m, h


def significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def make_barplot(part_df, tukey_df, anova_p, out_path: Path, seed: int):
    groups = GROUP4_ORDER
    labels = [GROUP4_DISPLAY[g] for g in groups]
    means, halves, ns = [], [], []
    for g in groups:
        vals = part_df.loc[part_df["group4"] == g, "score"].values
        m, h = ci95(vals)
        means.append(m)
        halves.append(h)
        ns.append(len(vals))

    colors = ["#1a6b3c", "#4a9e6f", "#e0a44c", "#c0392b"]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    x = np.arange(len(groups))
    ax.bar(x, means, yerr=halves, capsize=6, color=colors,
           edgecolor="black", linewidth=1.0, width=0.68)
    ax.axhline(0.0, color="gray", lw=0.8, ls="--", zorder=0)

    for xi, m, n in zip(x, means, ns):
        va = "bottom" if m >= 0 else "top"
        off = 0.03 if m >= 0 else -0.03
        ax.text(xi, m + off, f"{m:+.2f}\n(n={n})", ha="center", va=va, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score composite (moyenne participant)", fontsize=11)
    ax.set_title(
        f"Panneau E — Hybrid1 per-pair (seed {seed})\nANOVA p = {anova_p:.3g}",
        fontsize=12,
    )
    ax.set_ylim(-1.15, 1.15)

    pos = {g: i for i, g in enumerate(groups)}
    sig_pairs = []
    for _, row in tukey_df.iterrows():
        g1, g2 = row["group1"], row["group2"]
        if g1 in pos and g2 in pos and row["reject"]:
            sig_pairs.append((g1, g2, row["p_adj"]))
    sig_pairs.sort(key=lambda t: abs(pos[t[0]] - pos[t[1]]))

    y0, step = 0.80, 0.11
    for k, (g1, g2, padj) in enumerate(sig_pairs):
        i, j = sorted([pos[g1], pos[g2]])
        y = y0 + k * step
        ax.plot([i, i, j, j], [y, y + 0.02, y + 0.02, y], color="black", lw=1.2)
        star = significance_stars(padj)
        emph = " (E vs S)" if {g1, g2} == {"expert", "senior"} else ""
        ax.text((i + j) / 2, y + 0.025, f"{star}{emph} p={padj:.3g}",
                ha="center", va="bottom", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval Hybrid1 per-pair composite")
    ap.add_argument("--data", choices=["v1", "v2"], default=None,
                    help="Version trial_tensor (fixe pkl + run si non surcharges)")
    ap.add_argument("--pkl", type=Path, default=None)
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--butterworth", action=argparse.BooleanOptionalAction, default=False,
                    help="Defaut False : v1/v2 deja filtres en amont")
    args = ap.parse_args()

    pkl_path, run_dir = resolve_hybrid1_paths(args.data, args.pkl, args.run)
    args.pkl = pkl_path
    args.run = run_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.run.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("EVAL Hybrid1 per-pair — score composite + panneau E")
    logger.info("=" * 70)
    logger.info(f"[config] data={args.data or 'legacy'} pkl={args.pkl} run={args.run}")
    logger.info(f"[config] seed={args.seed} device={device} butterworth={args.butterworth}")

    if not args.pkl.exists():
        raise FileNotFoundError(f"Donnees manquantes : {args.pkl}")

    raw = load_raw_trials(str(args.pkl))
    all_trials = build_trials(
        raw, max_len=5000, full_resolution=True, apply_butterworth=args.butterworth,
    )
    logger.info(f"[data] {len(all_trials)} trials pleine resolution")

    pair_trials = [to_pair_trials(all_trials, k) for k in range(10)]
    per_pair_scores = score_all_pairs(pair_trials, args.run, device)

    composite_trial = per_pair_scores.mean(axis=1)
    logger.info(f"[composite] score essai = moyenne 10 paires "
                f"(min={composite_trial.min():+.3f} max={composite_trial.max():+.3f})")

    import collections
    by_part: dict[str, list[float]] = collections.defaultdict(list)
    part_meta: dict[str, tuple] = {}
    trial_rows = []
    for t, comp, pair_sc in zip(all_trials, composite_trial, per_pair_scores):
        trial_rows.append({
            "participant": t.participant,
            "trial": t.trial,
            "group4": t.group4,
            "level9": t.level9,
            "year": t.year,
            "score_composite": float(comp),
            "scores_per_pair": {PAIR_NAMES[k]: float(pair_sc[k]) for k in range(10)},
        })
        by_part[t.participant].append(float(comp))
        part_meta[t.participant] = (t.group4, t.level9, t.year)

    part_rows = []
    for p, scs in by_part.items():
        g4, lv9, yr = part_meta[p]
        part_rows.append({
            "participant": p,
            "group4": g4,
            "level9": lv9,
            "year": yr,
            "score": float(np.mean(scs)),
            "n_trials": len(scs),
        })
    logger.info(f"[participants] {len(part_rows)} — score = MOYENNE des essais (eApp 2)")

    with open(args.run / "per_pair_scores.pkl", "wb") as f:
        pickle.dump({"trials": trial_rows, "matrix": per_pair_scores}, f)
    with open(args.run / "predictions_composite.pkl", "wb") as f:
        pickle.dump(part_rows, f)

    import pandas as pd
    part_df = pd.DataFrame(part_rows)

    logger.info("\n[panneau E] Moyenne (IC95) du score participant par groupe :")
    summary = {}
    for g in GROUP4_ORDER:
        vals = part_df.loc[part_df["group4"] == g, "score"].values
        m, h = ci95(vals)
        summary[g] = {"mean": m, "ci95_half": h, "n": len(vals)}
        target = PAPER_TARGET[g]
        logger.info(f"   {GROUP4_DISPLAY[g]:9s} : {m:+.3f} +/- {h:.3f}  "
                    f"(n={len(vals)}, cible ~{target:+.2f})")

    group_vals = [part_df.loc[part_df["group4"] == g, "score"].values for g in GROUP4_ORDER]
    f_stat, anova_p = stats.f_oneway(*group_vals)
    logger.info(f"\n[ANOVA] F={f_stat:.3f}  p={anova_p:.4g}")

    tukey = pairwise_tukeyhsd(part_df["score"].values, part_df["group4"].values)
    tukey_df = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0],
    )
    tukey_df.rename(columns={"p-adj": "p_adj"}, inplace=True)
    tukey_df["reject"] = tukey_df["reject"].astype(bool)
    tukey_df["expert_vs_senior"] = tukey_df.apply(
        lambda r: {r["group1"], r["group2"]} == {"expert", "senior"}, axis=1,
    )

    logger.info("\n[Tukey HSD] 6 contrastes :")
    for _, r in tukey_df.iterrows():
        pair = f"{r['group1']} vs {r['group2']}"
        emph = "  <== EXPERT vs SENIOR" if {r["group1"], r["group2"]} == {"expert", "senior"} else ""
        logger.info(f"   {pair:22s} diff={r['meandiff']:+.3f} p_adj={r['p_adj']:.4g} "
                    f"{significance_stars(r['p_adj'])}{emph}")

    tukey_df.to_csv(args.run / "tukey_contrasts.csv", index=False)

    ranks = part_df["level9"].map(SUBLEVEL_RANK)
    valid = ranks.notna()
    rho, sp_p = stats.spearmanr(ranks[valid].values, part_df["score"][valid].values)
    logger.info(f"\n[Spearman 9 sous-niveaux] rho={rho:+.3f}  p={sp_p:.4g}")
    for sl in SUBLEVEL_ORDER:
        vals = part_df.loc[part_df["level9"] == sl, "score"].values
        if len(vals):
            logger.info(f"      {sl:7s} : {vals.mean():+.3f} (n={len(vals)})")

    mid = part_df[part_df["group4"].isin(["junior", "senior"])].copy()
    logger.info(f"\n[validite predictive] OLS score~year sur {len(mid)} middle :")
    X_ols = sm.add_constant(mid["year"].values.astype(float))
    y_ols = mid["score"].values.astype(float)
    ols = sm.OLS(y_ols, X_ols).fit()
    slope = float(ols.params[1])
    intercept = float(ols.params[0])
    r2 = float(ols.rsquared)
    yhat = ols.predict(X_ols)
    mse = float(np.mean((y_ols - yhat) ** 2))
    mae = float(np.mean(np.abs(y_ols - yhat)))
    logger.info(f"   pente={slope:+.4f}  R2={r2:.4f}  MSE={mse:.4f}  MAE={mae:.4f}")

    pred_validity = {
        "seed": args.seed,
        "n_middle": int(len(mid)),
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "mse": mse,
        "mae": mae,
    }
    with open(args.run / "predictive_validity.json", "w", encoding="utf-8") as f:
        json.dump(pred_validity, f, indent=2)

    spearman_9 = {"rho": float(rho), "p": float(sp_p)}
    with open(args.run / "spearman_9levels.json", "w", encoding="utf-8") as f:
        json.dump(spearman_9, f, indent=2)

    make_barplot(part_df, tukey_df, anova_p, args.run / "panelE_barplot.png", args.seed)

    e_vs_s = tukey_df[
        tukey_df.apply(lambda r: {r["group1"], r["group2"]} == {"expert", "senior"}, axis=1)
    ].iloc[0]
    report = {
        "seed": args.seed,
        "data": args.data,
        "pkl": str(args.pkl),
        "protocol": "perpair_composite_mean",
        "group_means_ci95": summary,
        "anova": {"F": float(f_stat), "p": float(anova_p)},
        "expert_vs_senior": {
            "diff": float(e_vs_s["meandiff"]),
            "p_adj": float(e_vs_s["p_adj"]),
            "significant": bool(e_vs_s["reject"]),
        },
        "spearman_9sublevels": {"rho": float(rho), "p": float(sp_p)},
        "predictive_validity_middle": pred_validity,
        "paper_targets": PAPER_TARGET,
    }
    with open(args.run / "panelE_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("\n[GATE] Cible papier : Exp~0.75 Sen~0.25 Jun~-0.09 Nov~-0.80 ; E-vs-S p~.045")
    logger.info(f"[GATE] Obtenu : Exp={summary['expert']['mean']:+.2f} "
                f"Sen={summary['senior']['mean']:+.2f} "
                f"Jun={summary['junior']['mean']:+.2f} "
                f"Nov={summary['novice']['mean']:+.2f} ; E-vs-S p={float(e_vs_s['p_adj']):.3g}")
    logger.info(f"[save] -> {args.run}")


if __name__ == "__main__":
    main()
