"""
eval_hybrid1.py
===============
Scoring & evaluation du Modele A (Hybrid1 faithful) — reproduction du panneau E.

Etapes :
  1) Charger le modele entraine + stats de normalisation (seed donnee).
  2) Scorer les 136 trials avec le Modele A -> score essai.
  3) score participant = moyenne OU médiane des scores d'essais
     (défaut SPIE = moyenne, comparabilité papier ; --agg median = sensibilité).
  4) Produire :
       * predictions_A_faithful.pkl : {participant, group4, level9, year, score}
       * Barplot 4 groupes (Experts/Seniors/Juniors/Novices) : moyenne + IC95,
         ANOVA + Tukey, brackets de significativite (style panneau E).
       * Table des 6 contrastes Tukey, en surlignant EXPERT vs SENIOR.
       * Spearman rho sur les 9 sous-niveaux (monotonicite).
       * Validite predictive : OLS score~year sur les 25 middle (juniors+seniors)
         -> R2, MSE, MAE, pente.
  5) Logger config + seed.

Usage :
    python eval_hybrid1.py --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from data_hybrid1 import (
    GROUP4_DISPLAY,
    GROUP4_ORDER,
    SUBLEVEL_ORDER,
    SUBLEVEL_RANK,
    TrialDataset,
    build_trials,
    collate_pad,
    load_raw_trials,
)
from models_hybrid1 import Hybrid1Config, Hybrid1ModelA
from metrics_middle import compute_metrics_middle

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("eval_hybrid1")


def _agg_fn(agg: str):
    return (lambda sc: float(np.mean(sc))) if agg == "mean" else (lambda sc: float(np.median(sc)))


def score_all_trials(model, trials, mean, std, device, batch_size=16):
    """Renvoie un vecteur de scores essai (dans l'ordre de `trials`)."""
    ds = TrialDataset(trials, mean, std, target="y_reg")  # target ignore ici
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)
    model.eval()
    scores = []
    with torch.no_grad():
        for X, mask, _ in loader:
            X, mask = X.to(device), mask.to(device)
            out = model(X, key_padding_mask=mask)  # (B, 1)
            scores.extend(out.squeeze(-1).cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float64)


def ci95(vals: np.ndarray):
    """Moyenne + demi-largeur IC95 (t de Student)."""
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


def make_barplot(part_df, tukey_df, anova_p, out_path: Path, seed: int, agg: str = "mean"):
    """Barplot 4 groupes avec IC95 + brackets de significativite (style panneau E)."""
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
    bars = ax.bar(x, means, yerr=halves, capsize=6, color=colors,
                  edgecolor="black", linewidth=1.0, width=0.68)
    ax.axhline(0.0, color="gray", lw=0.8, ls="--", zorder=0)

    for xi, m, n in zip(x, means, ns):
        va = "bottom" if m >= 0 else "top"
        off = 0.03 if m >= 0 else -0.03
        ax.text(xi, m + off, f"{m:+.2f}\n(n={n})", ha="center", va=va, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(f"Score d'expertise predit ({agg} participant)", fontsize=11)
    ax.set_title(f"Panneau E — Score par groupe d'expertise (seed {seed})\n"
                 f"ANOVA p = {anova_p:.3g}", fontsize=12)
    ax.set_ylim(-1.15, 1.15)

    # Brackets de significativite (paires significatives selon Tukey).
    pos = {g: i for i, g in enumerate(groups)}
    sig_pairs = []
    for _, row in tukey_df.iterrows():
        g1, g2 = row["group1"], row["group2"]
        if g1 in pos and g2 in pos and row["reject"]:
            sig_pairs.append((g1, g2, row["p_adj"]))
    # Trier par distance (paires proches d'abord -> brackets bas)
    sig_pairs.sort(key=lambda t: abs(pos[t[0]] - pos[t[1]]))

    y0 = 0.80
    step = 0.11
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
    ap = argparse.ArgumentParser(description="Eval Hybrid1 Model A — panneau E")
    ap.add_argument("--pkl", type=Path, default=Path("data/trial_tensor_v2.pkl"))
    ap.add_argument("--run", type=Path, default=Path("results/hybrid1_faithful"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--condition", type=str, default=None,
                    help="étiquette condition (gap/attn/...) ; défaut = pool_type du train")
    ap.add_argument("--bootstrap-B", type=int, default=5000)
    ap.add_argument(
        "--agg", choices=["mean", "median"], default="mean",
        help="agrégation essai→participant (SPIE principal=mean ; median=sensibilité)",
    )
    ap.add_argument(
        "--agg-both", action="store_true",
        help="calcule aussi la sensibilité avec l'autre agrégation",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = args.run / f"seed{args.seed}"

    logger.info("=" * 70)
    logger.info("EVAL Hybrid1 Model A — reproduction panneau E")
    logger.info("=" * 70)
    logger.info(f"[config] seed={args.seed} device={device} run_dir={run_dir}")

    # Charger config d'entrainement (pour max_len/dropout/pooling reels).
    cfg_path = run_dir / "train_config.json"
    max_len, dropout = args.max_len, args.dropout
    pool_kw = {
        "pool_type": "gap",
        "pool_heads": 1,
        "pool_d_attn": None,
        "pool_tau": 1.0,
        "pool_att_dropout": 0.0,
        "pool_time_shuffle": False,
    }
    if cfg_path.exists():
        tc = json.loads(cfg_path.read_text())
        max_len = tc.get("max_len", max_len)
        dropout = tc.get("dropout", dropout)
        for k in pool_kw:
            if k in tc:
                pool_kw[k] = tc[k]
        logger.info(f"[config] repris de train_config.json : max_len={max_len} "
                    f"dropout={dropout} pool={pool_kw['pool_type']} "
                    f"best_epoch={tc.get('best_epoch')}")

    with open(run_dir / "norm_stats.pkl", "rb") as f:
        ns = pickle.load(f)
    mean, std = ns["mean"], ns["std"]

    cfg = Hybrid1Config(n_features=10, max_len=max_len, dropout=dropout, **pool_kw)
    model = Hybrid1ModelA(cfg).to(device)
    state = torch.load(run_dir / "model_A_best.pt", map_location=device)
    model.load_state_dict(state)
    logger.info("[model] Model A charge (poids best).")

    raw = load_raw_trials(str(args.pkl))
    trials = build_trials(raw, max_len=max_len)
    logger.info(f"[data] {len(trials)} trials scores avec le Modele A.")

    trial_scores = score_all_trials(model, trials, mean, std, device)

    # score participant = mean (SPIE) ou median (sensibilité)
    import collections
    by_part = collections.defaultdict(list)
    part_meta = {}
    for t, s in zip(trials, trial_scores):
        by_part[t.participant].append(s)
        part_meta[t.participant] = (t.group4, t.level9, t.year)

    agg_fn = _agg_fn(args.agg)
    rows = []
    for p, sc in by_part.items():
        g4, lv9, yr = part_meta[p]
        rows.append({
            "participant": p,
            "group4": g4,
            "level9": lv9,
            "year": yr,
            "score": agg_fn(sc),
            "n_trials": len(sc),
            "agg": args.agg,
        })
    logger.info(
        f"[participants] {len(rows)} participants "
        f"(score = {args.agg.upper()} des essais)."
    )

    # Sauvegarde predictions
    with open(run_dir / "predictions_A_faithful.pkl", "wb") as f:
        pickle.dump(rows, f)

    import pandas as pd
    part_df = pd.DataFrame(rows)

    # ---- Barplot 4 groupes : moyennes + IC95 -------------------------------
    logger.info("\n[panneau E] Moyenne (IC95) du score participant par groupe :")
    summary = {}
    for g in GROUP4_ORDER:
        vals = part_df.loc[part_df["group4"] == g, "score"].values
        m, h = ci95(vals)
        summary[g] = {"mean": m, "ci95_half": h, "n": len(vals)}
        logger.info(f"   {GROUP4_DISPLAY[g]:9s} : {m:+.3f}  +/- {h:.3f}  (n={len(vals)})")

    # ---- ANOVA -------------------------------------------------------------
    group_vals = [part_df.loc[part_df["group4"] == g, "score"].values
                  for g in GROUP4_ORDER]
    f_stat, anova_p = stats.f_oneway(*group_vals)
    logger.info(f"\n[ANOVA] F={f_stat:.3f}  p={anova_p:.4g}")

    # ---- Tukey HSD ---------------------------------------------------------
    tukey = pairwise_tukeyhsd(part_df["score"].values, part_df["group4"].values)
    tukey_df = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0],
    )
    tukey_df.rename(columns={"p-adj": "p_adj", "meandiff": "meandiff"}, inplace=True)
    tukey_df["reject"] = tukey_df["reject"].astype(bool)

    logger.info("\n[Tukey HSD] 6 contrastes (surligne = EXPERT vs SENIOR) :")
    logger.info(f"   {'contraste':22s} {'diff':>8s} {'IC95 low':>9s} "
                f"{'IC95 up':>9s} {'p_adj':>9s}  signif")
    for _, r in tukey_df.iterrows():
        pair = f"{r['group1']} vs {r['group2']}"
        emph = "  <== EXPERT vs SENIOR" if {r["group1"], r["group2"]} == {"expert", "senior"} else ""
        logger.info(f"   {pair:22s} {r['meandiff']:+8.3f} {r['lower']:+9.3f} "
                    f"{r['upper']:+9.3f} {r['p_adj']:9.4g}  "
                    f"{significance_stars(r['p_adj'])}{emph}")

    tukey_df.to_csv(run_dir / "tukey_contrasts.csv", index=False)

    # ---- Spearman sur les 9 sous-niveaux -----------------------------------
    ranks = part_df["level9"].map(SUBLEVEL_RANK)
    valid = ranks.notna()
    rho, sp_p = stats.spearmanr(ranks[valid].values, part_df["score"][valid].values)
    logger.info(f"\n[Spearman 9 sous-niveaux] rho={rho:+.3f}  p={sp_p:.4g} "
                f"(monotonicite score vs niveau)")
    # Moyenne par sous-niveau (ordre canonique)
    logger.info("   Score moyen par sous-niveau :")
    for sl in SUBLEVEL_ORDER:
        vals = part_df.loc[part_df["level9"] == sl, "score"].values
        if len(vals):
            logger.info(f"      {sl:7s} (rank {SUBLEVEL_RANK[sl]}) : "
                        f"{vals.mean():+.3f} (n={len(vals)})")

    # ---- Validite predictive : OLS score~year sur les middle ---------------
    mid = part_df[part_df["group4"].isin(["junior", "senior"])].copy()
    logger.info(f"\n[validite predictive] OLS score~year sur {len(mid)} middle "
                f"(juniors+seniors) — scores BRUTS :")

    condition = args.condition
    if condition is None:
        condition = pool_kw.get("pool_type", "gap")
        if pool_kw.get("pool_time_shuffle"):
            condition = f"{condition}_shuffle"
        # SWA : lu depuis train_config si présent
        if cfg_path.exists():
            tc_swa = json.loads(cfg_path.read_text())
            if tc_swa.get("swa"):
                condition = f"{condition}+SWA"

    metrics, long_df = compute_metrics_middle(
        part_df,
        seed=args.seed,
        condition=condition,
        agg=args.agg,
        bootstrap_B=args.bootstrap_B,
    )
    ols = metrics["ols_bootstrap"]
    sp = metrics["spearman"]
    logger.info(
        f"   [{args.agg}] pente={ols['slope']:+.4f}  IC95=[{ols['slope_ci95'][0]:+.4f}, "
        f"{ols['slope_ci95'][1]:+.4f}]  R2={ols['r2']:.4f} "
        f"IC_BCa=[{ols['r2_ci95'][0]:.4f}, {ols['r2_ci95'][1]:.4f}]  MAE={ols['mae']:.4f}"
    )
    logger.info(
        f"   Spearman ρ  junior={sp['junior']['rho']:+.3f}  "
        f"senior={sp['senior']['rho']:+.3f}  middle={sp['middle']['rho']:+.3f}"
    )

    metrics_path = run_dir / "metrics_middle.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    long_path = run_dir / "metrics_middle_long.csv"
    long_df.to_csv(long_path, index=False)
    logger.info(f"[save] {metrics_path.name} + {long_path.name}")

    # Sensibilité : autre agrégation (annexe), sans toucher au principal
    if args.agg_both:
        other = "median" if args.agg == "mean" else "mean"
        other_fn = _agg_fn(other)
        rows_o = []
        for p, sc in by_part.items():
            g4, lv9, yr = part_meta[p]
            rows_o.append({
                "participant": p, "group4": g4, "level9": lv9, "year": yr,
                "score": other_fn(sc), "n_trials": len(sc), "agg": other,
            })
        m_o, long_o = compute_metrics_middle(
            pd.DataFrame(rows_o), seed=args.seed, condition=condition,
            agg=other, bootstrap_B=args.bootstrap_B,
        )
        with open(run_dir / f"metrics_middle_{other}.json", "w", encoding="utf-8") as f:
            json.dump(m_o, f, indent=2)
        long_o.to_csv(run_dir / f"metrics_middle_long_{other}.csv", index=False)
        logger.info(f"[sensibilité] agrégation {other} sauvegardée (annexe)")

    slope = ols["slope"]
    intercept = ols["intercept"]
    r2 = ols["r2"]
    mse = ols["mse"]
    mae = ols["mae"]

    # ---- Barplot ------------------------------------------------------------
    barplot_path = run_dir / "panelE_barplot.png"
    make_barplot(part_df, tukey_df, anova_p, barplot_path, args.seed, agg=args.agg)
    logger.info(f"\n[figure] barplot panneau E -> {barplot_path}")

    # ---- Recap JSON ---------------------------------------------------------
    e_vs_s = tukey_df[
        tukey_df.apply(lambda r: {r["group1"], r["group2"]} == {"expert", "senior"}, axis=1)
    ].iloc[0]
    report = {
        "seed": args.seed,
        "max_len": max_len,
        "condition": condition,
        "agg_trial_to_participant": args.agg,
        "group_means_ci95": summary,
        "anova": {"F": float(f_stat), "p": float(anova_p)},
        "expert_vs_senior": {
            "diff": float(e_vs_s["meandiff"]),
            "ci95": [float(e_vs_s["lower"]), float(e_vs_s["upper"])],
            "p_adj": float(e_vs_s["p_adj"]),
            "significant": bool(e_vs_s["reject"]),
        },
        "spearman_9sublevels": {"rho": float(rho), "p": float(sp_p)},
        "spearman_middle_participant": sp,
        "predictive_validity_middle": {
            "n": int(len(mid)), "slope": slope, "intercept": intercept,
            "r2": r2, "mse": mse, "mae": mae,
            "slope_ci95": ols["slope_ci95"],
            "r2_ci95": ols["r2_ci95"],
            "mae_ci95": ols["mae_ci95"],
            "bootstrap_B": ols["bootstrap_B"],
            "bootstrap_level": "participant",
        },
    }
    with open(run_dir / "panelE_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"[save] rapport panneau E -> {run_dir / 'panelE_report.json'}")

    # ---- Gate : ordres de grandeur du papier -------------------------------
    logger.info("\n[GATE] Cible papier : Experts~0.75, Seniors~0.25, Juniors~-0.09, "
                "Novices~-0.80 ; Expert-vs-Senior significatif (~p=.045)")
    logger.info(f"[GATE] Obtenu     : Experts={summary['expert']['mean']:+.2f}, "
                f"Seniors={summary['senior']['mean']:+.2f}, "
                f"Juniors={summary['junior']['mean']:+.2f}, "
                f"Novices={summary['novice']['mean']:+.2f} ; "
                f"E-vs-S p={float(e_vs_s['p_adj']):.3g}")


if __name__ == "__main__":
    main()
