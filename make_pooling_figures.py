#!/usr/bin/env python3
"""
make_pooling_figures.py
=======================
Figures A0 vs A2 pour mémoire / SPIE (scores BRUTS).

  fig1 : violin + strip par groupe, panneaux A0 | A2, axes [-1, 1]
  fig2 : régression score↔année milieu, bandes IC bootstrap participant
  fig3 : heatmap temporelle de α (pooling weights — NOT an explanation)

Palette : palette_icems.py
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_hybrid1 import GROUP4_DISPLAY, GROUP4_ORDER
from metrics_middle import bootstrap_ols_ci, MIDDLE_GROUPS, ols_point
from palette_icems import PALETTE, GRID, INK, apply_style

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("make_pooling_figures")

# group4 keys → palette keys
_PAL_KEY = {
    "novice": "Novice",
    "junior": "Junior",
    "senior": "Senior",
    "expert": "Expert",
}


def _load_predictions(run_dir: Path, seed: int) -> pd.DataFrame:
    p = run_dir / f"seed{seed}" / "predictions_A_faithful.pkl"
    if not p.exists():
        raise FileNotFoundError(p)
    rows = pickle.loads(p.read_bytes())
    return pd.DataFrame(rows)


def _mean_predictions(run_dir: Path, seeds: list[int]) -> pd.DataFrame:
    """Moyenne inter-seeds du score participant (aligné sur participant id)."""
    frames = []
    for s in seeds:
        try:
            df = _load_predictions(run_dir, s)
        except FileNotFoundError:
            logger.warning(f"[skip] predictions manquantes seed {s} @ {run_dir}")
            continue
        df = df[["participant", "group4", "level9", "year", "score"]].copy()
        df["seed"] = s
        frames.append(df)
    if not frames:
        raise SystemExit(f"Aucune prediction dans {run_dir}")
    all_df = pd.concat(frames, ignore_index=True)
    agg = (
        all_df.groupby(["participant", "group4", "level9", "year"], as_index=False)
        .agg(score=("score", "mean"), n_seeds=("seed", "nunique"))
    )
    return agg


def fig1_violin_strip(
    df_a0: pd.DataFrame,
    df_a2: pd.DataFrame,
    out: Path,
    label_a0: str = "A0 (GAP)",
    label_a2: str = "A2 (Attention)",
) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    panels = [(axes[0], df_a0, label_a0), (axes[1], df_a2, label_a2)]
    plot_order = ["novice", "junior", "senior", "expert"]

    for ax, df, title in panels:
        data = [df.loc[df["group4"] == g, "score"].to_numpy() for g in plot_order]
        colors = [PALETTE[_PAL_KEY[g]] for g in plot_order]
        parts = ax.violinplot(
            data, positions=range(len(plot_order)), showmeans=False,
            showmedians=False, showextrema=False, widths=0.7,
        )
        for b, c in zip(parts["bodies"], colors):
            b.set_facecolor(c)
            b.set_alpha(0.35)
            b.set_edgecolor(c)
        rng = np.random.default_rng(0)
        for i, (g, c) in enumerate(zip(plot_order, colors)):
            vals = df.loc[df["group4"] == g, "score"].to_numpy()
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(i + jitter, vals, s=18, color=c, alpha=0.85,
                       edgecolors=INK, linewidths=0.3, zorder=3)
            if len(vals):
                ax.hlines(np.median(vals), i - 0.28, i + 0.28,
                          colors=INK, linewidths=1.6, zorder=4)
        ax.set_xticks(range(len(plot_order)))
        ax.set_xticklabels([GROUP4_DISPLAY[g] for g in plot_order])
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(title)
        ax.axhline(0, color=GRID, lw=0.8, ls="--")
        if ax is axes[0]:
            ax.set_ylabel("Score participant (brut)")

    fig.suptitle("Distribution des scores par groupe — A0 vs A2", y=1.02)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"[fig1] {out}")


def fig2_regression_middle(
    df_a0: pd.DataFrame,
    df_a2: pd.DataFrame,
    out: Path,
    label_a0: str = "A0 (GAP)",
    label_a2: str = "A2 (Attention)",
    bootstrap_B: int = 5000,
) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    panels = [(axes[0], df_a0, label_a0), (axes[1], df_a2, label_a2)]

    for ax, df, title in panels:
        mid = df[df["group4"].isin(MIDDLE_GROUPS)].copy()
        year = mid["year"].to_numpy(dtype=float)
        score = mid["score"].to_numpy(dtype=float)
        ols = bootstrap_ols_ci(year, score, B=bootstrap_B, seed=0)

        for g in MIDDLE_GROUPS:
            sub = mid[mid["group4"] == g]
            ax.scatter(
                sub["year"], sub["score"],
                s=36, color=PALETTE[_PAL_KEY[g]],
                edgecolors=INK, linewidths=0.4,
                label=GROUP4_DISPLAY[g], zorder=3,
            )

        x_line = np.linspace(year.min() - 0.2, year.max() + 0.2, 60)
        y_line = ols["intercept"] + ols["slope"] * x_line
        ax.plot(x_line, y_line, color=INK, lw=2.0, zorder=2)

        # bande IC bootstrap sur la droite (percentile des droites)
        rng = np.random.default_rng(0)
        n = len(score)
        ys = []
        for _ in range(min(bootstrap_B, 2000)):
            idx = rng.integers(0, n, size=n)
            if np.unique(year[idx]).size < 2:
                continue
            m = ols_point(year[idx], score[idx])
            ys.append(m["intercept"] + m["slope"] * x_line)
        if ys:
            Y = np.vstack(ys)
            lo, hi = np.percentile(Y, [2.5, 97.5], axis=0)
            ax.fill_between(x_line, lo, hi, color=PALETTE["Senior"], alpha=0.18, zorder=1)

        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Année de formation")
        ax.set_title(
            f"{title}\npente={ols['slope']:+.3f} "
            f"[{ols['slope_ci95'][0]:+.3f}, {ols['slope_ci95'][1]:+.3f}]"
        )
        if ax is axes[0]:
            ax.set_ylabel("Score participant (brut)")
            ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("Validité prédictive milieu — score ↔ année (IC bootstrap participant)", y=1.05)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"[fig2] {out}")


def fig3_alpha_heatmap(
    alpha_by_group: dict[str, np.ndarray],
    out: Path,
) -> None:
    """
    alpha_by_group[g] : (n_trials_or_parts, L) poids α moyens.
    Titre explicite : pooling weights — NOT an explanation (Jain & Wallace 2019).
    """
    apply_style()
    groups = [g for g in ["novice", "junior", "senior", "expert"] if g in alpha_by_group]
    if not groups:
        logger.warning("[fig3] aucun alpha fourni — skip")
        return

    fig, axes = plt.subplots(len(groups), 1, figsize=(9.0, 1.6 * len(groups)), sharex=True)
    if len(groups) == 1:
        axes = [axes]

    for ax, g in zip(axes, groups):
        A = np.asarray(alpha_by_group[g], dtype=float)
        mean_a = A.mean(axis=0)
        # normalise pour lisibilité visuelle (déjà Σα=1)
        im = ax.imshow(
            mean_a[None, :], aspect="auto", cmap="magma",
            vmin=0, vmax=max(mean_a.max(), 1e-6),
        )
        ax.set_yticks([0])
        ax.set_yticklabels([GROUP4_DISPLAY[g]])
        ax.set_ylabel("")
    axes[-1].set_xlabel("Timestep (fenêtre)")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="α moyen")
    fig.suptitle(
        "Pooling weights α — NOT an explanation (Jain & Wallace 2019)\n"
        "Causal claims remain on occlusion only",
        fontsize=11,
    )
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"[fig3] {out}")


def extract_alpha_by_group(
    run_dir: Path,
    seed: int,
    pkl: Path,
    max_trials_per_group: int = 12,
) -> dict[str, np.ndarray]:
    """Forward pass pour récupérer last_alpha (attn uniquement)."""
    pack = extract_alpha_and_activity(run_dir, seed, pkl, max_trials_per_group)
    return {g: v["alpha"] for g, v in pack.items() if "alpha" in v}


def extract_alpha_and_activity(
    run_dir: Path,
    seed: int,
    pkl: Path,
    max_trials_per_group: int = 40,
    vel_thresh_q: float = 0.50,
) -> dict[str, dict]:
    """
    Pour le contrôle de confond α↔activité :
      - alpha : (n, L)
      - activity_frac : fraction de frames « actives » (vitesse instruments)
      - alpha_on_active : masse d'attention sur frames actives
      - jerk_frac_high : fraction frames à fort jerk (proxy dynamique)

    Indices features Hybrid1 (PAIR_NAMES) :
      vel = 0,3,6 (bip_v, sci_v, cav_v) ; jerk = 2,5,8
    """
    import torch
    from data_hybrid1 import TrialDataset, build_trials, collate_pad, load_raw_trials
    from models_hybrid1 import Hybrid1Config, Hybrid1ModelA
    from torch.utils.data import DataLoader
    from scipy.stats import spearmanr

    seed_dir = run_dir / f"seed{seed}"
    tc = json.loads((seed_dir / "train_config.json").read_text())
    if tc.get("pool_type") != "attn":
        raise ValueError(f"requiert pool_type=attn (got {tc.get('pool_type')})")

    cfg = Hybrid1Config(
        n_features=10,
        max_len=tc.get("max_len", 1024),
        dropout=tc.get("dropout", 0.30),
        pool_type="attn",
        pool_heads=tc.get("pool_heads", 1),
        pool_d_attn=tc.get("pool_d_attn"),
        pool_tau=tc.get("pool_tau", 1.0),
        pool_att_dropout=tc.get("pool_att_dropout", 0.0),
        pool_time_shuffle=bool(tc.get("pool_time_shuffle", False)),
    )
    device = torch.device("cpu")
    model = Hybrid1ModelA(cfg).to(device)
    state = torch.load(seed_dir / "model_A_best.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    with open(seed_dir / "norm_stats.pkl", "rb") as f:
        ns = pickle.load(f)
    raw = load_raw_trials(str(pkl))
    trials = build_trials(raw, max_len=cfg.max_len)

    by_g: dict[str, list] = {g: [] for g in GROUP4_ORDER}
    for t in trials:
        if len(by_g[t.group4]) < max_trials_per_group:
            by_g[t.group4].append(t)

    VEL_IDX = [0, 3, 6]
    JERK_IDX = [2, 5, 8]
    pack: dict[str, dict] = {}

    with torch.no_grad():
        for g, ts in by_g.items():
            if not ts:
                continue
            alphas, act_fracs, a_on_act, jerk_fracs = [], [], [], []
            ds = TrialDataset(ts, ns["mean"], ns["std"], target="y_reg")
            loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_pad)
            t_i = 0
            for X, mask, _ in loader:
                _ = model(X, key_padding_mask=mask)
                alpha = model.attn_pool.last_alpha.mean(dim=-1).cpu().numpy()  # (B,L)
                B = X.shape[0]
                for b in range(B):
                    # features BRUTES du trial (non normalisées) pour l'activité
                    x_raw = ts[t_i].X  # (L_raw, 10)
                    L = alpha.shape[1]
                    x_use = x_raw[:L]
                    if len(x_use) < L:
                        # pad déjà dans le modèle ; tronquer alpha
                        a = alpha[b, : len(x_use)]
                        x_use = x_raw
                    else:
                        a = alpha[b]
                    vel = np.linalg.norm(x_use[:, VEL_IDX], axis=1)
                    jerk = np.linalg.norm(x_use[:, JERK_IDX], axis=1)
                    # seuil relatif au trial (médiane) — robuste cross-groupe
                    thr_v = float(np.quantile(vel, vel_thresh_q))
                    thr_j = float(np.quantile(jerk, 0.75))
                    active = vel >= thr_v
                    high_j = jerk >= thr_j
                    if a.sum() <= 0:
                        t_i += 1
                        continue
                    a_n = a / a.sum()
                    alphas.append(a_n)
                    act_fracs.append(float(active.mean()))
                    a_on_act.append(float(a_n[active].sum()) if active.any() else 0.0)
                    jerk_fracs.append(float(a_n[high_j].sum()) if high_j.any() else 0.0)
                    t_i += 1

            if not act_fracs:
                continue
            af = np.asarray(act_fracs)
            aa = np.asarray(a_on_act)
            jf = np.asarray(jerk_fracs)
            rho_act, p_act = spearmanr(af, aa)
            rho_jerk, p_jerk = spearmanr(af, jf)  # activité vs masse-sur-jerk
            # mieux : corréler alpha_on_active avec activity_frac
            pack[g] = {
                "alpha": np.stack(alphas),
                "activity_frac": af,
                "alpha_on_active": aa,
                "alpha_on_high_jerk": jf,
                "rho_alphaActive_vs_activityFrac": float(rho_act),
                "p_alphaActive_vs_activityFrac": float(p_act),
                "rho_alphaJerk_vs_activityFrac": float(rho_jerk),
                "p_alphaJerk_vs_activityFrac": float(p_jerk),
                "n": int(len(af)),
            }
    return pack


def fig4_confound_alpha_activity(
    pack: dict[str, dict],
    out: Path,
) -> dict:
    """
    Contrôle de confond : si Σα sur frames actives suit surtout la fraction
    d'activité brute → drapeau rouge (redécouverte du confond d'activité).
    Si α se concentre sur la dynamique (jerk) indépendamment de la quantité
    d'activité → mécanisme plus propre.
    """
    apply_style()
    rows = []
    for g, d in pack.items():
        rows.append({
            "group": g,
            "n": d["n"],
            "rho_act": d["rho_alphaActive_vs_activityFrac"],
            "p_act": d["p_alphaActive_vs_activityFrac"],
            "rho_jerk": d["rho_alphaJerk_vs_activityFrac"],
            "p_jerk": d["p_alphaJerk_vs_activityFrac"],
        })
    if not rows:
        logger.warning("[fig4] aucun groupe — skip")
        return {}

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    # panneau gauche : scatter global
    ax = axes[0]
    for g, d in pack.items():
        c = PALETTE[_PAL_KEY[g]]
        ax.scatter(
            d["activity_frac"], d["alpha_on_active"],
            s=28, color=c, alpha=0.75, edgecolors=INK, linewidths=0.3,
            label=GROUP4_DISPLAY[g],
        )
    ax.set_xlabel("Fraction frames actives (activité instrumentale)")
    ax.set_ylabel("Masse d'attention Σα sur frames actives")
    ax.set_title("Confond check : α suit-il l'activité brute ?")
    ax.legend(fontsize=8, loc="best")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # panneau droit : ρ par groupe
    ax = axes[1]
    groups = [r["group"] for r in rows]
    x = np.arange(len(groups))
    w = 0.35
    ax.bar(x - w / 2, [r["rho_act"] for r in rows], w,
           color=[PALETTE[_PAL_KEY[g]] for g in groups], label="ρ(α_active, act_frac)")
    ax.bar(x + w / 2, [r["rho_jerk"] for r in rows], w,
           color=[PALETTE[_PAL_KEY[g]] for g in groups], alpha=0.45,
           label="ρ(α_jerk, act_frac)")
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP4_DISPLAY[g] for g in groups])
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Par groupe")
    ax.legend(fontsize=8)
    ax.set_ylim(-1, 1)

    # drapeau
    rhos = [r["rho_act"] for r in rows if np.isfinite(r["rho_act"])]
    flag = "RED_FLAG" if (rhos and float(np.nanmean(rhos)) > 0.5) else "OK_or_weak"
    fig.suptitle(
        f"Pooling weights vs activity confound — {flag}\n"
        "(NOT an explanation ; Jain & Wallace 2019)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"[fig4] {out}  flag={flag}")

    summary = {"flag": flag, "per_group": rows}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Figures pooling A0 vs A2")
    ap.add_argument("--a0", type=Path, required=True)
    ap.add_argument("--a2", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2024])
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--label-a0", default="A0 (GAP)")
    ap.add_argument("--label-a2", default="A2 (Attention)")
    ap.add_argument("--bootstrap-B", type=int, default=5000)
    ap.add_argument("--pkl", type=Path, default=Path("data/trial_tensor_v2.pkl"),
                    help="requis pour fig3/fig4 (alpha)")
    ap.add_argument("--fig3-seed", type=int, default=42)
    ap.add_argument("--skip-fig3", action="store_true")
    ap.add_argument("--skip-fig4", action="store_true",
                    help="skip contrôle confond α↔activité")
    args = ap.parse_args()

    out_dir = args.out_dir or (args.a2 / "figures_pooling")
    out_dir.mkdir(parents=True, exist_ok=True)

    df0 = _mean_predictions(args.a0, args.seeds)
    df2 = _mean_predictions(args.a2, args.seeds)

    fig1_violin_strip(df0, df2, out_dir / "fig1_violin_strip_A0_A2.png",
                      args.label_a0, args.label_a2)
    fig2_regression_middle(df0, df2, out_dir / "fig2_regression_middle_A0_A2.png",
                           args.label_a0, args.label_a2, args.bootstrap_B)

    pack = None
    if not args.skip_fig3 or not args.skip_fig4:
        try:
            pack = extract_alpha_and_activity(args.a2, args.fig3_seed, args.pkl)
        except Exception as e:
            logger.warning(f"[alpha] skip extract ({e})")

    if not args.skip_fig3 and pack is not None:
        try:
            alphas = {g: v["alpha"] for g, v in pack.items()}
            fig3_alpha_heatmap(alphas, out_dir / "fig3_alpha_pooling_weights.png")
        except Exception as e:
            logger.warning(f"[fig3] skip ({e})")

    if not args.skip_fig4 and pack is not None:
        try:
            fig4_confound_alpha_activity(
                pack, out_dir / "fig4_confound_alpha_activity.png",
            )
        except Exception as e:
            logger.warning(f"[fig4] skip ({e})")

    logger.info(f"[done] figures -> {out_dir}")


if __name__ == "__main__":
    main()
