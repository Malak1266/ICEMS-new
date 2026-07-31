#!/usr/bin/env python3
"""
compare_architectures.py
========================
Comparaison statistiquement défendable de deux architectures évaluées en LOPO
nested (cf. step_B_classification.run_corrected_lopo), à protocole identique.

Conçu pour la grille produite par scripts/narval/run_hybrid_vs_gru.sbatch :

    results/arch_compare/
    ├── seed42/gru/lopo_predictions.pkl
    ├── seed42/hybrid/lopo_predictions.pkl
    ├── seed43/gru/lopo_predictions.pkl
    ├── seed43/hybrid/lopo_predictions.pkl
    └── ...

Ce que le script calcule (et POURQUOI) :

  1. r_participant et MAE par seed, puis moyenne ± écart-type.
     → répond à la critique « un seul seed = conclusion fragile ».

  2. Test de Wilcoxon APPARIÉ sur l'erreur absolue par participant.
     → comme les deux bras partagent le MÊME seed, ils partagent les mêmes
       folds LOPO : les prédictions par participant sont appariées. On teste
       si l'erreur du Hybrid est significativement < celle du GRU.
       (Wilcoxon = non-paramétrique, robuste au petit n et aux outliers.)

  3. Wilcoxon apparié sur le r par fold (r_per_fold).

  4. Intervalle de confiance bootstrap (participant-level) sur Δr_participant.
     → barre d'erreur honnête sur le gain réel.

Usage :
    python compare_architectures.py --root results/arch_compare
    python compare_architectures.py --root results/arch_compare --baseline gru --challenger hybrid
    python compare_architectures.py \
        --gru results/.../gru/lopo_predictions.pkl \
        --hybrid results/.../hybrid/lopo_predictions.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

PRED_FILE = "lopo_predictions.pkl"


# ── Chargement ───────────────────────────────────────────────────────────────
def load_preds_df(pkl_path: Path) -> pd.DataFrame:
    """Charge le preds_df TEST-only d'un run run_corrected_lopo."""
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "preds_df" in payload:
        df = payload["preds_df"]
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy()
    # Fallback : reconstruire depuis la liste `preds` (clé legacy).
    if isinstance(payload, dict) and "preds" in payload:
        rows = []
        for p in payload["preds"]:
            rows.append({
                "participant": str(p.get("participant")),
                "true_score": float(p.get("y_reg", p.get("y4_reg"))),
                "pred_score": float(p["score"]),
            })
        return pd.DataFrame(rows)
    raise ValueError(f"Format inattendu dans {pkl_path}")


def per_participant(df: pd.DataFrame) -> pd.DataFrame:
    """Agrège true/pred par participant (médiane des trials = robuste)."""
    return (
        df.groupby("participant", as_index=False)
        .agg(true_score=("true_score", "median"),
             pred_score=("pred_score", "median"))
        .sort_values("participant")
        .reset_index(drop=True)
    )


def r_participant(df: pd.DataFrame) -> float:
    g = per_participant(df)
    if len(g) < 2 or g["true_score"].std() < 1e-9 or g["pred_score"].std() < 1e-9:
        return float("nan")
    return float(pearsonr(g["true_score"], g["pred_score"])[0])


def mae_participant(df: pd.DataFrame) -> float:
    g = per_participant(df)
    return float(np.mean(np.abs(g["true_score"] - g["pred_score"])))


# ── Découverte de la grille ──────────────────────────────────────────────────
def discover_runs(root: Path, models: Tuple[str, str]) -> Dict[str, Dict[int, Path]]:
    """{model: {seed: pkl_path}} à partir de root/seed<S>/<model>/lopo_predictions.pkl."""
    found: Dict[str, Dict[int, Path]] = {m: {} for m in models}
    for seed_dir in sorted(root.glob("seed*")):
        if not seed_dir.is_dir():
            continue
        try:
            seed = int(seed_dir.name.replace("seed", ""))
        except ValueError:
            continue
        for m in models:
            pkl = seed_dir / m / PRED_FILE
            if pkl.exists():
                found[m][seed] = pkl
    return found


# ── Statistiques ─────────────────────────────────────────────────────────────
def bootstrap_delta_r(
    base: pd.DataFrame,
    chal: pd.DataFrame,
    n_boot: int = 5000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """IC bootstrap (participant-level) sur Δr = r(chal) - r(base).

    Ré-échantillonne les participants APPARIÉS (mêmes folds) avec remise.
    """
    gb = per_participant(base)
    gc = per_participant(chal)
    merged = gb.merge(gc, on="participant", suffixes=("_b", "_c"))
    if merged.empty or len(merged) < 3:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(merged)
    deltas = []
    tb = merged["true_score_b"].to_numpy()
    pb = merged["pred_score_b"].to_numpy()
    pc = merged["pred_score_c"].to_numpy()
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if tb[idx].std() < 1e-9 or pb[idx].std() < 1e-9 or pc[idx].std() < 1e-9:
            continue
        rb = pearsonr(tb[idx], pb[idx])[0]
        rc = pearsonr(tb[idx], pc[idx])[0]
        deltas.append(rc - rb)
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    deltas = np.asarray(deltas)
    return float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def paired_abs_error_test(base: pd.DataFrame, chal: pd.DataFrame) -> Dict:
    """Wilcoxon apparié sur |erreur| par participant (challenger vs baseline)."""
    gb = per_participant(base).assign(err=lambda d: np.abs(d.true_score - d.pred_score))
    gc = per_participant(chal).assign(err=lambda d: np.abs(d.true_score - d.pred_score))
    merged = gb[["participant", "err"]].merge(
        gc[["participant", "err"]], on="participant", suffixes=("_base", "_chal"),
    )
    if len(merged) < 5:
        return {"n": len(merged), "p_value": float("nan"), "median_delta": float("nan")}
    e_base = merged["err_base"].to_numpy()
    e_chal = merged["err_chal"].to_numpy()
    delta = e_chal - e_base  # < 0 ⇒ challenger meilleur
    try:
        stat, p = wilcoxon(e_chal, e_base, zero_method="wilcox", alternative="less")
    except ValueError:
        stat, p = float("nan"), float("nan")
    return {
        "n": int(len(merged)),
        "wilcoxon_stat": float(stat),
        "p_value": float(p),
        "median_abs_err_base": float(np.median(e_base)),
        "median_abs_err_chal": float(np.median(e_chal)),
        "median_delta": float(np.median(delta)),
        "n_participants_improved": int(np.sum(delta < 0)),
    }


def pooled_fold_r_test(
    base_runs: Dict[int, pd.DataFrame],
    chal_runs: Dict[int, pd.DataFrame],
) -> Dict:
    """Wilcoxon apparié sur le r par fold (un r par participant tenu out), poolé sur seeds."""
    base_r: List[float] = []
    chal_r: List[float] = []
    for seed in sorted(set(base_runs) & set(chal_runs)):
        b = base_runs[seed]
        c = chal_runs[seed]
        for pid in sorted(set(b.participant) & set(c.participant)):
            bb = b[b.participant == pid]
            cc = c[c.participant == pid]
            if len(bb) >= 2 and bb.true_score.std() > 1e-9:
                rb = pearsonr(bb.true_score, bb.pred_score)[0]
                rc = pearsonr(cc.true_score, cc.pred_score)[0]
                if np.isfinite(rb) and np.isfinite(rc):
                    base_r.append(rb)
                    chal_r.append(rc)
    if len(base_r) < 5:
        return {"n": len(base_r), "p_value": float("nan")}
    try:
        stat, p = wilcoxon(chal_r, base_r, alternative="greater")
    except ValueError:
        stat, p = float("nan"), float("nan")
    return {
        "n": len(base_r),
        "wilcoxon_stat": float(stat),
        "p_value": float(p),
        "mean_r_base": float(np.mean(base_r)),
        "mean_r_chal": float(np.mean(chal_r)),
    }


# ── Rapport ──────────────────────────────────────────────────────────────────
def summarize_model(runs: Dict[int, pd.DataFrame]) -> Dict:
    rs = {s: r_participant(df) for s, df in runs.items()}
    maes = {s: mae_participant(df) for s, df in runs.items()}
    r_vals = [v for v in rs.values() if np.isfinite(v)]
    mae_vals = [v for v in maes.values() if np.isfinite(v)]
    return {
        "seeds": sorted(runs.keys()),
        "r_participant_per_seed": {int(s): rs[s] for s in rs},
        "mae_participant_per_seed": {int(s): maes[s] for s in maes},
        "r_participant_mean": float(np.mean(r_vals)) if r_vals else float("nan"),
        "r_participant_std": float(np.std(r_vals)) if r_vals else float("nan"),
        "mae_participant_mean": float(np.mean(mae_vals)) if mae_vals else float("nan"),
        "mae_participant_std": float(np.std(mae_vals)) if mae_vals else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("results/arch_compare"),
                    help="Racine de la grille seed*/<model>/lopo_predictions.pkl")
    ap.add_argument("--baseline", type=str, default="gru")
    ap.add_argument("--challenger", type=str, default="hybrid")
    ap.add_argument("--gru", type=Path, default=None,
                    help="Chemin direct vers un lopo_predictions.pkl baseline (mode 1 seed).")
    ap.add_argument("--hybrid", type=Path, default=None,
                    help="Chemin direct vers un lopo_predictions.pkl challenger (mode 1 seed).")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON de sortie (défaut : <root>/architecture_comparison.json)")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    base_name, chal_name = args.baseline, args.challenger

    # Mode direct (1 seed) ou découverte de grille.
    if args.gru is not None and args.hybrid is not None:
        base_runs = {0: load_preds_df(args.gru)}
        chal_runs = {0: load_preds_df(args.hybrid)}
    else:
        found = discover_runs(args.root, (base_name, chal_name))
        base_runs = {s: load_preds_df(p) for s, p in found[base_name].items()}
        chal_runs = {s: load_preds_df(p) for s, p in found[chal_name].items()}

    if not base_runs or not chal_runs:
        print(f"❌ Aucun run trouvé. Vérifiez {args.root}/seed*/{{{base_name},{chal_name}}}/{PRED_FILE}")
        sys.exit(1)

    base_summary = summarize_model(base_runs)
    chal_summary = summarize_model(chal_runs)

    # Tests appariés sur les seeds communs.
    common_seeds = sorted(set(base_runs) & set(chal_runs))
    per_seed_tests = {}
    for s in common_seeds:
        per_seed_tests[int(s)] = paired_abs_error_test(base_runs[s], chal_runs[s])

    # Pool de tous les participants × seeds appariés pour le test global d'erreur.
    base_all = pd.concat(
        [base_runs[s].assign(_seed=s) for s in common_seeds], ignore_index=True,
    )
    chal_all = pd.concat(
        [chal_runs[s].assign(_seed=s) for s in common_seeds], ignore_index=True,
    )
    base_all["participant"] = base_all["_seed"].astype(str) + ":" + base_all["participant"].astype(str)
    chal_all["participant"] = chal_all["_seed"].astype(str) + ":" + chal_all["participant"].astype(str)
    pooled_err_test = paired_abs_error_test(base_all, chal_all)
    fold_r_test = pooled_fold_r_test(base_runs, chal_runs)

    # Bootstrap Δr sur le 1er seed commun (splits identiques garantis).
    if common_seeds:
        s0 = common_seeds[0]
        d_mean, d_lo, d_hi = bootstrap_delta_r(
            base_runs[s0], chal_runs[s0], n_boot=args.n_boot,
        )
    else:
        d_mean = d_lo = d_hi = float("nan")

    report = {
        "baseline": base_name,
        "challenger": chal_name,
        "common_seeds": [int(s) for s in common_seeds],
        f"{base_name}_summary": base_summary,
        f"{chal_name}_summary": chal_summary,
        "paired_abs_error_test_per_seed": per_seed_tests,
        "paired_abs_error_test_pooled": pooled_err_test,
        "fold_r_test": fold_r_test,
        "delta_r_participant_bootstrap": {
            "mean": d_mean, "ci95_low": d_lo, "ci95_high": d_hi,
        },
    }

    # ── Affichage ──
    print("=" * 64)
    print(f" COMPARAISON D'ARCHITECTURES  —  {base_name}  vs  {chal_name}")
    print("=" * 64)
    print(f" Seeds communs : {report['common_seeds']}")
    print("\n— r_participant (moyenne ± std sur seeds) —")
    print(f"  {base_name:>8} : {base_summary['r_participant_mean']:+.4f} "
          f"± {base_summary['r_participant_std']:.4f}")
    print(f"  {chal_name:>8} : {chal_summary['r_participant_mean']:+.4f} "
          f"± {chal_summary['r_participant_std']:.4f}")
    print("\n— MAE_participant (moyenne ± std) —")
    print(f"  {base_name:>8} : {base_summary['mae_participant_mean']:.4f} "
          f"± {base_summary['mae_participant_std']:.4f}")
    print(f"  {chal_name:>8} : {chal_summary['mae_participant_mean']:.4f} "
          f"± {chal_summary['mae_participant_std']:.4f}")

    print("\n— Wilcoxon apparié sur |erreur| par participant (poolé) —")
    pe = pooled_err_test
    print(f"  n={pe['n']}  médiane |err| {base_name}={pe.get('median_abs_err_base', float('nan')):.4f} "
          f"→ {chal_name}={pe.get('median_abs_err_chal', float('nan')):.4f}")
    print(f"  p-value (H1: {chal_name} < {base_name}) = {pe['p_value']:.4g}  "
          f"| participants améliorés : {pe.get('n_participants_improved', '?')}")

    print("\n— Wilcoxon apparié sur r par fold —")
    fr = fold_r_test
    print(f"  n={fr['n']}  mean r {base_name}={fr.get('mean_r_base', float('nan')):+.4f} "
          f"→ {chal_name}={fr.get('mean_r_chal', float('nan')):+.4f}  p={fr['p_value']:.4g}")

    print("\n— Δr_participant (bootstrap, 1er seed) —")
    print(f"  Δr = {d_mean:+.4f}  IC95 = [{d_lo:+.4f}, {d_hi:+.4f}]")
    significatif = np.isfinite(d_lo) and (d_lo > 0 or d_hi < 0)
    print(f"  IC95 exclut 0 : {'OUI (gain significatif)' if significatif else 'NON'}")

    # Verdict synthétique.
    print("\n" + "=" * 64)
    better_r = chal_summary["r_participant_mean"] > base_summary["r_participant_mean"]
    sig_err = np.isfinite(pe["p_value"]) and pe["p_value"] < 0.05
    if better_r and sig_err:
        print(f" VERDICT : {chal_name} surpasse {base_name} (r↑ ET erreur↓ significative).")
    elif better_r:
        print(f" VERDICT : {chal_name} a un r supérieur mais sans significativité d'erreur claire.")
    else:
        print(f" VERDICT : pas d'avantage net de {chal_name} — conserver {base_name} (parcimonie).")
    print("=" * 64)

    out_path = args.out or (args.root / "architecture_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[Sauvegarde] {out_path.resolve()}")


if __name__ == "__main__":
    main()
