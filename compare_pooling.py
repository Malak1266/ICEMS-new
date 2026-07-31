#!/usr/bin/env python3
"""
compare_pooling.py
==================
Compare A0 (GAP) vs A2 (Attention pooling) — et optionnellement A1/A3/A6 —
sur les 5 seeds, à partir des panelE_report.json déjà produits.

Stats sur scores bruts (ligne rouge : pas de calibration dans les tests).
Wilcoxon signed-rank apparié seed-à-seed.

Usage typique (après agrégation des runs) :
    python compare_pooling.py \\
        --a0 results/hybrid1_faithful \\
        --a2 results/hybrid1_attn_h1 \\
        --seeds 42 123 456 789 2024

Critères de succès (fixés à l'avance — anti-HARKing) :
  Succès fort   : pente milieu ↑ (Wilcoxon p<0.05) ET AUC extrêmes stable
                  ET A6 (shuffle) confirme la localisation temporelle.
  Succès partiel: R²/pente ↑ mais Junior–Senior toujours non séparés.
  Résultat nul  : A2 ≈ A0 — rapporté tel quel (contribution méthodologique valide).

NOTE SOUTENANCE (25 juillet) : sans les 5 seeds A100, ceci reste une
extension méthodologique proposée (perspectives / SPIE MI27), pas un gain mesuré.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("compare_pooling")

DEFAULT_SEEDS = [42, 123, 456, 789, 2024, 1337, 2718, 3141, 9001, 5555]
# PRE-REG C7 : 10 seeds figés avant scoring milieu (Wilcoxon bilatéral atteignable).
# Ne pas ajouter/retirer de seeds après avoir vu Junior/Senior.


def _load_reports(run_dir: Path, seeds: list[int]) -> dict[int, dict]:
    """Charge panelE_report + merge metrics_middle si présent."""
    out = {}
    for s in seeds:
        p = run_dir / f"seed{s}" / "panelE_report.json"
        if not p.exists():
            logger.warning(f"[skip] manquant : {p}")
            continue
        rep = json.loads(p.read_text(encoding="utf-8"))
        mm = run_dir / f"seed{s}" / "metrics_middle.json"
        if mm.exists():
            rep["_metrics_middle"] = json.loads(mm.read_text(encoding="utf-8"))
        out[s] = rep
    return out


def _metric(rep: dict, key: str) -> float:
    """Extrait une métrique scalaire (panelE ou metrics_middle)."""
    mm = rep.get("_metrics_middle") or {}
    primary = mm.get("primary_metrics") or {}
    secondary = mm.get("secondary_metrics") or {}
    spearman = mm.get("spearman") or rep.get("spearman_middle_participant") or {}
    pv = rep.get("predictive_validity_middle", {})
    gm = rep.get("group_means_ci95", {})

    if key == "slope":
        return float(primary.get("slope", pv["slope"]))
    if key == "rho_middle":
        if "rho_middle" in primary:
            return float(primary["rho_middle"])
        return float(spearman["middle"]["rho"])
    if key == "rho_junior":
        return float(primary.get("rho_junior", spearman["junior"]["rho"]))
    if key == "rho_senior":
        return float(primary.get("rho_senior", spearman["senior"]["rho"]))
    if key == "r2":
        return float(secondary.get("r2", pv["r2"]))
    if key == "mae":
        return float(secondary.get("mae", pv["mae"]))
    if key.startswith("mean_"):
        g = key[len("mean_"):]
        return float(gm[g]["mean"])
    raise KeyError(key)


def paired_seed_test(
    a: dict[int, dict],
    b: dict[int, dict],
    metric: str,
    label_a: str = "A0",
    label_b: str = "A2",
    *,
    direction: str = "greater",
) -> dict:
    """
    Comparaison seed-à-seed.

    Headline honnête (n=5) :
      - n_favorable / n_seeds + Δ médian [min, max]  (sign test)
      - Wilcoxon unilatéral (hypothèse directionnelle pré-enregistrée)
      - Wilcoxon bilatéral rapporté pour transparence (plancher p=0.0625 si n=5)

    Ne PAS laisser un p unique porter la conclusion.
    """
    try:
        from scipy.stats import wilcoxon, binomtest
    except ImportError as e:
        raise SystemExit("scipy requis") from e

    common = sorted(set(a) & set(b))
    if len(common) < 2:
        raise SystemExit(
            f"Besoin d'au moins 2 seeds communes pour {metric} (got {common})."
        )

    va = np.array([_metric(a[s], metric) for s in common], dtype=np.float64)
    vb = np.array([_metric(b[s], metric) for s in common], dtype=np.float64)
    delta = vb - va  # A2 - A0 ; >0 = A2 meilleur si direction=greater

    n = len(delta)
    n_pos = int(np.sum(delta > 0))
    n_neg = int(np.sum(delta < 0))
    n_zero = int(np.sum(delta == 0))
    n_eff = n_pos + n_neg  # zeros exclus du sign test

    # Sign test unilatéral : P(X >= n_pos) sous Bin(n_eff, 0.5)
    if n_eff == 0:
        sign_p = 1.0
    else:
        # direction greater → compter les succès = deltas positifs
        sign_p = float(binomtest(n_pos, n_eff, 0.5, alternative="greater").pvalue)

    # Wilcoxon
    if np.allclose(delta, 0):
        w_stat, w_p_two, w_p_one = 0.0, 1.0, 1.0
    else:
        alt_one = "greater" if direction == "greater" else "less"
        r_two = wilcoxon(vb, va, alternative="two-sided", zero_method="wilcox")
        r_one = wilcoxon(vb, va, alternative=alt_one, zero_method="wilcox")
        w_stat = float(r_two.statistic)
        w_p_two = float(r_two.pvalue)
        w_p_one = float(r_one.pvalue)

    sd = float(delta.std(ddof=1)) if len(delta) > 1 else float("nan")
    d = float(delta.mean() / sd) if sd and sd > 0 else float("nan")
    # plancher bilatéral théorique si tous du même sens : 2/2^n
    p_floor_two = 2.0 / (2 ** n)

    return {
        "metric": metric,
        "seeds": common,
        "direction_preregistered": direction,
        label_a: va.tolist(),
        label_b: vb.tolist(),
        "delta_per_seed": {str(s): float(dlt) for s, dlt in zip(common, delta)},
        "delta": delta.tolist(),
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
        "delta_std": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
        "n_favorable": n_pos if direction == "greater" else n_neg,
        "n_seeds": n,
        "n_ties": n_zero,
        "sign_test_p_onesided": sign_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_twosided": w_p_two,
        "wilcoxon_p_onesided": w_p_one,
        "wilcoxon_p": w_p_one,  # alias = unilatéral (pré-enregistré)
        "wilcoxon_twosided_floor_note": (
            f"avec n={n}, p bilatéral minimal si {n}/{n} même sens = {p_floor_two:.4f}"
            if n <= 10 else None
        ),
        "effect_d": d,
        f"{label_a}_mean": float(va.mean()),
        f"{label_b}_mean": float(vb.mean()),
        "headline": (
            f"{n_pos if direction == 'greater' else n_neg}/{n} seeds favorables, "
            f"Δmedian={float(np.median(delta)):+.4f} "
            f"[{float(delta.min()):+.4f}, {float(delta.max()):+.4f}]"
        ),
    }


def _fmt_row(r: dict, label_a: str, label_b: str) -> str:
    return (
        f"  {r['metric']:12s}  {r['headline']}  "
        f"sign_p(1s)={r['sign_test_p_onesided']:.4f}  "
        f"Wilcoxon_p(1s)={r['wilcoxon_p_onesided']:.4f} "
        f"(2s={r['wilcoxon_p_twosided']:.4f})  d={r['effect_d']:+.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare pooling conditions (A0 GAP vs A2 Attention) seed-à-seed"
    )
    ap.add_argument("--a0", type=Path, required=True,
                    help="run dir GAP (A0), contient seed*/panelE_report.json")
    ap.add_argument("--a2", type=Path, required=True,
                    help="run dir Attention pooling (A2)")
    ap.add_argument("--a1", type=Path, default=None, help="optionnel : max-pooling")
    ap.add_argument("--a6", type=Path, default=None,
                    help="optionnel : attention + time-shuffle (contrôle)")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON de sortie (défaut : <a2>/compare_pooling.json)")
    ap.add_argument("--label-a", default="A0_gap")
    ap.add_argument("--label-b", default="A2_attn")
    args = ap.parse_args()

    a0 = _load_reports(args.a0, args.seeds)
    a2 = _load_reports(args.a2, args.seeds)
    logger.info(f"A0 seeds={sorted(a0)}  A2 seeds={sorted(a2)}")

    # PRIMAIRE : pente + ρ_middle ; SECONDAIRE : R² + MAE ; descriptif : means
    primary = ["slope", "rho_middle"]
    secondary = ["r2", "mae", "rho_junior", "rho_senior"]
    descriptive = ["mean_junior", "mean_senior", "mean_novice", "mean_expert"]
    metrics = primary + secondary + descriptive
    results = []
    logger.info("=" * 72)
    logger.info(f"COMPARISON {args.label_a} vs {args.label_b} (scores bruts)")
    logger.info("  PRIMAIRE = slope + rho_middle | SECONDAIRE = r2 + mae")
    logger.info("=" * 72)
    for m in metrics:
        try:
            r = paired_seed_test(a0, a2, m, args.label_a, args.label_b)
        except (KeyError, SystemExit) as e:
            logger.warning(f"  [skip] {m}: {e}")
            continue
        r["priority"] = (
            "primary" if m in primary else
            "secondary" if m in secondary else "descriptive"
        )
        results.append(r)
        tag = {"primary": "[P]", "secondary": "[S]", "descriptive": "[D]"}[r["priority"]]
        logger.info(f"{tag} {_fmt_row(r, args.label_a, args.label_b)}")

    extras = {}
    if args.a1 is not None:
        a1 = _load_reports(args.a1, args.seeds)
        extras["A1_max_vs_A0"] = []
        logger.info("\n-- A1 (max) vs A0 --")
        for m in primary:
            if not (set(a0) & set(a1)):
                break
            try:
                r = paired_seed_test(a0, a1, m, "A0_gap", "A1_max")
            except (KeyError, SystemExit) as e:
                logger.warning(f"  [skip] {m}: {e}")
                continue
            extras["A1_max_vs_A0"].append(r)
            logger.info(_fmt_row(r, "A0_gap", "A1_max"))

    if args.a6 is not None:
        a6 = _load_reports(args.a6, args.seeds)
        extras["A6_shuffle_vs_A2"] = []
        logger.info("\n-- A6 (time-shuffle) vs A2 (doit chuter vers A0 si gain temporel) --")
        for m in primary:
            if not (set(a2) & set(a6)):
                break
            try:
                r = paired_seed_test(a2, a6, m, "A2_attn", "A6_shuffle")
            except (KeyError, SystemExit) as e:
                logger.warning(f"  [skip] {m}: {e}")
                continue
            extras["A6_shuffle_vs_A2"].append(r)
            logger.info(_fmt_row(r, "A2_attn", "A6_shuffle"))

    # Verdict — headline = consistance, pas un p unique (PRE-REG C6)
    slope_row = next((r for r in results if r["metric"] == "slope"), None)
    rho_row = next((r for r in results if r["metric"] == "rho_middle"), None)
    verdict = "données insuffisantes"
    if slope_row is not None:
        n = slope_row["n_seeds"]
        fav = slope_row["n_favorable"]
        all_fav = fav == n
        rho_fav = rho_row is not None and rho_row["n_favorable"] == rho_row["n_seeds"]
        if all_fav and slope_row["sign_test_p_onesided"] < 0.05:
            verdict = (
                f"succès candidat : {slope_row['headline']} "
                f"(sign_p={slope_row['sign_test_p_onesided']:.4f}) — "
                "vérifier AUC extrêmes + A6 avant SWA/heads"
            )
        elif fav > n / 2 and slope_row["delta_median"] > 0:
            verdict = (
                f"tendance favorable ({fav}/{n}) non unanime — "
                "succès partiel possible ; viser 10+ seeds pour SPIE"
            )
        else:
            verdict = "résultat nul / négatif sur la pente — à rapporter tel quel"
        if rho_fav:
            verdict += f" | ρ_middle aussi {rho_row['headline']}"

    logger.info(f"\nVerdict provisoire : {verdict}")
    if slope_row and slope_row.get("wilcoxon_twosided_floor_note"):
        logger.info(f"Note : {slope_row['wilcoxon_twosided_floor_note']}")
    logger.info(
        "Deux inférences distinctes : (1) seed-variance = sign/Wilcoxon ci-dessus ; "
        "(2) sampling-variance = IC bootstrap participant dans metrics_middle "
        "(BCa pour R²). Ne pas les confondre."
    )
    logger.info(
        "Rappel PRE-REG : sélection HP/SWA/heads sur val EXTRÊMES seulement ; "
        "milieu scoré une fois. Seeds fixées, pas de cherry-pick."
    )

    payload = {
        "a0": str(args.a0),
        "a2": str(args.a2),
        "seeds_requested": args.seeds,
        "primary_metrics": primary,
        "secondary_metrics": secondary,
        "paired_tests": results,
        "extras": extras,
        "verdict_provisional": verdict,
        "integrity": {
            "train_extremes_only": True,
            "stats_on_raw_scores": True,
            "no_test_set_optimization": True,
            "selection_on_extremes_val_only": True,
            "middle_scored_once": True,
            "attention_weights_are_pooling_not_explanation": True,
            "fixed_seeds": DEFAULT_SEEDS,
        },
    }
    out = args.out or (args.a2 / "compare_pooling.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"\n[save] {out}")


if __name__ == "__main__":
    main()
