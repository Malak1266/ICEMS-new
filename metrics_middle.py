"""
metrics_middle.py
=================
Métriques milieu (Junior+Senior) sur scores BRUTS participant-level.

PRIMAIRE : pente OLS + Spearman ρ_middle
SECONDAIRE : R² (IC **BCa**), MAE (IC percentile)

Agrégation essai→participant : `mean` (principal SPIE / papier) ou `median`
(sensibilité). A0 et A2 doivent toujours partager la même.

Deux inférences distinctes (ne pas confondre dans le texte) :
  - sampling-variance (n=25) = IC bootstrap participant
  - seed-variance = sign-test / Wilcoxon dans compare_pooling.py
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

MIDDLE_GROUPS = ("junior", "senior")
AggMode = Literal["mean", "median"]


def aggregate_trials_to_participants(
    trial_rows: list[dict] | pd.DataFrame,
    *,
    agg: AggMode = "mean",
    score_key: str = "score",
) -> pd.DataFrame:
    """
    Essai → participant. `trial_rows` : participant, group4, level9, year, score.
    Si plusieurs lignes/participant = essais ; sinon déjà participant-level.
    """
    df = pd.DataFrame(trial_rows)
    need = {"participant", "group4", "year", score_key}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"colonnes manquantes: {missing}")

    def _agg(s: pd.Series) -> float:
        return float(s.mean() if agg == "mean" else s.median())

    rows = []
    for pid, sub in df.groupby("participant", sort=False):
        rows.append({
            "participant": pid,
            "group4": sub["group4"].iloc[0],
            "level9": sub["level9"].iloc[0] if "level9" in sub.columns else None,
            "year": float(sub["year"].iloc[0]),
            "score": _agg(sub[score_key]),
            "n_trials": int(len(sub)),
            "agg": agg,
        })
    return pd.DataFrame(rows)


def _spearman_safe(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def spearman_by_group(part_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for g in MIDDLE_GROUPS:
        sub = part_df[part_df["group4"] == g]
        rho, p = _spearman_safe(
            sub["year"].to_numpy(dtype=float),
            sub["score"].to_numpy(dtype=float),
        )
        out[g] = {"rho": rho, "p": p, "n": int(len(sub))}

    mid = part_df[part_df["group4"].isin(MIDDLE_GROUPS)]
    rho, p = _spearman_safe(
        mid["year"].to_numpy(dtype=float),
        mid["score"].to_numpy(dtype=float),
    )
    out["middle"] = {"rho": rho, "p": p, "n": int(len(mid))}
    return out


def ols_point(year: np.ndarray, score: np.ndarray) -> dict[str, float]:
    year = np.asarray(year, dtype=float)
    score = np.asarray(score, dtype=float)
    X = sm.add_constant(year)
    fit = sm.OLS(score, X).fit()
    yhat = fit.predict(X)
    resid = score - yhat
    return {
        "slope": float(fit.params[1]),
        "intercept": float(fit.params[0]),
        "r2": float(fit.rsquared),
        "mae": float(np.mean(np.abs(resid))),
        "mse": float(np.mean(resid ** 2)),
        "n": int(len(score)),
    }


def _percentile_ci(arr: np.ndarray, alpha: float = 0.05) -> list[float]:
    a = arr[np.isfinite(arr)]
    if len(a) < 10:
        return [float("nan"), float("nan")]
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return [float(np.percentile(a, lo)), float(np.percentile(a, hi))]


def _bca_ci(
    theta_hat: float,
    boots: np.ndarray,
    jack: np.ndarray,
    alpha: float = 0.05,
) -> list[float]:
    """
    Bias-Corrected and accelerated (BCa) interval (Efron).
    Nécessaire pour R² (borne [0,1], asymétrie) ; pente/ρ restent en percentile.
    """
    boots = boots[np.isfinite(boots)]
    jack = jack[np.isfinite(jack)]
    if len(boots) < 50 or len(jack) < 3:
        return _percentile_ci(boots, alpha)

    # bias-correction
    prop = np.mean(boots < theta_hat)
    prop = np.clip(prop, 1e-6, 1 - 1e-6)
    z0 = stats.norm.ppf(prop)

    # acceleration via jackknife
    j_bar = jack.mean()
    num = np.sum((j_bar - jack) ** 3)
    den = np.sum((j_bar - jack) ** 2)
    if den <= 0:
        return _percentile_ci(boots, alpha)
    a_hat = num / (6.0 * den ** 1.5)

    def _adj(a: float) -> float:
        z_a = stats.norm.ppf(a)
        num_a = z0 + z_a
        return float(stats.norm.cdf(z0 + num_a / (1 - a_hat * num_a)))

    a1 = _adj(alpha / 2)
    a2 = _adj(1 - alpha / 2)
    a1, a2 = np.clip([a1, a2], 0.0, 1.0)
    return [float(np.quantile(boots, a1)), float(np.quantile(boots, a2))]


def bootstrap_ols_ci(
    year: np.ndarray,
    score: np.ndarray,
    *,
    B: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Bootstrap AU NIVEAU PARTICIPANT.
    - slope, mae : IC percentile
    - r2 : IC **BCa** (plus honnête sur n=25)
    """
    year = np.asarray(year, dtype=float)
    score = np.asarray(score, dtype=float)
    n = len(score)
    rng = np.random.default_rng(seed)

    point = ols_point(year, score)
    slopes = np.empty(B)
    r2s = np.empty(B)
    maes = np.empty(B)

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        if np.unique(year[idx]).size < 2:
            slopes[b] = r2s[b] = maes[b] = np.nan
            continue
        m = ols_point(year[idx], score[idx])
        slopes[b] = m["slope"]
        r2s[b] = m["r2"]
        maes[b] = m["mae"]

    # Jackknife leave-one-out pour BCa sur R²
    jack_r2 = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        if np.unique(year[mask]).size < 2:
            jack_r2[i] = np.nan
            continue
        jack_r2[i] = ols_point(year[mask], score[mask])["r2"]

    r2_bca = _bca_ci(point["r2"], r2s, jack_r2, alpha=alpha)

    return {
        **point,
        "bootstrap_B": B,
        "bootstrap_level": "participant",
        "slope_ci95": _percentile_ci(slopes, alpha),
        "slope_ci_method": "percentile",
        "r2_ci95": r2_bca,
        "r2_ci_method": "BCa",
        "r2_ci95_percentile": _percentile_ci(r2s, alpha),  # pour sensibilité
        "mae_ci95": _percentile_ci(maes, alpha),
        "mae_ci_method": "percentile",
        "n_valid_boot": int(np.isfinite(slopes).sum()),
    }


def compute_metrics_middle(
    part_df: pd.DataFrame,
    *,
    seed: int,
    condition: str = "gap",
    agg: AggMode = "mean",
    bootstrap_B: int = 5000,
    bootstrap_seed: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    part_df : déjà au niveau participant (colonnes group4, year, score).
    `agg` est enregistré pour traçabilité (l'agrégation a lieu en amont).
    """
    mid = part_df[part_df["group4"].isin(MIDDLE_GROUPS)].copy()
    if len(mid) == 0:
        raise ValueError("Aucun participant Junior/Senior dans part_df.")

    spearman = spearman_by_group(part_df)
    boot_seed = seed if bootstrap_seed is None else bootstrap_seed
    ols = bootstrap_ols_ci(
        mid["year"].to_numpy(dtype=float),
        mid["score"].to_numpy(dtype=float),
        B=bootstrap_B,
        seed=boot_seed,
    )

    payload: dict[str, Any] = {
        "seed": seed,
        "condition": condition,
        "agg_trial_to_participant": agg,
        "n_middle": int(len(mid)),
        "scores": "raw",
        "calibration": "display_only_not_used",
        "spearman": spearman,
        "ols_bootstrap": ols,
        "primary_metrics": {
            "slope": ols["slope"],
            "slope_ci95": ols["slope_ci95"],
            "rho_middle": spearman["middle"]["rho"],
            "rho_junior": spearman["junior"]["rho"],
            "rho_senior": spearman["senior"]["rho"],
        },
        "secondary_metrics": {
            "r2": ols["r2"],
            "r2_ci95": ols["r2_ci95"],
            "r2_ci_method": "BCa",
            "mae": ols["mae"],
            "mae_ci95": ols["mae_ci95"],
        },
        "inference_note": {
            "bootstrap": "sampling-variance on n_middle participants (one run)",
            "seed_tests": "see compare_pooling.py (sign-test / Wilcoxon) — separate inference",
        },
    }

    rows = [
        {"seed": seed, "cond": condition, "agg": agg, "metric": "slope", "value": ols["slope"]},
        {"seed": seed, "cond": condition, "agg": agg, "metric": "slope_ci95_lo", "value": ols["slope_ci95"][0]},
        {"seed": seed, "cond": condition, "agg": agg, "metric": "slope_ci95_hi", "value": ols["slope_ci95"][1]},
        {"seed": seed, "cond": condition, "agg": agg, "metric": "rho_middle", "value": spearman["middle"]["rho"]},
        {"seed": seed, "cond": condition, "agg": agg, "metric": "rho_junior", "value": spearman["junior"]["rho"]},
        {"seed": seed, "cond": condition, "agg": agg, "metric": "rho_senior", "value": spearman["senior"]["rho"]},
        {"seed": seed, "cond": condition, "agg": agg, "metric": "r2", "value": ols["r2"]},
        {"seed": seed, "cond": condition, "agg": agg, "metric": "r2_ci95_lo", "value": ols["r2_ci95"][0]},
        {"seed": seed, "cond": condition, "agg": agg, "metric": "r2_ci95_hi", "value": ols["r2_ci95"][1]},
        {"seed": seed, "cond": condition, "agg": agg, "metric": "mae", "value": ols["mae"]},
        {"seed": seed, "cond": condition, "agg": agg, "metric": "mae_ci95_lo", "value": ols["mae_ci95"][0]},
        {"seed": seed, "cond": condition, "agg": agg, "metric": "mae_ci95_hi", "value": ols["mae_ci95"][1]},
        {"seed": seed, "cond": condition, "agg": agg, "metric": "intercept", "value": ols["intercept"]},
        {"seed": seed, "cond": condition, "agg": agg, "metric": "n_middle", "value": float(len(mid))},
    ]
    return payload, pd.DataFrame(rows)
