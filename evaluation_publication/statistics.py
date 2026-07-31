"""Statistical helpers for publication metrics."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import stats


def mean_ci95(values: Sequence[float], n_boot: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) via percentile bootstrap."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        v = float(arr[0])
        return v, v, v
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = rng.choice(arr, size=arr.size, replace=True).mean()
    return float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def spearman_with_p(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 3:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(x_arr, y_arr)
    return float(rho), float(p)


def pearson_r2(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, float]:
    """OLS slope/intercept on (x, y) and coefficient of determination R²."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 2:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x_arr, y_arr, 1)
    y_hat = slope * x_arr + intercept
    ss_res = float(np.sum((y_arr - y_hat) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r2 = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    return float(slope), float(intercept), r2


def summarise_groups(
    values_by_key: Dict[str, Sequence[float]],
    order: Sequence[str],
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for key in order:
        vals = list(values_by_key.get(key, []))
        mean, lo, hi = mean_ci95(vals)
        arr = np.asarray(vals, dtype=float)
        out[key] = {
            "n": int(arr.size),
            "mean": mean,
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "median": float(np.median(arr)) if arr.size else float("nan"),
        }
    return out
