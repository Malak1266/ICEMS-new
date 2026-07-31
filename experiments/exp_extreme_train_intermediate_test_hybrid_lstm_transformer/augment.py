"""
Augmentation conservative — TRAIN extrêmes uniquement.
jitter | time-warp | magnitude scaling | masking partiel
"""
from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Sequence

import numpy as np

from exp_config import ExpAugmentConfig
from data_split import TrialRecord


def _import_augment_ops():
    import sys
    from pathlib import Path
    src = Path(__file__).resolve().parents[2] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from eval.augment_extremes import (
        augment_trial,
        jitter_sequence,
        magnitude_warp_sequence,
        time_warp_sequence,
    )
    return jitter_sequence, time_warp_sequence, magnitude_warp_sequence, augment_trial


def partial_mask_sequence(
    X: np.ndarray,
    mask_ratio: float = 0.10,
    min_len: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Masque un segment contigu (zéros) — préserve début/fin chirurgicaux."""
    if rng is None:
        rng = np.random.default_rng()
    T, F = X.shape
    seg_len = max(min_len, int(T * mask_ratio))
    seg_len = min(seg_len, T - 1)
    if seg_len <= 0:
        return X.astype(np.float32)
    start = int(rng.integers(0, T - seg_len + 1))
    out = X.copy()
    out[start : start + seg_len] = 0.0
    return out.astype(np.float32)


def augment_trial_conservative(
    X: np.ndarray,
    cfg: ExpAugmentConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    jitter_sequence, time_warp_sequence, magnitude_warp_sequence, _ = _import_augment_ops()
    out = X.copy()
    if rng.random() < cfg.p_time_warp:
        out = time_warp_sequence(
            out, sigma=cfg.warp_sigma, n_knots=cfg.warp_knots, rng=rng,
        )
    if rng.random() < cfg.p_magnitude:
        out = magnitude_warp_sequence(
            out, sigma=cfg.magnitude_sigma, n_knots=4, rng=rng,
        )
    if rng.random() < cfg.p_jitter:
        out = jitter_sequence(out, sigma=cfg.jitter_sigma, rng=rng)
    if rng.random() < cfg.p_mask:
        out = partial_mask_sequence(
            out, mask_ratio=cfg.mask_ratio, min_len=cfg.mask_min_len, rng=rng,
        )
    return out.astype(np.float32)


def augment_train_extremes(
    train_real: Sequence[TrialRecord],
    seed: int = 42,
    cfg: Optional[ExpAugmentConfig] = None,
    global_multiplier: int = 4,
) -> List[TrialRecord]:
    """
    Étape 1 : équilibrage staff→ms (jitter léger)
    Étape 2 : ×global_multiplier via pipeline conservative
    """
    if cfg is None:
        cfg = ExpAugmentConfig()
    jitter_sequence, _, _, _ = _import_augment_ops()
    rng = np.random.default_rng(seed)

    ms_real = [t for t in train_real if t.sublevel == "ms"]
    staff_real = [t for t in train_real if t.sublevel == "staff"]
    n_ms, n_staff = len(ms_real), len(staff_real)
    n_to_create = max(0, n_ms - n_staff)

    synth: List[TrialRecord] = []
    for _ in range(n_to_create):
        if not staff_real:
            break
        parent = staff_real[int(rng.integers(0, len(staff_real)))]
        X_new = jitter_sequence(parent.X, sigma=cfg.jitter_sigma, rng=rng)
        synth.append(_clone_trial(parent, X_new, f"{parent.trial_id}_eq_jitter"))

    balanced = list(train_real) + synth
    global_synth: List[TrialRecord] = []
    for parent in balanced:
        for j in range(max(0, global_multiplier - 1)):
            X_new = augment_trial_conservative(parent.X, cfg, rng)
            global_synth.append(
                _clone_trial(parent, X_new, f"{parent.trial_id}_aug_{j}")
            )

    out = balanced + global_synth
    rng.shuffle(out)
    return out


def _clone_trial(parent: TrialRecord, X_new: np.ndarray, trial_id: str) -> TrialRecord:
    return TrialRecord(
        X=X_new,
        score=parent.score,
        train_score=parent.train_score,
        tier=parent.tier,
        rank=parent.rank,
        participant=parent.participant,
        sublevel=parent.sublevel,
        trial_id=trial_id,
        subgroup=parent.subgroup,
        is_augmented=True,
    )
