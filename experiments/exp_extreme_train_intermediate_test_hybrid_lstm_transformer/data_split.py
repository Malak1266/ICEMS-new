"""
Chargement ATRACSYS + split strict extrêmes (train) vs intermédiaires (test/val).
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from exp_config import (
    DEFAULT_CSV,
    DEFAULT_PKL,
    EXTREME_TARGETS,
    INTERMEDIATE_SUBLEVELS,
    JUNIOR_SUBLEVELS,
    N_BASE_FEATURES,
    N_FEATURES,
    REPO_ROOT,
    SENIOR_SUBLEVELS,
    SRC_ROOT,
    TRAIN_SUBLEVELS,
    VALID_COL,
)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import normalize_sublevel, sublevel_score  # noqa: E402
from config import SUBLEVEL_TO_RANK, SUBLEVEL_TO_TIER  # noqa: E402


def enrich_kinematic_features(X: np.ndarray) -> np.ndarray:
    """10 canaux ATRACSYS → 13 (vel, jerk/vel, valid dérivés)."""
    vel = np.linalg.norm(X[:, [0, 3, 6]], axis=1).astype(np.float32)
    jerk = np.linalg.norm(X[:, [2, 5, 8]], axis=1).astype(np.float32)
    valid = X[:, VALID_COL].astype(np.float32)
    derived = np.stack([vel, jerk / (vel + 1e-6), valid], axis=1)
    return np.concatenate([X[:, :N_BASE_FEATURES], derived], axis=1).astype(np.float32)


@dataclass
class TrialRecord:
    X: np.ndarray
    score: float
    train_score: float
    tier: int
    rank: int
    participant: str
    sublevel: str
    trial_id: str
    subgroup: str
    is_augmented: bool = False


def _load_participant_levels(csv_path: Path) -> Dict[str, str]:
    df = pd.read_csv(csv_path, sep=";", encoding="latin-1")
    df = df.dropna(subset=["ID"])
    df["participant"] = df["ID"].astype(int).astype(str).str.zfill(8)
    df["level"] = df["Level"].astype(str).str.strip()
    return dict(zip(df["participant"], df["level"]))


def _subgroup_for(sublevel: str) -> str:
    if sublevel in JUNIOR_SUBLEVELS:
        return "junior"
    if sublevel in SENIOR_SUBLEVELS:
        return "senior"
    if sublevel in TRAIN_SUBLEVELS:
        return "extreme"
    return "other"


def load_atracsys_trials(
    pkl_path: Path = DEFAULT_PKL,
    csv_path: Path = DEFAULT_CSV,
) -> List[TrialRecord]:
    """Charge continuous_per_trial.pkl (trajectoires ATRACSYS) → TrialRecord."""
    pkl_path = Path(pkl_path)
    csv_path = Path(csv_path)
    pid_to_level = _load_participant_levels(csv_path)

    with open(pkl_path, "rb") as f:
        raw: dict = pickle.load(f)

    trials: List[TrialRecord] = []
    missing: set = set()

    for (pid, tid), rec in raw.items():
        pid = str(pid)
        tid = str(tid)
        level_raw = pid_to_level.get(pid) or rec.get("level", "")
        if pid not in pid_to_level:
            missing.add(pid)

        canonical = normalize_sublevel(str(level_raw))
        score = sublevel_score(str(level_raw))
        tier = SUBLEVEL_TO_TIER[canonical]
        rank = SUBLEVEL_TO_RANK[canonical]
        X = enrich_kinematic_features(np.asarray(rec["X"], dtype=np.float32))
        train_score = EXTREME_TARGETS.get(canonical, score)

        trials.append(TrialRecord(
            X=X,
            score=float(score),
            train_score=float(train_score),
            tier=int(tier),
            rank=int(rank),
            participant=pid,
            sublevel=canonical,
            trial_id=str(tid),
            subgroup=_subgroup_for(canonical),
            is_augmented=False,
        ))

    if missing:
        raise KeyError(
            f"{len(missing)} participant(s) absents du CSV : {sorted(missing)[:5]}"
        )
    return trials


def split_extreme_intermediate(
    trials: Sequence[TrialRecord],
) -> Tuple[List[TrialRecord], List[TrialRecord]]:
    """
    Split strict :
      TRAIN  = ms + staff (extrêmes)
      TEST   = pgy1…pgy5 (junior) + pgy6 + fellow (senior)
    """
    real = [t for t in trials if not t.is_augmented]
    train = [t for t in real if t.sublevel in TRAIN_SUBLEVELS]
    test = [t for t in real if t.sublevel in INTERMEDIATE_SUBLEVELS]

    train_pids = {t.participant for t in train}
    test_pids = {t.participant for t in test}
    overlap = train_pids & test_pids
    if overlap:
        raise ValueError(f"Fuite participant train/test : {overlap}")

    if not train:
        raise ValueError("Pool TRAIN vide — aucun trial ms/staff.")
    if not test:
        raise ValueError("Pool TEST vide — aucun trial intermédiaire.")

    return train, test


def split_summary(train: Sequence[TrialRecord], test: Sequence[TrialRecord]) -> dict:
    from collections import Counter

    def _counts(items: Sequence[TrialRecord]) -> dict:
        return dict(Counter(t.sublevel for t in items))

    return {
        "n_train_trials": len(train),
        "n_test_trials": len(test),
        "n_train_participants": len({t.participant for t in train}),
        "n_test_participants": len({t.participant for t in test}),
        "train_by_sublevel": _counts(train),
        "test_by_sublevel": _counts(test),
        "test_junior": sum(1 for t in test if t.subgroup == "junior"),
        "test_senior": sum(1 for t in test if t.subgroup == "senior"),
    }
