"""
Configuration expérimentale — Train extrêmes → Test intermédiaires.
Isolé du code principal ; importe uniquement les constantes partagées.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet

# Racine du repo ICEMS (parents[2] depuis ce fichier)
REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"

# ─── Split extrêmes vs intermédiaires ─────────────────────────────────────────
TRAIN_SUBLEVELS: FrozenSet[str] = frozenset({"ms", "staff"})
INTERMEDIATE_SUBLEVELS: FrozenSet[str] = frozenset({
    "pgy1", "pgy2", "pgy3", "pgy4", "pgy5", "pgy6", "fellow",
})
JUNIOR_SUBLEVELS: FrozenSet[str] = frozenset({
    "pgy1", "pgy2", "pgy3", "pgy4", "pgy5",
})
SENIOR_SUBLEVELS: FrozenSet[str] = frozenset({"pgy6", "fellow"})

# Cibles d'entraînement (extrêmes uniquement)
EXTREME_TARGETS = {"ms": -1.0, "staff": +1.0}

# ─── Données ATRACSYS (continuous_per_trial.pkl) ─────────────────────────────
DEFAULT_PKL = REPO_ROOT / "data" / "continuous_per_trial.pkl"
DEFAULT_CSV = REPO_ROOT / "data" / "Exvivo_trial_Participants(Sheet1).csv"

# Features : 10 canaux ATRACSYS + 3 dérivées = 13 (aligné train_hybrid_lopo)
N_BASE_FEATURES = 10
VALID_COL = 9
N_FEATURES = 13
SEQ_LEN = 4000
SEQ_LEN_SMOKE = 512

# ─── Entraînement ─────────────────────────────────────────────────────────────
PARALLEL_SEEDS = (42, 123, 456, 789, 2024)


@dataclass
class ExpTrainConfig:
    seq_len: int = SEQ_LEN
    crop_mode: str = "start"
    batch_size: int = 16
    max_epochs: int = 50
    patience: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    global_multiplier: int = 4
    use_hoel: bool = True


@dataclass
class ExpAugmentConfig:
    jitter_sigma: float = 0.015
    warp_sigma: float = 0.04
    warp_knots: int = 3
    magnitude_sigma: float = 0.06
    mask_ratio: float = 0.10
    mask_min_len: int = 50
    p_time_warp: float = 0.60
    p_magnitude: float = 0.50
    p_jitter: float = 0.70
    p_mask: float = 0.40


def output_dir(seed: int, smoke: bool = False) -> Path:
    base = EXP_ROOT / "results"
    if smoke:
        return base / "smoke" / f"seed_{seed}"
    return base / f"seed_{seed}"
