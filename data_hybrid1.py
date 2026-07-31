"""
data_hybrid1.py
===============
Utilitaires de donnees pour la reproduction Hybrid1 (EV-ICEMS).

Donnees reelles uniquement :
  - data/trial_tensor_v2.pkl (136 trials, 47 participants, tracking_flag corrige)
Chaque trial : X_feat (T, 10), mask (T, 3), meta (group4, level9, y_reg, year,
label_extreme, T, fs=10).

Modes :
  - legacy (max_len < T) : average-pooling adaptatif vers max_len (hybrid1_faithful).
  - full_resolution (per-pair) : pas de pooling ; trials > window_len decoupes en
    fenetres non chevauchantes pour entrainement / scoring (eApp 2).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import pickle
import torch
from scipy.signal import butter, filtfilt

HYBRID1_DATA_FILES: dict[str, Path] = {
    "v1": Path("data/trial_tensor_v2.pkl"),
    "v2": Path("data/trial_tensor_v2.pkl"),
}


def resolve_hybrid1_paths(
    data: str | None = None,
    pkl: Path | None = None,
    out: Path | None = None,
) -> tuple[Path, Path]:
    """Resout (--data) -> (pkl, results_dir). --pkl/--out surchargent si fournis."""
    if data is not None:
        if data not in HYBRID1_DATA_FILES:
            raise ValueError(f"--data invalide : {data!r} (attendu v1 ou v2)")
        default_pkl = HYBRID1_DATA_FILES[data]
        default_out = Path(f"results/hybrid1_perpair_{data}")
    else:
        default_pkl = HYBRID1_DATA_FILES["v2"]
        default_out = Path("results/hybrid1_perpair")
    return pkl or default_pkl, out or default_out

GROUP4_ORDER = ["expert", "senior", "junior", "novice"]
GROUP4_DISPLAY = {
    "expert": "Experts",
    "senior": "Seniors",
    "junior": "Juniors",
    "novice": "Novices",
}
SUBLEVEL_ORDER = ["ms", "pgy1", "pgy2", "pgy3", "pgy4", "pgy5", "pgy6", "fellow", "staff"]
SUBLEVEL_RANK = {k: i for i, k in enumerate(SUBLEVEL_ORDER)}

# 10 paires outil-metrique (indices dans X_feat[:, k])
PAIR_NAMES = [
    "bip_v", "bip_a", "bip_j",
    "sci_v", "sci_a", "sci_j",
    "cav_v", "cav_a", "cav_j",
    "dist_bip_cav",
]
N_PAIRS = len(PAIR_NAMES)

# fc Butterworth ordre 4 a fs=10 Hz (Nyquist=5 Hz).
# Valeurs ICEMS par defaut — a confirmer vs eApp 1 ; toutes < Nyquist.
DEFAULT_BUTTERWORTH_FC: dict[str, float] = {
    "velocity": 2.0,
    "acceleration": 2.0,
    "jerk": 2.0,
    "distance": 2.0,
}
BUTTERWORTH_ORDER = 4


@dataclass
class Trial:
    participant: str
    trial: str
    X: np.ndarray          # (L, F) float32 ; F=10 (multivarie) ou 1 (per-pair)
    length: int            # nombre de frames valides (== len(X))
    group4: str
    level9: str
    y_reg: float
    year: int
    label_extreme: int | None
    pair_idx: int | None = None
    window_start: int = 0   # offset dans le trial original (0 si pas de fenetre)


def load_raw_trials(pkl_path: str) -> list[dict]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Format inattendu : {type(data)} (liste de dicts attendue).")
    return data


def _metric_kind(pair_idx: int) -> str:
    name = PAIR_NAMES[pair_idx]
    if name.endswith("_v"):
        return "velocity"
    if name.endswith("_a"):
        return "acceleration"
    if name.endswith("_j"):
        return "jerk"
    return "distance"


def fc_for_pair(pair_idx: int, fc_map: dict[str, float] | None = None) -> float:
    fc_map = fc_map or DEFAULT_BUTTERWORTH_FC
    return float(fc_map[_metric_kind(pair_idx)])


def butterworth_lowpass(
    x: np.ndarray, fs: float, fc: float, order: int = BUTTERWORTH_ORDER,
) -> np.ndarray:
    """Filtre passe-bas Butterworth (ordre 4 par defaut), zero-phase filtfilt."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < order * 3 + 1:
        return x.astype(np.float32)
    wn = fc / (0.5 * fs)
    if wn <= 0.0 or wn >= 1.0:
        return x.astype(np.float32)
    b, a = butter(order, wn, btype="low")
    try:
        y = filtfilt(b, a, x, axis=0)
    except ValueError:
        return x.astype(np.float32)
    return y.astype(np.float32)


def apply_butterworth_multivariate(
    X: np.ndarray, fs: float = 10.0,
    fc_map: dict[str, float] | None = None,
) -> np.ndarray:
    """Applique Butterworth par canal selon le type de metrique."""
    out = np.empty_like(X, dtype=np.float32)
    for k in range(X.shape[1]):
        fc = fc_for_pair(k, fc_map)
        out[:, k] = butterworth_lowpass(X[:, k], fs=fs, fc=fc)
    return out


def downsample_to_cap(X: np.ndarray, max_len: int) -> np.ndarray:
    """Average-pooling adaptatif : borne la longueur a max_len en preservant la forme."""
    T = X.shape[0]
    if T <= max_len:
        return X.astype(np.float32)
    groups = np.array_split(X, max_len, axis=0)
    pooled = np.stack([g.mean(axis=0) for g in groups], axis=0)
    return pooled.astype(np.float32)


def build_trials(
    raw: Sequence[dict],
    max_len: int,
    *,
    full_resolution: bool = False,
    apply_butterworth: bool = False,
    butterworth_fc: dict[str, float] | None = None,
) -> List[Trial]:
    """Construit la liste de trials.

    full_resolution=True : conserve T natif (pas de pooling).
    apply_butterworth=True : filtre Butterworth ordre 4 avant normalisation.
    """
    trials: List[Trial] = []
    for r in raw:
        X = np.asarray(r["X_feat"], dtype=np.float32)
        fs = float(r.get("fs", 10.0))
        if apply_butterworth:
            X = apply_butterworth_multivariate(X, fs=fs, fc_map=butterworth_fc)
        if not full_resolution:
            X = downsample_to_cap(X, max_len)
        trials.append(Trial(
            participant=str(r["participant"]),
            trial=str(r.get("trial", "")),
            X=X,
            length=int(X.shape[0]),
            group4=str(r["group4"]).strip().lower(),
            level9=str(r["level9"]).strip().lower(),
            y_reg=float(r["y_reg"]),
            year=int(r["year"]),
            label_extreme=(None if r["label_extreme"] is None
                           else int(r["label_extreme"])),
        ))
    return trials


def to_pair_trials(trials: Sequence[Trial], pair_idx: int) -> List[Trial]:
    """Extrait la paire k -> entree univariee (L, 1)."""
    out: List[Trial] = []
    for t in trials:
        x = t.X[:, pair_idx:pair_idx + 1].astype(np.float32)
        out.append(replace(t, X=x, length=int(x.shape[0]), pair_idx=pair_idx))
    return out


def iter_non_overlapping_windows(
    X: np.ndarray,
    window_len: int,
    stride: int | None = None,
) -> Iterator[Tuple[np.ndarray, int]]:
    """Decoupe X en fenetres de taille <= window_len.

    stride=None ou stride=window_len : fenetres non chevauchantes.
    stride < window_len : fenetres chevauchantes (train augmentation).
    """
    step = window_len if stride is None else stride
    T = X.shape[0]
    if T <= window_len:
        yield X, 0
        return
    start = 0
    while start < T:
        end = min(start + window_len, T)
        yield X[start:end], start
        if end >= T:
            break
        start += step


def expand_trials_to_windows(
    trials: Sequence[Trial],
    window_len: int,
    stride: int | None = None,
) -> List[Trial]:
    """Eclate les trials longs en fenetres (meme meta/label)."""
    expanded: List[Trial] = []
    for t in trials:
        for chunk, wstart in iter_non_overlapping_windows(t.X, window_len, stride=stride):
            expanded.append(replace(
                t,
                X=chunk.astype(np.float32),
                length=int(chunk.shape[0]),
                window_start=wstart,
            ))
    return expanded


def select_extremes(trials: Sequence[Trial]) -> List[Trial]:
    """Trials avec label_extreme in {+1 (expert), -1 (novice)}."""
    return [t for t in trials if t.label_extreme in (1, -1)]


def compute_norm_stats(trials: Sequence[Trial]) -> Tuple[np.ndarray, np.ndarray]:
    """Mean/std par feature sur toutes les frames des trials fournis."""
    allframes = np.concatenate([t.X for t in trials], axis=0)
    mean = allframes.mean(axis=0)
    std = allframes.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def stratified_trial_split(
    trials: list[Trial], seed: int,
    frac_train: float = 0.70, frac_val: float = 0.15,
) -> tuple[list[Trial], list[Trial], list[Trial]]:
    """Split 70/15/15 au niveau essai, stratifie par groupe et participant."""
    rng = np.random.RandomState(seed)
    train: list[Trial] = []
    val: list[Trial] = []
    test: list[Trial] = []

    by_group: dict[str, list[Trial]] = {}
    for t in trials:
        by_group.setdefault(t.group4, []).append(t)

    for _, gtrials in sorted(by_group.items()):
        by_part: dict[str, list[Trial]] = {}
        for t in gtrials:
            by_part.setdefault(t.participant, []).append(t)

        idx = []
        for part in sorted(by_part):
            idx.extend(by_part[part])
        perm = rng.permutation(len(idx))
        idx = [idx[i] for i in perm]

        n = len(idx)
        n_train = int(round(frac_train * n))
        n_val = int(round(frac_val * n))
        n_val = max(1, n_val) if n - n_train >= 2 else n_val
        train.extend(idx[:n_train])
        val.extend(idx[n_train:n_train + n_val])
        test.extend(idx[n_train + n_val:])

    return train, val, test


class TrialDataset(torch.utils.data.Dataset):
    """Dataset renvoyant (X_norm (L,F), length, target)."""

    def __init__(self, trials: Sequence[Trial], mean: np.ndarray, std: np.ndarray,
                 target: str = "label_extreme") -> None:
        self.trials = list(trials)
        self.mean = mean
        self.std = std
        self.target = target

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx: int):
        t = self.trials[idx]
        X = apply_norm(t.X, self.mean, self.std)
        if self.target == "label_extreme":
            y = float(t.label_extreme) if t.label_extreme is not None else 0.0
        elif self.target == "y_reg":
            y = float(t.y_reg)
        else:
            raise ValueError(f"target inconnu : {self.target}")
        return torch.from_numpy(X), t.length, torch.tensor([y], dtype=torch.float32)


def collate_pad(batch):
    """Pad a droite ; key_padding_mask True = padding."""
    xs, lengths, ys = zip(*batch)
    B = len(xs)
    Lmax = max(x.shape[0] for x in xs)
    F = xs[0].shape[1]
    X = torch.zeros(B, Lmax, F, dtype=torch.float32)
    key_padding_mask = torch.ones(B, Lmax, dtype=torch.bool)
    for i, x in enumerate(xs):
        L = x.shape[0]
        X[i, :L] = x
        key_padding_mask[i, :L] = False
    y = torch.stack(ys, dim=0)
    return X, key_padding_mask, y


def log_gpu_memory(tag: str = "") -> None:
    """Log memoire GPU allouee / reservee (Mo)."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    peak = torch.cuda.max_memory_allocated() / 1e6
    prefix = f"[gpu]{f' {tag}' if tag else ''}"
    print(f"{prefix} alloc={alloc:.0f}MB reserved={reserved:.0f}MB peak={peak:.0f}MB")
