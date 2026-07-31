# -*- coding: utf-8 -*-
"""Utilitaires partagés pour audit / build depuis data/full_data.json."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

SEED = 42
SAMPLE_RATE_HZ = 10.0

MONO_INSTRUMENTS = ("bipolar", "scissors", "cavitron")
DIST_INSTRUMENT = "bipolar_to_cavitron"
MONO_KIN_METRICS = ("velocity", "acceleration", "jerk")
MONO_REQUIRED_METRICS = MONO_KIN_METRICS + ("position",)
# v1 : captured_flag → inuse_flag → tracking_flag (masque + zeroing cinématiques)
CAPTURE_FLAG_CHAIN = ("captured_flag", "inuse_flag", "tracking_flag")
# v2 : tracking_flag seul (détection tracker, pas d'activité inuse)
TRACKING_FLAG_CHAIN = ("tracking_flag",)
DIST_FLAG_CHAIN = ("bidist_captured_flag", "captured_flag")

EXPERTISE_TO_GROUP4 = {
    "Student": "novice",
    "Junior": "junior",
    "Senior": "senior",
    "Expert": "expert",
}

YEAR_BY_LEVEL9 = {
    "ms": 0,
    "pgy1": 1,
    "pgy2": 2,
    "pgy3": 3,
    "pgy4": 4,
    "pgy5": 5,
    "pgy6": 6,
    "fellow": 7,
    "staff": 8,
}

LEVEL9_ORDER = [
    "Medical student",
    "Resident PGY1",
    "Resident PGY2",
    "Resident PGY3",
    "Resident PGY4",
    "Resident PGY5",
    "Resident PGY6",
    "Fellow",
    "Staff",
]
LEVEL_TO_IDX = {n: i for i, n in enumerate(LEVEL9_ORDER)}
CLASS_TO_REG = {i: round(i / 4.0 - 1.0, 4) for i in range(9)}
CLASS_TO_REG[7] = 0.86


def _idx_key(i: int | str, container: dict) -> str | int:
    if str(i) in container:
        return str(i)
    return i


def _first(bundle: dict | None, key: str) -> Any:
    if bundle is None:
        return None
    arrs = bundle.get(key)
    if arrs is None:
        return None
    if isinstance(arrs, list) and arrs:
        return arrs[0]
    return arrs


def _as_1d(x: Any, dtype: type = float) -> np.ndarray | None:
    if x is None:
        return None
    return np.asarray(x, dtype=dtype).ravel()


def align_len(*arrs: np.ndarray | None) -> tuple[list[np.ndarray | None], int]:
    lengths = [len(a) for a in arrs if a is not None]
    if not lengths:
        return [None for _ in arrs], 0
    n = min(lengths)
    if n <= 0:
        return [None for _ in arrs], 0
    out: list[np.ndarray | None] = []
    for a in arrs:
        if a is None:
            out.append(None)
        else:
            out.append(a[:n])
    return out, n


def load_full_data(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_by_participant_trial(raw: dict) -> tuple[dict[tuple[str, str], dict[str, dict]], dict[tuple[str, str, str], tuple[str, str]]]:
    """Regroupe full_data.json par (participant, trial) -> instrument -> metric -> array."""
    n = len(raw["participant"])
    by_pt: dict[tuple[str, str], dict[str, dict[str, list]]] = defaultdict(lambda: defaultdict(dict))
    labels: dict[tuple[str, str, str], tuple[str, str]] = {}

    for i in range(n):
        ik = _idx_key(i, raw["participant"])
        pid = str(raw["participant"][ik])
        tid = str(raw["trial"][ik])
        inst = str(raw["instrument"][ik])
        met = str(raw["metric"][ik])
        arr = raw["data"][ik]

        key_pt = (pid, tid)
        key_pti = (pid, tid, inst)
        labels[key_pti] = (
            str(raw["expertise"][ik] or "").strip(),
            str(raw["level"][ik] or "").strip(),
        )
        by_pt[key_pt][inst].setdefault(met, []).append(arr)

    return by_pt, labels


def get_flag_bundle(bundle: dict, chain: tuple[str, ...] = CAPTURE_FLAG_CHAIN) -> np.ndarray | None:
    for key in chain:
        arr = _first(bundle, key)
        if arr is not None:
            return _as_1d(arr, bool)
    return None


def metric_present(bundle: dict, metric: str) -> bool:
    return _first(bundle, metric) is not None


def metric_length(bundle: dict, metric: str) -> int:
    arr = _first(bundle, metric)
    if arr is None:
        return 0
    return len(_as_1d(arr))


def capture_fraction(bundle: dict) -> float | None:
    cap = get_flag_bundle(bundle, CAPTURE_FLAG_CHAIN)
    if cap is None or len(cap) == 0:
        return None
    return float(np.mean(cap.astype(bool)))


def normalize_level(raw: str) -> str | None:
    """Aligné sur build_continuous_dataset.normalize_level."""
    if raw is None:
        return None
    s = " ".join(str(raw).strip().split()).lower()

    if s.startswith("medical student"):
        return "Medical student"
    if s.startswith("staff") or s.startswith("neurosurgeon"):
        return "Staff"
    if s.startswith("fellow"):
        return "Fellow"
    if s.startswith("resident pgy"):
        for k in range(1, 7):
            if f"pgy{k}" in s:
                return f"Resident PGY{k}"
    if s.startswith("pgy"):
        for k in range(1, 7):
            if s == f"pgy{k}":
                return f"Resident PGY{k}"
    return None


def level_to_level9(level_canon: str) -> str:
    mapping = {
        "Medical student": "ms",
        "Resident PGY1": "pgy1",
        "Resident PGY2": "pgy2",
        "Resident PGY3": "pgy3",
        "Resident PGY4": "pgy4",
        "Resident PGY5": "pgy5",
        "Resident PGY6": "pgy6",
        "Fellow": "fellow",
        "Staff": "staff",
    }
    return mapping[level_canon]


def trial_labels(
    pid: str,
    tid: str,
    inst_map: dict[str, dict],
    labels: dict[tuple[str, str, str], tuple[str, str]],
) -> tuple[str, str, str | None, str | None, float | None, int | None, int | None, int | None]:
    """Retourne expertise, level_raw, group4, level9, y_reg, year, label_extreme, y9."""
    expertise = level_raw = None
    for inst in MONO_INSTRUMENTS:
        key = (pid, tid, inst)
        if key in labels:
            expertise, level_raw = labels[key]
            break
    if expertise is None:
        for inst in inst_map:
            key = (pid, tid, inst)
            if key in labels:
                expertise, level_raw = labels[key]
                break

    group4 = EXPERTISE_TO_GROUP4.get(expertise) if expertise else None
    level_canon = normalize_level(level_raw) if level_raw else None
    level9 = level_to_level9(level_canon) if level_canon else None
    y9 = LEVEL_TO_IDX.get(level_canon) if level_canon else None
    y_reg = CLASS_TO_REG[y9] if y9 is not None else None
    year = YEAR_BY_LEVEL9.get(level9) if level9 else None

    if group4 == "expert":
        label_extreme = 1
    elif group4 == "novice":
        label_extreme = -1
    else:
        label_extreme = None

    return expertise, level_raw, group4, level9, y_reg, year, label_extreme, y9


def take_metric(bundle: dict, metric: str, n: int, dtype: type = float) -> np.ndarray:
    arr = _first(bundle, metric)
    if arr is None:
        return np.zeros(n, dtype=dtype)
    a = _as_1d(arr, dtype)
    if len(a) >= n:
        return a[:n]
    out = np.zeros(n, dtype=dtype)
    out[: len(a)] = a
    return out


def masked_metric(bundle: dict, metric: str, cap: np.ndarray) -> np.ndarray:
    n = len(cap)
    values = take_metric(bundle, metric, n, float)
    return np.where(cap, values, 0.0)


def raw_metric(bundle: dict, metric: str, n: int, dtype: type = float) -> np.ndarray:
    """Cinématique réelle depuis full_data.json, sans zeroing arbitraire."""
    return take_metric(bundle, metric, n, dtype)


def trial_common_length(
    inst_map: dict[str, dict],
    kinematic_flag_chain: tuple[str, ...] = CAPTURE_FLAG_CHAIN,
) -> int:
    """Longueur commune T = min sur tous les instruments/métriques requis."""
    lengths: list[int] = []

    for inst in MONO_INSTRUMENTS:
        bundle = inst_map.get(inst)
        if bundle is None:
            continue
        for met in MONO_REQUIRED_METRICS:
            lengths.append(metric_length(bundle, met))
        cap = get_flag_bundle(bundle, kinematic_flag_chain)
        if cap is not None:
            lengths.append(len(cap))

    dist_bundle = inst_map.get(DIST_INSTRUMENT)
    if dist_bundle is not None:
        lengths.append(metric_length(dist_bundle, "distance"))
        flag = get_flag_bundle(dist_bundle, DIST_FLAG_CHAIN)
        if flag is not None:
            lengths.append(len(flag))

    return min(lengths) if lengths else 0


def build_trial_record(
    pid: str,
    tid: str,
    inst_map: dict[str, dict],
    labels: dict,
    *,
    kinematic_flag_chain: tuple[str, ...],
    zero_kinematics: bool,
    zero_distance: bool,
    mask_field: str,
) -> dict | None:
    """Construit un enregistrement trial-level (v1 ou v2 selon les flags)."""
    t = trial_common_length(inst_map, kinematic_flag_chain)
    if t <= 0:
        return None

    x_feat = np.zeros((t, 10), dtype=np.float32)
    mask = np.zeros((t, 3), dtype=bool)

    col = 0
    for i_inst, inst in enumerate(MONO_INSTRUMENTS):
        bundle = inst_map.get(inst)
        if bundle is None:
            col += 3
            continue

        trk = get_flag_bundle(bundle, kinematic_flag_chain)
        if trk is None:
            trk = np.ones(t, dtype=bool)
        else:
            aligned, _ = align_len(trk)
            trk = np.asarray(aligned[0], dtype=bool)[:t]

        mask[:, i_inst] = trk

        for met in MONO_KIN_METRICS:
            if zero_kinematics:
                x_feat[:, col] = masked_metric(bundle, met, trk).astype(np.float32)
            else:
                x_feat[:, col] = raw_metric(bundle, met, t, float).astype(np.float32)
            col += 1

    dist_bundle = inst_map.get(DIST_INSTRUMENT)
    if dist_bundle is not None:
        dist = raw_metric(dist_bundle, "distance", t, float)
        dflag = get_flag_bundle(dist_bundle, DIST_FLAG_CHAIN)
        if dflag is None:
            x_feat[:, 9] = dist.astype(np.float32)
        else:
            aligned, _ = align_len(dflag)
            dflag = np.asarray(aligned[0], dtype=bool)[:t]
            if zero_distance:
                x_feat[:, 9] = np.where(dflag, dist, 0.0).astype(np.float32)
            else:
                x_feat[:, 9] = dist.astype(np.float32)

    active_count = mask.sum(axis=1)
    mask_ge1 = active_count >= 1
    mask_ge2 = active_count >= 2

    expertise, level_raw, group4, level9, y_reg, year, label_extreme, y9 = trial_labels(
        pid, tid, inst_map, labels
    )

    rec = {
        "participant": pid,
        "trial": tid,
        "X_feat": x_feat,
        mask_field: mask,
        "active_count": active_count.astype(np.int8),
        "mask_ge1": mask_ge1,
        "mask_ge2": mask_ge2,
        "group4": group4,
        "level9": level9,
        "y_reg": y_reg,
        "year": year,
        "label_extreme": label_extreme,
        "expertise": expertise,
        "level": level_raw,
        "y9": y9,
        "T": int(t),
        "fs": float(SAMPLE_RATE_HZ),
    }
    return rec
