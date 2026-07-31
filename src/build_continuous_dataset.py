"""
build_continuous_dataset.py
============================
Construit `data/continuous_per_trial.pkl` à partir de `data/filtered_data.json`.

Format de sortie (cf. PROJECT_PROGRESSION.md §4.5) :
    {
        (participant_id, trial_id): {
            "X":      np.ndarray,   # (T, n_features)  — séquence variable
            "y9":     int,          # classe 0..8
            "y_reg":  float,        # cible régression dans [-1, +1]
            "level":  str,          # ex : "Medical student", "Staff"
            "T":      int,          # nombre de frames
            "fs":     float,        # fréquence d'échantillonnage estimée (Hz)
        },
        ...
    }

Architecture des features (10 canaux) :
    [ bipolar.vel,  bipolar.acc,  bipolar.jerk,
      scissors.vel, scissors.acc, scissors.jerk,
      cavitron.vel, cavitron.acc, cavitron.jerk,
      valid_ratio ]                          # proxy : 1.0 si activité, 0 sinon

⚠️  Note sur les features :
    - Le pipeline MAE local (build_mae_dataset_enrichi.py) travaille à 4 features
      pour une session ex vivo SINGLE-instrument (sphères Atracsys).
    - Ici on a 3 instruments simultanés en chirurgie simulée → 9 canaux cinématiques
      + 1 valid_ratio = 10 canaux. C'est volontairement différent : ce sont deux
      pipelines avec des sources de données différentes.
    - `continuous_scorer.py` accepte n'importe quel n_features via ScorerConfig.

Mapping classes (Tâche D + barème §4.3 du PROJECT_PROGRESSION.md) :
    Class 0 : Medical student → y_reg = -1.00
    Class 1 : Resident PGY1   → y_reg = -0.71
    Class 2 : Resident PGY2   → y_reg = -0.43
    Class 3 : Resident PGY3   → y_reg = -0.14
    Class 4 : Resident PGY4   → y_reg = +0.14
    Class 5 : Resident PGY5   → y_reg = +0.43
    Class 6 : Resident PGY6   → y_reg = +0.71
    Class 7 : Fellow          → y_reg = +0.86  (toutes spécialités fusionnées)
    Class 8 : Staff           → y_reg = +1.00

Usage :
    python src/build_continuous_dataset.py
    python src/build_continuous_dataset.py --input data/filtered_data.json --output data/continuous_per_trial.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


# ── Constantes du mapping (alignées sur PROJECT_PROGRESSION.md §4.3) ───────
LEVEL9_ORDER = [
    "Medical student",   # 0
    "Resident PGY1",     # 1
    "Resident PGY2",     # 2
    "Resident PGY3",     # 3
    "Resident PGY4",     # 4
    "Resident PGY5",     # 5
    "Resident PGY6",     # 6
    "Fellow",            # 7
    "Staff",             # 8
]
LEVEL_TO_IDX = {n: i for i, n in enumerate(LEVEL9_ORDER)}

# Barème de régression linéaire entre Student (-1) et Staff (+1).
# Calculé comme : y_reg = (idx / 4) - 1, soit 9 valeurs équidistantes dans [-1, +1].
CLASS_TO_REG = {i: round(i / 4.0 - 1.0, 4) for i in range(9)}
# Override Fellow à +0.86 pour respecter la projection §4.3 (slot 7 ≠ 7/4-1=0.75).
CLASS_TO_REG[7] = 0.86

# Architecture des features — 9 canaux cinématiques + 1 proxy valid_ratio.
INSTRUMENT_ORDER = ["bipolar", "scissors", "cavitron"]
METRIC_ORDER     = ["velocity", "acceleration", "jerk"]
N_KIN_FEATURES   = len(INSTRUMENT_ORDER) * len(METRIC_ORDER)   # = 9
N_FEATURES_TOTAL = N_KIN_FEATURES + 1                          # +1 pour valid_ratio


# ── Normalisation des labels ───────────────────────────────────────────────
def normalize_level(raw: str) -> str | None:
    """
    Convertit un label brut en l'un des 9 niveaux canoniques de LEVEL9_ORDER.
    Retourne None si le label n'est pas reconnu (à ignorer dans le pipeline).

    Règles :
      - Whitespace ignoré, casse normalisée.
      - Tous les "Fellow X" (Spine, Epilepsy, Oncology, Pediatrics, functional,
        Fellow/Spine, ...) → "Fellow" (classe 7 unique).
      - "Medical student" → classe 0.
      - "Resident PGYn" → classe n.
      - "Staff" → classe 8.
    """
    if raw is None:
        return None
    s = " ".join(str(raw).strip().split()).lower()

    if s.startswith("medical student"):
        return "Medical student"
    if s.startswith("staff"):
        return "Staff"
    if s.startswith("fellow"):
        return "Fellow"
    if s.startswith("resident pgy"):
        # "resident pgy1" → "Resident PGY1"
        for k in range(1, 7):
            if f"pgy{k}" in s:
                return f"Resident PGY{k}"
    if s.startswith("pgy"):
        for k in range(1, 7):
            if s == f"pgy{k}":
                return f"Resident PGY{k}"
    return None


# ── Chargement et regroupement ─────────────────────────────────────────────
def load_filtered_data(path: Path) -> dict:
    """Charge filtered_data.json et le retourne tel quel."""
    print(f"Chargement de {path} …")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    n = len(d["participant"])
    print(f"  {n} entrées brutes chargées.")
    return d


def estimate_sampling_rate(timestamps: np.ndarray) -> float:
    """Estime la fréquence d'échantillonnage en Hz à partir d'un vecteur de timestamps."""
    if len(timestamps) < 2:
        return 10.0  # défaut Atracsys
    dt = np.diff(timestamps)
    dt = dt[dt > 1e-6]   # ignorer les zéros
    if len(dt) == 0:
        return 10.0
    return float(1.0 / np.median(dt))


def synthesize_valid_ratio(kin_features: np.ndarray) -> np.ndarray:
    """
    Génère un proxy valid_ratio (T,) à partir des 9 features cinématiques (T, 9).

    Heuristique : une frame est "occluse" si TOUS les canaux sont à 0 (pas
    d'activité détectée sur aucun instrument). Dans ce cas valid_ratio = 0.0.
    Sinon valid_ratio = 1.0.

    Remplace `valid_ratio` réel des sphères Atracsys (non disponible ici).
    """
    activity = np.any(np.abs(kin_features) > 1e-6, axis=1)
    return activity.astype(np.float32)


def build_trial_tensor(
    entries: list[dict],
    T_target: int,
) -> Tuple[np.ndarray, float]:
    """
    Reconstruit un tenseur (T, 10) pour un trial donné, à partir de ses entrées.

    Chaque entrée correspond à un (instrument, metric) parmi les
    {bipolar, scissors, cavitron} × {velocity, acceleration, jerk}.

    Args:
        entries  : liste des dicts {instrument, metric, data, len, …}
                   pour un même (participant, trial).
        T_target : longueur cible (la plus longue des entrées du trial).

    Returns:
        X    : ndarray (T_target, 10)
        fs   : fréquence d'échantillonnage estimée (Hz)
    """
    X = np.zeros((T_target, N_FEATURES_TOTAL), dtype=np.float32)

    timestamps = None
    for e in entries:
        instr  = e["instrument"]
        metric = e["metric"]
        data   = np.asarray(e["data"], dtype=np.float32)

        if metric == "timestamp":
            if timestamps is None or len(data) > len(timestamps):
                timestamps = data
            continue
        if instr not in INSTRUMENT_ORDER or metric not in METRIC_ORDER:
            continue

        col = INSTRUMENT_ORDER.index(instr) * len(METRIC_ORDER) + METRIC_ORDER.index(metric)
        # Padding ou truncation à T_target.
        n = min(len(data), T_target)
        X[:n, col] = data[:n]

    # Synthétiser valid_ratio dans la dernière colonne.
    X[:, N_KIN_FEATURES] = synthesize_valid_ratio(X[:, :N_KIN_FEATURES])

    fs = estimate_sampling_rate(timestamps) if timestamps is not None else 10.0
    return X, fs


# ── Pipeline principal ─────────────────────────────────────────────────────
def build_continuous_dataset(
    input_path: Path,
    output_path: Path,
    verbose: bool = True,
) -> Dict[Tuple[str, str], dict]:
    """Construit et sauvegarde le dataset continu trial-level."""
    raw = load_filtered_data(input_path)
    n = len(raw["participant"])

    # Regrouper par (participant, trial).
    by_pti: dict[Tuple[str, str], list[dict]] = defaultdict(list)
    by_level: dict[Tuple[str, str], str] = {}

    skipped_unknown_level = 0
    for i in range(n):
        s = str(i)
        pid   = raw["participant"][s]
        tid   = raw["trial"][s]
        level_raw = raw["level"][s]
        level = normalize_level(level_raw)

        if level is None:
            skipped_unknown_level += 1
            continue

        by_pti[(pid, tid)].append({
            "instrument": raw["instrument"][s],
            "metric":     raw["metric"][s],
            "data":       raw["data"][s],
            "len":        raw["len"][s],
        })
        by_level[(pid, tid)] = level

    if verbose:
        print(f"\n{len(by_pti)} trials uniques regroupés "
              f"(skipped {skipped_unknown_level} entrées au level inconnu).")

    # Construire le tenseur par trial.
    out: Dict[Tuple[str, str], dict] = {}
    class_counts: dict[str, int] = defaultdict(int)

    for (pid, tid), entries in by_pti.items():
        T_target = max(int(e["len"]) for e in entries)
        X, fs = build_trial_tensor(entries, T_target)

        level = by_level[(pid, tid)]
        y9    = LEVEL_TO_IDX[level]
        y_reg = CLASS_TO_REG[y9]

        out[(pid, tid)] = {
            "X":     X,
            "y9":    y9,
            "y_reg": y_reg,
            "level": level,
            "T":     int(T_target),
            "fs":    fs,
        }
        class_counts[level] += 1

    # Diagnostics.
    if verbose:
        print(f"\n=== Distribution par classe ({len(out)} trials) ===")
        for lvl in LEVEL9_ORDER:
            c = class_counts.get(lvl, 0)
            tag = ""
            idx = LEVEL_TO_IDX[lvl]
            if idx in (0, 8):
                tag = "  ← TRAIN extrêmes"
            elif 1 <= idx <= 7:
                tag = "  ← VAL aveugle"
            print(f"  {lvl:20s} (class {idx})  : {c:3d} trials  y_reg={CLASS_TO_REG[idx]:+.2f}{tag}")

        Ts = [v["T"] for v in out.values()]
        fss = [v["fs"] for v in out.values()]
        print(f"\nLongueurs T   : min={min(Ts)}, médiane={int(np.median(Ts))}, max={max(Ts)}")
        print(f"Fréquences fs : min={min(fss):.2f}, médiane={np.median(fss):.2f}, max={max(fss):.2f} Hz")

    # Sauvegarder.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(out, f)

    if verbose:
        size_mb = output_path.stat().st_size / 1e6
        print(f"\n✅ Sauvegardé : {output_path}  ({size_mb:.1f} MB, {len(out)} trials)")

    return out


# ── CLI ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",  type=Path, default=Path("data/filtered_data.json"))
    parser.add_argument("--output", type=Path, default=Path("data/continuous_per_trial.pkl"))
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()

    build_continuous_dataset(args.input, args.output, verbose=not args.quiet)


if __name__ == "__main__":
    main()
