"""
clf_model2_augment.py
=====================
Augmentation des données pour le Modèle 2 :
classification par niveau de formation (9 classes y9).

Réutilise les techniques DBA + Jitter + Time Warping de icems_v3_augment.py
avec des ratios recalibrés pour équilibrer les 9 classes.

Distribution réelle du dataset (136 trials) :
    ms(0)    : 42   pgy1(1) : 14   pgy2(2) :  9
    pgy3(3)  :  6   pgy4(4) :  3   pgy5(5) :  9
    pgy6(6)  : 11   fellow(7): 20  staff(8) : 22

Ratios définis pour atteindre ~42 trials par classe après augmentation.

RÈGLE ANTI-FUITE (LOPO) :
    Pour une validation LOPO correcte, appeler augment_model2_fold()
    UNIQUEMENT sur les trials du train fold — jamais sur le fold de test.
    Le dataset global clf_model2_aug.pkl (généré par build_clf_datasets.py)
    est fourni à titre d'exploration uniquement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Assure que src/ est dans le path (compatible run direct et run depuis la racine)
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from icems_v3_augment import dba_augment, jitter, time_warp

# ─── Noms canoniques des 9 classes ────────────────────────────────────────────
Y9_NAMES: list[str] = ["ms", "pgy1", "pgy2", "pgy3", "pgy4", "pgy5", "pgy6", "fellow", "staff"]

# ─── Ratios d'augmentation par classe y9 ──────────────────────────────────────
# Format : {"dba": n_dba_per_trial, "jitter": n_jit_per_trial, "timewarp": n_tw_per_trial}
# Exemple : dba=2 avec 6 trials → 12 échantillons DBA générés.
#
# Cibles approximatives après augmentation :
#   ms    : 42  (aucune aug. — classe de référence)
#   pgy1  : 14 + 14×2  = ~42
#   pgy2  :  9 +  9×4  = ~45
#   pgy3  :  6 +  6×6  = ~42
#   pgy4  :  3 +  3×13 = ~42  ← classe la plus rare, augmentation agressive
#   pgy5  :  9 +  9×4  = ~45
#   pgy6  : 11 + 11×3  = ~44
#   fellow: 20 + 20×1  = ~40
#   staff : 22 + 22×1  = ~44
AUG_RATIOS_MODEL2: dict[int, dict[str, int]] = {
    0: {"dba": 0, "jitter": 0, "timewarp": 0},   # ms     → pas d'augmentation
    1: {"dba": 0, "jitter": 1, "timewarp": 1},   # pgy1   → +2 par trial
    2: {"dba": 1, "jitter": 2, "timewarp": 1},   # pgy2   → +4 par trial
    3: {"dba": 2, "jitter": 3, "timewarp": 1},   # pgy3   → +6 par trial
    4: {"dba": 3, "jitter": 7, "timewarp": 3},   # pgy4   → +13 par trial (aggressif)
    5: {"dba": 1, "jitter": 2, "timewarp": 1},   # pgy5   → +4 par trial
    6: {"dba": 1, "jitter": 2, "timewarp": 0},   # pgy6   → +3 par trial
    7: {"dba": 0, "jitter": 1, "timewarp": 0},   # fellow → +1 par trial
    8: {"dba": 0, "jitter": 1, "timewarp": 0},   # staff  → +1 par trial
}


def _synth_record(
    seq: np.ndarray,
    y9: int,
    y_reg: float,
    source_pids: list[str],
    aug_type: str,
) -> dict:
    """Crée un enregistrement synthétique compatible avec le format du dataset."""
    return {
        "X": seq.astype(np.float32),
        "y9": y9,
        "y_reg": y_reg,
        "is_augmented": True,
        "aug_type": aug_type,
        "aug_source_participants": source_pids,
    }


def augment_model2_fold(
    train_trials: dict,
    fold_i: int,
    seed: int = 42,
    fast_mode: bool = False,
) -> dict:
    """Augmentation pour un fold LOPO du Modèle 2 (9 classes).

    Génère des échantillons synthétiques par DBA, Jitter et Time Warping
    pour équilibrer les classes minoritaires dans le fold d'entraînement.

    Parameters
    ----------
    train_trials : dict
        Sous-ensemble train du dataset.
        Clés : (participant_id, trial_id) ou str.
        Valeurs : dict avec champs ``X`` (np.ndarray), ``y9`` (int), ``y_reg`` (float).
    fold_i : int
        Index du fold courant (reproductibilité via seed + fold_i).
    seed : int
        Graine de base pour le générateur aléatoire.
    fast_mode : bool
        Si True, réduit les compteurs à 1 DBA + 2 jitter + 1 timewarp par classe
        (smoke test rapide).

    Returns
    -------
    dict
        Trials synthétiques (même format que train_trials) à fusionner avec
        le fold d'entraînement. Les clés sont des str de la forme
        ``"synth_<type>_<cls>_<idx>"``.
    """
    np.random.seed(seed + fold_i)

    by_class: dict[int, list[tuple]] = {c: [] for c in range(9)}
    for key, trial in train_trials.items():
        y9 = int(trial["y9"])
        if 0 <= y9 <= 8:
            by_class[y9].append((key, trial))

    synth: dict = {}
    synth_idx = 0

    for cls, items in by_class.items():
        if not items:
            continue

        ratios = AUG_RATIOS_MODEL2[cls]
        if fast_mode:
            ratios = {"dba": min(1, ratios["dba"]), "jitter": 2, "timewarp": min(1, ratios["timewarp"])}

        if all(v == 0 for v in ratios.values()):
            continue

        seqs = [it[1]["X"] for it in items]
        keys = [it[0] for it in items]
        ref_y_reg = float(items[0][1]["y_reg"])
        source_pids = sorted(
            {str(k[0]) if isinstance(k, tuple) else str(k) for k in keys}
        )

        # DBA — nécessite au moins 2 séquences distinctes
        n_dba = ratios["dba"] * len(seqs)
        if n_dba > 0 and len(seqs) >= 2:
            for seq in dba_augment(seqs, n_dba):
                key_s = f"synth_dba_{cls}_{synth_idx}"
                synth[key_s] = _synth_record(seq, cls, ref_y_reg, source_pids, "dba")
                synth_idx += 1

        # Jitter
        n_jitter = ratios["jitter"] * len(seqs)
        for _ in range(n_jitter):
            idx_src = np.random.randint(len(seqs))
            src = seqs[idx_src]
            pid = keys[idx_src]
            pid = pid[0] if isinstance(pid, tuple) else str(pid)
            key_s = f"synth_jitter_{cls}_{synth_idx}"
            synth[key_s] = _synth_record(jitter(src), cls, ref_y_reg, [str(pid)], "jitter")
            synth_idx += 1

        # Time Warp
        n_twarp = ratios["timewarp"] * len(seqs)
        for _ in range(n_twarp):
            idx_src = np.random.randint(len(seqs))
            src = seqs[idx_src]
            pid = keys[idx_src]
            pid = pid[0] if isinstance(pid, tuple) else str(pid)
            key_s = f"synth_twarp_{cls}_{synth_idx}"
            synth[key_s] = _synth_record(time_warp(src), cls, ref_y_reg, [str(pid)], "timewarp")
            synth_idx += 1

    return synth


def print_augmentation_summary(original: dict, augmented: dict) -> None:
    """Affiche un résumé de la distribution avant et après augmentation."""
    from collections import Counter

    orig_counts = Counter(int(v["y9"]) for v in original.values())
    aug_counts  = Counter(int(v["y9"]) for v in augmented.values())

    print("\n=== Résumé augmentation Modèle 2 ===")
    print(f"{'Classe':<12} {'Avant':>8} {'Synth':>8} {'Après':>8}")
    print("-" * 42)
    for cls in range(9):
        before = orig_counts.get(cls, 0)
        synth  = aug_counts.get(cls, 0)
        after  = before + synth
        print(f"{Y9_NAMES[cls]:<12} {before:>8} {synth:>8} {after:>8}")
    print(f"{'TOTAL':<12} {sum(orig_counts.values()):>8} "
          f"{sum(aug_counts.values()):>8} "
          f"{sum(orig_counts.values()) + sum(aug_counts.values()):>8}")
    print()
