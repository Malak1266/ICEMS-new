"""
Augmentation globale V3.
3 techniques : DBA + Jitter + Time Warping.

Ratios par classe :
Student  : DBA×2 + Jitter×2 + TimeWarp×1 → ~210 trials
Junior   : DBA×2 + Jitter×2 + TimeWarp×1 → ~205 trials
Senior   : DBA×3 + Jitter×2 + TimeWarp×2 → ~217 trials
Expert   : DBA×4 + Jitter×3 + TimeWarp×2 → ~220 trials

Règle anti-fuite : strictement dans le train fold.
PGY3/PGY4 : augmentés pour l'entraînement uniquement
(pas pour les métriques ni les graphes post-hoc).
"""

import numpy as np
from tslearn.barycenters import dtw_barycenter_averaging

AUGMENTATION_RATIOS_V3 = {
    0: {"dba": 2, "jitter": 2, "timewarp": 1},  # Student
    1: {"dba": 2, "jitter": 2, "timewarp": 1},  # Junior
    2: {"dba": 3, "jitter": 2, "timewarp": 2},  # Senior
    3: {"dba": 4, "jitter": 3, "timewarp": 2},  # Expert
}

MAX_DBA_FRAMES = 500


def _resample_sequence(x: np.ndarray, target_len: int) -> np.ndarray:
    """Rééchantillonne (T, F) vers target_len."""
    from scipy.interpolate import interp1d

    T, F = x.shape
    if T == target_len:
        return x
    if T <= 1:
        return np.tile(x, (target_len, 1))[:target_len]
    t_old = np.linspace(0.0, 1.0, T)
    t_new = np.linspace(0.0, 1.0, target_len)
    out = np.zeros((target_len, F), dtype=np.float64)
    for f in range(F):
        out[:, f] = interp1d(t_old, x[:, f], kind="linear", fill_value="extrapolate")(t_new)
    return out


def jitter(x, sigma=0.02):
    return x + np.random.normal(0, sigma, x.shape)


def time_warp(x, warp_factor_range=(0.85, 1.15)):
    """Déformation non-linéaire de l'axe temporel."""
    from scipy.interpolate import interp1d

    T, F = x.shape
    warp = np.random.uniform(*warp_factor_range)
    new_T = max(int(T * warp), 10)
    t_orig = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, new_T)
    warped = np.zeros((new_T, F))
    for f in range(F):
        warped[:, f] = interp1d(t_orig, x[:, f])(t_new)
    t_back = np.linspace(0, 1, new_T)
    result = np.zeros((T, F))
    for f in range(F):
        result[:, f] = interp1d(t_back, warped[:, f])(t_orig)
    return result


def dba_augment(sequences, n_samples):
    """Génère n_samples via DBA sur les séquences fournies."""
    results = []
    for _ in range(n_samples):
        n_pick = min(len(sequences), np.random.randint(3, 6))
        picked = [sequences[i] for i in np.random.choice(len(sequences), n_pick, replace=False)]
        t_target = min(min(s.shape[0] for s in picked), MAX_DBA_FRAMES)
        aligned = [_resample_sequence(s, t_target) for s in picked]
        group = np.stack(aligned, axis=0)  # [N, T, F]
        bary = dtw_barycenter_averaging(group, max_iter=5)  # [T, F]
        results.append(bary.astype(np.float32))
    return results


def _synth_record(seq, cls, ref_trial, source_pids, aug_type):
    return {
        "X": seq,
        "y4": cls,
        "y_reg": ref_trial["y_reg"],
        "y9": ref_trial.get("y9", cls),
        "aug_source_participants": [str(p) for p in source_pids],
        "is_augmented": True,
        "aug_type": aug_type,
    }


def augment_fold_v3(train_trials, fold_i, seed=42, fast_mode=False):
    """
    Augmentation globale V3 pour un fold LOPO.
    Retourne un dict de trials synthétiques.

    fast_mode=True : compteurs réduits pour smoke test (1 DBA, 2 jitter, 1 timewarp/classe).
    """
    np.random.seed(seed + fold_i)

    by_class = {0: [], 1: [], 2: [], 3: []}
    for key, trial in train_trials.items():
        y4 = trial["y4"]
        if y4 in by_class:
            by_class[y4].append((key, trial))

    synth = {}
    synth_idx = 0

    for cls, items in by_class.items():
        if len(items) == 0:
            continue

        seqs = [it[1]["X"] for it in items]
        keys = [it[0] for it in items]
        ref = items[0][1]
        ratios = AUGMENTATION_RATIOS_V3[cls]
        if fast_mode:
            ratios = {"dba": 1, "jitter": 2, "timewarp": 1}
        source_pids = sorted({str(k[0]) if isinstance(k, tuple) else str(k) for k in keys})

        n_dba = min(ratios["dba"] * len(seqs), 3 if fast_mode else ratios["dba"] * len(seqs))
        dba_samples = dba_augment(seqs, n_dba)
        for seq in dba_samples:
            k = f"synth_dba_{cls}_{synth_idx}"
            synth[k] = _synth_record(seq, cls, ref, source_pids, "dba")
            synth_idx += 1

        n_jitter = min(ratios["jitter"] * len(seqs), 4 if fast_mode else ratios["jitter"] * len(seqs))
        for _ in range(n_jitter):
            src = seqs[np.random.randint(len(seqs))]
            src_key = keys[np.random.randint(len(keys))]
            pid = src_key[0] if isinstance(src_key, tuple) else str(src_key)
            k = f"synth_jitter_{cls}_{synth_idx}"
            synth[k] = _synth_record(jitter(src), cls, ref, [pid], "jitter")
            synth_idx += 1

        n_twarp = min(ratios["timewarp"] * len(seqs), 2 if fast_mode else ratios["timewarp"] * len(seqs))
        for _ in range(n_twarp):
            src = seqs[np.random.randint(len(seqs))]
            src_key = keys[np.random.randint(len(keys))]
            pid = src_key[0] if isinstance(src_key, tuple) else str(src_key)
            k = f"synth_twarp_{cls}_{synth_idx}"
            synth[k] = _synth_record(time_warp(src), cls, ref, [pid], "timewarp")
            synth_idx += 1

    return synth
