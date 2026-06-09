"""
step_A_v3_dba_jitter.py
"""
from __future__ import annotations
import argparse, copy, pickle, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from tslearn.barycenters import dtw_barycenter_averaging

CLASS4_NAMES = ["Student", "Junior", "Senior", "Expert"]
Y9_TO_Y4 = {0:0,1:1,2:1,3:1,4:1,5:1,6:2,7:2,8:3}
FEATURE_ROWS = list(range(2,8))
JERK_ROW = 5
LABEL_ROWS = (0,1)
EPSILON = 1e-8
MAX_DBA_FRAMES = 500

def _data_to_array(data):
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"data doit etre 2D (C,T), recu shape={arr.shape}")
    return arr

def _dict_pkl_to_entries(raw):
    entries = []
    for (participant, trial), rec in raw.items():
        X = np.asarray(rec["X"], dtype=np.float64)
        y9 = int(rec["y9"])
        y4 = Y9_TO_Y4[y9]
        expertise = CLASS4_NAMES[y4]
        T = X.shape[0]
        label_exp = np.full(T, y4, dtype=np.float64)
        label_lvl = np.full(T, y9, dtype=np.float64)
        pos_mag = np.linalg.norm(X[:,[0,3,6]], axis=1)
        data = np.vstack([label_exp, label_lvl, pos_mag, X[:,0], X[:,1], X[:,2], X[:,6], X[:,3]])
        entries.append({"name": f"{participant}_{trial}", "participant": str(participant),
            "trial": str(trial), "instrument": rec.get("instrument","bipolar"),
            "expertise": expertise, "level": str(rec.get("level","")),
            "expertise_idx": y4, "level_idx": y9, "is_augmented": False,
            "data": data.tolist()})
    return entries

def load_entries(path):
    with open(path,"rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, dict):
        entries = _dict_pkl_to_entries(raw)
    elif isinstance(raw, list):
        entries = raw
    else:
        raise TypeError(f"Format pkl non supporte : {type(raw)}")
    for e in entries:
        e.setdefault("is_augmented", False)
        e.setdefault("aug_type", None)
    return entries

def extract_features(data):
    return _data_to_array(data)[FEATURE_ROWS, :]

def resample_features(feats, target_len):
    c, t = feats.shape
    if t == target_len: return feats
    if t <= 1: return np.tile(feats,(1,target_len))[:,:target_len]
    x_old = np.linspace(0.,1.,t)
    x_new = np.linspace(0.,1.,target_len)
    out = np.zeros((c,target_len), dtype=np.float64)
    from scipy.interpolate import interp1d
    for i in range(c):
        out[i,:] = interp1d(x_old, feats[i,:], kind="linear", fill_value="extrapolate")(x_new)
    return out

def pad_feature_group(seqs):
    t_target = min(min(s.shape[1] for s in seqs), MAX_DBA_FRAMES)
    aligned = [resample_features(s, t_target) for s in seqs]
    padded = np.zeros((len(aligned), t_target, aligned[0].shape[0]), dtype=np.float64)
    for i, seq in enumerate(aligned):
        padded[i,:,:] = seq.T
    return padded, t_target

def run_dba_on_group(parent_features, max_iter=30):
    group_array, _ = pad_feature_group(parent_features)
    bary = dtw_barycenter_averaging(group_array, max_iter=max_iter)
    return bary.T

def rebuild_data_matrix(parent_data, feature_block):
    t_bary = feature_block.shape[1]
    out = np.zeros((parent_data.shape[0], t_bary), dtype=np.float64)
    for row in LABEL_ROWS:
        out[row,:] = parent_data[row,0] if parent_data.shape[1] > 0 else 0.
    out[FEATURE_ROWS,:] = feature_block
    return out

def make_augmented_entry(template, data_matrix, aug_type, name_suffix, dba_parents=None):
    entry = copy.deepcopy(template)
    entry["data"] = data_matrix.tolist()
    entry["is_augmented"] = True
    entry["aug_type"] = aug_type
    entry["name"] = f"{template['name']}_{name_suffix}"
    if dba_parents is not None:
        entry["dba_parents"] = list(dba_parents)
    return entry

def apply_jitter(data, alpha, rng):
    out = data.copy()
    for row in FEATURE_ROWS:
        channel = out[row,:]
        sigma = alpha * (np.std(channel) + EPSILON)
        out[row,:] = channel + rng.normal(0., sigma, size=channel.shape)
    out[LABEL_ROWS,:] = data[LABEL_ROWS,:]
    return out

def compute_rugosity(data):
    feats = extract_features(data)
    return float(np.mean([np.std(feats[i,:]) + EPSILON for i in range(feats.shape[0])]))

def generate_dba_entries(real_entries, n_parents, n_dba_per_class, seed):
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for e in real_entries:
        by_class[str(e["expertise"])].append(e)
    dba_entries = []
    for class_name in CLASS4_NAMES:
        pool = by_class.get(class_name, [])
        if len(pool) < n_parents:
            print(f"Classe {class_name} : seulement {len(pool)} trials — DBA ignore.")
            continue
        for j in range(n_dba_per_class):
            parents = list(rng.choice(pool, size=n_parents, replace=False))
            parent_arrays = [_data_to_array(p["data"]) for p in parents]
            parent_feats = [extract_features(p["data"]) for p in parents]
            bary_feats = run_dba_on_group(parent_feats)
            full = rebuild_data_matrix(parent_arrays[0], bary_feats)
            dba_entries.append(make_augmented_entry(parents[0], full, aug_type="dba",
                name_suffix=f"dba_{class_name.lower()}_{j}",
                dba_parents=[p["name"] for p in parents]))
    return dba_entries

def generate_jitter_entries(dba_entries, alpha, seed):
    rng = np.random.default_rng(seed+1)
    jitter_entries = []
    for i, dba_entry in enumerate(dba_entries):
        data = _data_to_array(dba_entry["data"])
        jittered = apply_jitter(data, alpha, rng)
        jitter_entries.append(make_augmented_entry(dba_entry, jittered,
            aug_type="dba+jitter", name_suffix=f"jitter_{i}"))
    return jitter_entries

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", default="data/continuous_per_trial.pkl")
    parser.add_argument("--out", dest="output_path", default="data/augmented_v4.pkl")
    parser.add_argument("--n_parents", type=int, default=6)
    parser.add_argument("--n_dba_per_class", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    input_path = root / args.input_path
    output_path = root / args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Chargement : {input_path}")
    real_entries = load_entries(input_path)
    real_entries = [e for e in real_entries if not e.get("is_augmented", False)]
    print(f"  -> {len(real_entries)} trials reels")

    dba_entries = generate_dba_entries(real_entries,
        n_parents=args.n_parents, n_dba_per_class=args.n_dba_per_class, seed=args.seed)
    jitter_entries = generate_jitter_entries(dba_entries, alpha=args.alpha, seed=args.seed)
    all_entries = real_entries + dba_entries + jitter_entries

    with open(output_path,"wb") as f:
        pickle.dump(all_entries, f)
    print(f"Sauvegarde : {output_path} ({len(all_entries)} entries)")

    real_rug = defaultdict(list)
    dba_rug = defaultdict(list)
    for e in real_entries: real_rug[e["expertise"]].append(compute_rugosity(e["data"]))
    for e in dba_entries: dba_rug[e["expertise"]].append(compute_rugosity(e["data"]))

    print("\nValidation rugosité :")
    for c in CLASS4_NAMES:
        r = float(np.median(real_rug[c])) if real_rug[c] else float("nan")
        d = float(np.median(dba_rug[c])) if dba_rug[c] else float("nan")
        print(f"  {c:<8}: reel={r:.4f}  dba={d:.4f}")

    print(f"\nTotal : {len(real_entries)} reels + {len(dba_entries)} DBA + {len(jitter_entries)} jitter = {len(all_entries)}")

if __name__ == "__main__":
    main()
