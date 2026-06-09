"""Test isolé : construction du dataset windows-level + sample_weight balanced."""
import pickle
import numpy as np
from collections import Counter

N_CONTEXT = 300
HOP       = 50


def extract_sliding_windows(X, N=N_CONTEXT, hop=HOP):
    T = X.shape[0]
    if T < N:
        return None, []
    starts = list(range(0, T - N + 1, hop))
    windows = np.stack([X[s:s + N] for s in starts], axis=0)
    return windows, starts


def build_window_dataset(trials_dict, N=N_CONTEXT, hop=HOP):
    Xs, ys, trial_keys, y9s = [], [], [], []
    for key, rec in trials_dict.items():
        windows, _ = extract_sliding_windows(rec["X"], N, hop)
        if windows is None:
            continue
        n_win = windows.shape[0]
        Xs.append(windows)
        ys.append(np.full(n_win, rec["y_reg"], dtype=np.float32))
        y9s.append(np.full(n_win, rec["y9"], dtype=np.int32))
        trial_keys.extend([key] * n_win)
    X_windows = np.concatenate(Xs, axis=0)
    y_windows = np.concatenate(ys, axis=0)
    y9_windows = np.concatenate(y9s, axis=0)
    return X_windows, y_windows, y9_windows, trial_keys


def compute_balanced_sample_weights(y9_windows):
    """sample_weight balanced à la sklearn : w_c = N / (n_classes * n_c)."""
    counts = Counter(y9_windows.tolist())
    n_total = len(y9_windows)
    n_classes = len(counts)
    weights_per_class = {c: n_total / (n_classes * n_c) for c, n_c in counts.items()}
    sample_weights = np.array([weights_per_class[int(y)] for y in y9_windows], dtype=np.float32)
    return sample_weights, weights_per_class


with open("data/continuous_per_trial.pkl", "rb") as f:
    dataset = pickle.load(f)

train_set = {k: v for k, v in dataset.items() if v["y9"] in (0, 8)}
print(f"Train set : {len(train_set)} trials")

X_w, y_w, y9_w, keys = build_window_dataset(train_set)
print(f"\nWindows dataset :")
print(f"  X_windows.shape  : {X_w.shape}")
print(f"  y_windows.shape  : {y_w.shape}")
print(f"  Nombre de keys   : {len(keys)} (doit être == X_w.shape[0])")

print(f"\nDistribution des classes par fenêtre :")
for c, n in Counter(y9_w.tolist()).items():
    print(f"  Class {c} : {n:5d} fenêtres")

sw, wpc = compute_balanced_sample_weights(y9_w)
print(f"\nsample_weight 'balanced' (à la sklearn) :")
for c, w in wpc.items():
    print(f"  Class {c} : w = {w:.4f}")
print(f"\nSomme totale pondérée par classe (vérification équilibrage) :")
for c in wpc:
    mask = y9_w == c
    print(f"  Class {c} : n_windows={mask.sum():5d}  ×  w={wpc[c]:.4f}  =  {mask.sum() * wpc[c]:.1f}")