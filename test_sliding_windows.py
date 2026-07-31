"""Test isolé de la fonction extract_sliding_windows."""
import pickle
import numpy as np

N_CONTEXT = 300
HOP       = 50


def extract_sliding_windows(X: np.ndarray, N: int = N_CONTEXT, hop: int = HOP):
    """
    Découpe une séquence (T, F) en fenêtres (N, F) avec un pas `hop`.
    
    Returns:
        windows : ndarray (n_win, N, F)
        start_idx : liste des indices de début de chaque fenêtre
    """
    T = X.shape[0]
    if T < N:
        return None, []   # trial trop court pour extraire une seule fenêtre
    
    starts = list(range(0, T - N + 1, hop))
    windows = np.stack([X[s:s + N] for s in starts], axis=0)
    return windows, starts


with open("data/continuous_per_trial.pkl", "rb") as f:
    dataset = pickle.load(f)

key = next(iter(dataset.keys()))
trial = dataset[key]
print(f"Trial test : participant={key[0]}, trial={key[1]}, level={trial['level']}")
print(f"X shape : {trial['X'].shape}  (T={trial['T']}, F={trial['X'].shape[1]})")

windows, starts = extract_sliding_windows(trial["X"], N=N_CONTEXT, hop=HOP)
print(f"\nFenêtres extraites :")
print(f"  Nombre   : {len(windows)}")
print(f"  Shape    : {windows.shape}")
print(f"  Starts[:5] : {starts[:5]}")
print(f"  Starts[-3:] : {starts[-3:]}")

expected = (trial["T"] - N_CONTEXT) // HOP + 1
print(f"\nVérification :")
print(f"  Formule (T - N) // hop + 1 = {expected}")
print(f"  Match : {len(windows) == expected}")