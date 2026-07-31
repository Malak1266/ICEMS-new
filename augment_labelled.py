"""
augment_labelled.py
==================
P1 — Augmentation de données sur les trials LABELLISÉS (Student..Staff).

⚠️  N'inclut QUE les données labellisées (continuous_per_trial.pkl).
    NE PAS y mettre ta collecte Atracsys personnelle : elle n'a pas de niveau
    d'expertise → elle ne sert qu'au MAE (auto-supervisé), pas au scoring.

Transformations réalistes appliquées aux séries temporelles cinématiques :
    - jitter     : bruit gaussien proportionnel à l'écart-type du canal
    - scaling    : facteur d'amplitude global (gestes plus amples / plus fins)
    - time-warp  : ré-échantillonnage (geste légèrement plus rapide / plus lent)
    - time-mask  : occlusion temporelle simulée (segment mis à 0 + valid_ratio=0)

Le canal valid_ratio (dernière colonne) n'est PAS bruité/scalé ; il est mis à 0
uniquement dans les segments masqués (occlusion réaliste).

La distribution des classes est PRÉSERVÉE : on applique le même facteur à toutes.

Usage (depuis ICEMS-main) :
    python augment_labelled.py --target 7500
    python augment_labelled.py --target 8000 --out data/continuous_per_trial_aug.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

N_CONTEXT = 300
HOP = 50
VALID_RATIO_COL = -1  # dernière colonne


def n_windows(X, N=N_CONTEXT, hop=HOP):
    T = X.shape[0]
    return 0 if T < N else len(range(0, T - N + 1, hop))


def jitter(X, rng, sigma=0.05):
    out = X.copy()
    kin = out[:, :VALID_RATIO_COL]
    noise = rng.normal(0, 1, kin.shape) * (kin.std(axis=0, keepdims=True) * sigma)
    out[:, :VALID_RATIO_COL] = kin + noise
    return out


def scale(X, rng, lo=0.9, hi=1.1):
    out = X.copy()
    factor = rng.uniform(lo, hi)
    out[:, :VALID_RATIO_COL] *= factor
    return out


def time_warp(X, rng, lo=0.9, hi=1.1):
    T = X.shape[0]
    new_T = max(N_CONTEXT, int(T * rng.uniform(lo, hi)))
    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, new_T)
    out = np.empty((new_T, X.shape[1]), dtype=np.float32)
    for c in range(X.shape[1]):
        out[:, c] = np.interp(new_idx, old_idx, X[:, c])
    return out


def time_mask(X, rng, max_frac=0.15):
    out = X.copy()
    T = X.shape[0]
    seg = int(T * rng.uniform(0.05, max_frac))
    if seg < 1 or seg >= T:
        return out
    start = rng.integers(0, T - seg)
    out[start:start + seg, :VALID_RATIO_COL] = 0.0
    out[start:start + seg, VALID_RATIO_COL] = 0.0  # occlusion → valid_ratio = 0
    return out


def augment_one(X, rng):
    """Applique une combinaison aléatoire de transformations."""
    funcs = [jitter, scale, time_warp, time_mask]
    rng.shuffle(funcs)
    for fn in funcs:
        if rng.random() < 0.6:  # chaque transfo a 60% de chance d'être appliquée
            X = fn(X, rng)
    return X.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="P1 — augmentation des trials labellisés.")
    ap.add_argument("--input", type=Path, default=Path("data/continuous_per_trial.pkl"))
    ap.add_argument("--out", type=Path, default=Path("data/continuous_per_trial_aug.pkl"))
    ap.add_argument("--target", type=int, default=7500,
                    help="Nombre total de fenêtres visé après augmentation.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} introuvable. Lance d'abord : "
            f"python src/build_continuous_dataset.py")

    rng = np.random.default_rng(args.seed)
    with open(args.input, "rb") as f:
        dataset = pickle.load(f)

    base_windows = sum(n_windows(v["X"]) for v in dataset.values())
    dist_before = Counter(v["y9"] for v in dataset.values())
    print(f"[Avant] {len(dataset)} trials, {base_windows} fenêtres")

    if base_windows == 0:
        raise RuntimeError("0 fenêtre : trials trop courts pour N_CONTEXT.")

    # Même facteur pour toutes les classes → distribution préservée.
    n_copies = max(0, int(np.ceil(args.target / base_windows)) - 1)
    print(f"[Plan] {n_copies} copie(s) augmentée(s) par trial "
          f"→ ~{base_windows * (n_copies + 1)} fenêtres visées (~{args.target})")

    augmented = dict(dataset)  # garde les originaux
    for (pid, tid), rec in dataset.items():
        for c in range(n_copies):
            X_aug = augment_one(rec["X"], rng)
            augmented[(f"{pid}_aug{c}", tid)] = {**rec, "X": X_aug, "T": X_aug.shape[0]}

    total_windows = sum(n_windows(v["X"]) for v in augmented.values())
    dist_after = Counter(v["y9"] for v in augmented.values())

    print(f"\n[Après] {len(augmented)} trials, {total_windows} fenêtres")
    print("\n[Vérif distribution des classes — proportions préservées]")
    print(f"  {'classe':>6} | {'avant':>6} | {'après':>6} | {'ratio':>6}")
    for c in sorted(dist_before):
        b, a = dist_before[c], dist_after[c]
        print(f"  {c:>6} | {b:>6} | {a:>6} | {a / b:>6.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(augmented, f)
    print(f"\n✅ Dataset augmenté sauvegardé : {args.out}")
    print("   (À utiliser ensuite avec --dataset dans train_long.py / kfold_cv.py.)")


if __name__ == "__main__":
    main()
