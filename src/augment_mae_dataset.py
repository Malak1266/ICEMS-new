"""
augment_mae_dataset.py
======================
Augmente X_pretrain_clean_4ch.npy pour le pré-entraînement MAE.

Transformations (sur fenêtres déjà normalisées) :
  - jitter gaussien sur vel/acc/jerk (alpha × std local)
  - time-warp léger (0.85–1.15×) via interpolation linéaire

Le canal valid_ratio (index 3) n'est pas jitteré ; il est rééchantillonné avec le warp.

Usage :
    python src/augment_mae_dataset.py --in data/X_pretrain_clean_4ch.npy --factor 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

KIN_SLICE = slice(0, 3)
VALID_IDX = 3
WIN_LEN = 32
EPS = 1e-8
KEEP_IDX_6CH = [0, 1, 2, 5]  # vel, acc, jerk, valid_ratio


def to_4ch(X: np.ndarray) -> np.ndarray:
    """Accepte (N, 32, 4) ou (N, 32, 6) → (N, 32, 4)."""
    if X.shape[-1] == 4:
        return X.astype(np.float32)
    if X.shape[-1] == 6:
        return X[:, :, KEEP_IDX_6CH].astype(np.float32)
    raise ValueError(f"Attendu 4 ou 6 canaux, reçu shape={X.shape}")


def time_warp_window(x: np.ndarray, scale: float) -> np.ndarray:
    """Rééchantillonne (T, 4) vers une longueur WIN_LEN avec facteur scale."""
    t_src = np.linspace(0.0, 1.0, x.shape[0])
    new_len = max(4, int(round(x.shape[0] * scale)))
    t_mid = np.linspace(0.0, 1.0, new_len)
    warped = np.zeros((new_len, x.shape[1]), dtype=np.float32)
    for c in range(x.shape[1]):
        warped[:, c] = np.interp(t_mid, t_src, x[:, c])
    # Ramener à WIN_LEN
    t_out = np.linspace(0.0, 1.0, WIN_LEN)
    t_w = np.linspace(0.0, 1.0, new_len)
    out = np.zeros((WIN_LEN, x.shape[1]), dtype=np.float32)
    for c in range(x.shape[1]):
        out[:, c] = np.interp(t_out, t_w, warped[:, c])
    out[:, VALID_IDX] = np.clip(out[:, VALID_IDX], 0.0, 1.0)
    return out


def jitter_window(x: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    out = x.copy()
    for c in range(KIN_SLICE.stop):
        sigma = alpha * (np.std(out[:, c]) + EPS)
        out[:, c] += rng.normal(0.0, sigma, size=out.shape[0])
    return out


def augment_one(x: np.ndarray, rng: np.random.Generator, alpha: float) -> np.ndarray:
    """Une vue augmentée : jitter puis time-warp aléatoire."""
    y = jitter_window(x, alpha, rng)
    scale = rng.uniform(0.85, 1.15)
    y = time_warp_window(y, scale)
    return y.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Augmente le dataset MAE 4ch.")
    ap.add_argument("--in", dest="in_path", type=Path,
                    default=Path("data/X_pretrain_clean_4ch.npy"))
    ap.add_argument("--out", type=Path, default=Path("data/X_pretrain_aug_4ch.npy"))
    ap.add_argument("--factor", type=int, default=4,
                    help="Multiplicateur total (4 = 1 réel + 3 synthétiques/fenêtre).")
    ap.add_argument("--alpha", type=float, default=0.03, help="Intensité jitter.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.in_path.exists():
        raise FileNotFoundError(f"{args.in_path} introuvable. Lance build_mae_clean_dataset.py.")

    X = to_4ch(np.load(args.in_path))
    if X.ndim != 3 or X.shape[1] != WIN_LEN:
        raise ValueError(f"Attendu (N, {WIN_LEN}, 4|6), reçu {X.shape}")

    n_aug = max(0, args.factor - 1)
    rng = np.random.default_rng(args.seed)

    chunks = [X]
    for aug_i in range(n_aug):
        aug_rng = np.random.default_rng(args.seed + 1000 * (aug_i + 1))
        synth = np.stack([augment_one(X[i], aug_rng, args.alpha) for i in range(len(X))])
        chunks.append(synth)
        print(f"  [aug {aug_i + 1}/{n_aug}] {synth.shape[0]} fenêtres")

    X_out = np.concatenate(chunks, axis=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, X_out)

    print("=" * 60)
    print(f"  Entrée  : {X.shape[0]} fenêtres")
    print(f"  Sortie  : {X_out.shape[0]} fenêtres (×{args.factor})")
    print(f"  ✅ {args.out.resolve()}")


if __name__ == "__main__":
    main()
