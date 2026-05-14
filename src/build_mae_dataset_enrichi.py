"""
build_mae_dataset_enrichi.py
============================
Construit X_pretrain_v1_enrichi.npy en fusionnant toutes les sessions Atracsys :
  - 2026-04-24  (move / rotate / static — pipeline_output)
  - 2026-04-30  (cycle — pipeline_output_v3)
  - 2026-05-06  (cycle — pipeline_output_new)
        # Session 2026-05-13 (slow/hesitant/continuous)
        (base / "pipeline_output_new13", "2026-05-13", ["slow", "hesitant", "continuous"]),

Chaque séquence est découpée en fenêtres glissantes de WIN_LEN=32 frames.
Les features (T, 6) proviennent des fichiers features_6ch.npy déjà produits
par tracking_hungarian.py.

Normalisation globale : z-score par feature (mean=0, std=1).
Les fenêtres avec trop de frames invalides sont écartées.

Usage (Windows) :
    python build_mae_dataset_enrichi.py --out C:/ICEMS/data/

Usage (Narval) :
    python build_mae_dataset_enrichi.py --out ~/icems/data/

Sorties :
    X_pretrain_v1_enrichi.npy   (N, 32, 6)  float32
    norm_params_enrichi.npy     (2, 6)       [mean, std]
    meta_enrichi.csv            N lignes     métadonnées par fenêtre
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paramètres ─────────────────────────────────────────────────────────────
WIN_LEN      = 32       # longueur de fenêtre (frames)
STRIDE       = 16       # pas glissant (50% overlap)
MIN_VALID    = 0.10     # fraction minimale de frames valides par fenêtre
MIN_FRAMES   = 50       # séquences trop courtes → ignorées

INSTRUMENTS  = ["scissors", "bipolar", "aspirator"]
MOTIONS_V1   = ["move", "rotate", "static"]
MOTIONS_NEW  = ["cycle"]

# ── Sources de données ──────────────────────────────────────────────────────
def get_sources(base: Path):
    """Retourne la liste des dossiers pipeline_output à scanner."""
    return [
        # Session 2026-04-24 (move/rotate/static)
        (base / "pipeline_output",     "2026-04-24", MOTIONS_V1),
        # Session 2026-04-30 (cycle V3)
        (base / "pipeline_output_v3",  "2026-04-30", MOTIONS_NEW),
        # Session 2026-05-06 (cycle nouveau)
        (base / "pipeline_output_new", "2026-05-06", MOTIONS_NEW),
        # Session 2026-05-13 (slow/hesitant/continuous)
        (base / "pipeline_output_new13", "2026-05-13", ["slow", "hesitant", "continuous"]),
    ]


def load_features(pipeline_dir: Path, motions: list, session: str):
    """Charge tous les features_6ch.npy d'un dossier pipeline."""
    records = []
    for instr in INSTRUMENTS:
        for motion in motions:
            motion_dir = pipeline_dir / instr / motion
            if not motion_dir.exists():
                continue
            for rep_dir in sorted(motion_dir.iterdir()):
                if not rep_dir.is_dir():
                    continue
                npy_path = rep_dir / "features_6ch.npy"
                if not npy_path.exists():
                    continue
                feat = np.load(npy_path)   # (T, 6)
                if feat.shape[0] < MIN_FRAMES:
                    continue
                records.append({
                    "features": feat,
                    "session":  session,
                    "instrument": instr,
                    "motion":   motion,
                    "rep":      rep_dir.name,
                    "T":        feat.shape[0],
                })
    return records


def sliding_windows(feat: np.ndarray, win_len: int, stride: int, min_valid: float):
    """
    Découpe (T, 6) en fenêtres (win_len, 6).
    La feature index 5 est valid_ratio — on l'utilise pour filtrer.
    Retourne array (N_win, win_len, 6) et masque des fenêtres gardées.
    """
    T = feat.shape[0]
    windows = []
    for start in range(0, T - win_len + 1, stride):
        w = feat[start:start + win_len]          # (win_len, 6)
        valid_ratio = w[:, 5].mean()             # feature 5 = valid_ratio
        if valid_ratio >= min_valid:
            windows.append(w)
    return np.array(windows, dtype=np.float32) if windows else None


def main(args):
    base = Path(args.base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = get_sources(base)

    all_windows = []
    meta_rows   = []

    print("=" * 60)
    print("Chargement des séquences...")
    print("=" * 60)

    total_seq = 0
    for pipeline_dir, session, motions in sources:
        if not pipeline_dir.exists():
            print(f"  ⚠️  Dossier absent : {pipeline_dir}")
            continue

        records = load_features(pipeline_dir, motions, session)
        print(f"\n{session} ({pipeline_dir.name}) : {len(records)} séquences chargées")

        for rec in records:
            wins = sliding_windows(rec["features"], WIN_LEN, STRIDE, MIN_VALID)
            if wins is None or len(wins) == 0:
                continue
            all_windows.append(wins)
            for i in range(len(wins)):
                meta_rows.append({
                    "window_idx":  len(meta_rows),
                    "session":     rec["session"],
                    "instrument":  rec["instrument"],
                    "motion":      rec["motion"],
                    "rep":         rec["rep"],
                    "win_in_seq":  i,
                    "T_seq":       rec["T"],
                })
            total_seq += 1
            print(f"  {rec['instrument']:10s} {rec['motion']:8s} {rec['rep']} "
                  f"→ T={rec['T']:5d}  fenêtres={len(wins)}")

    if not all_windows:
        print("\n❌ Aucune donnée chargée — vérifier les chemins.")
        return

    # ── Concaténer ──────────────────────────────────────────────────────────
    X = np.concatenate(all_windows, axis=0)   # (N, 32, 6)
    print(f"\n{'='*60}")
    print(f"Total séquences  : {total_seq}")
    print(f"Total fenêtres   : {X.shape[0]}")
    print(f"Shape X          : {X.shape}")
    print(f"dtype            : {X.dtype}")

    # ── Normalisation z-score globale par feature ────────────────────────────
    # On normalise features 0-4 (cinématiques) mais pas feature 5 (valid_ratio 0-1)
    mean = X[:, :, :5].reshape(-1, 5).mean(axis=0)
    std  = X[:, :, :5].reshape(-1, 5).std(axis=0)
    std  = np.where(std < 1e-8, 1.0, std)       # éviter division par zéro

    X_norm = X.copy()
    X_norm[:, :, :5] = (X[:, :, :5] - mean) / std

    # Vérification
    print(f"\nNormalisation (features 0-4) :")
    feat_names = ["velocity", "accel", "jerk", "spread", "axis_angle"]
    for i, name in enumerate(feat_names):
        print(f"  {name:12s} : mean={mean[i]:8.3f}  std={std[i]:8.3f}")

    check_mean = X_norm[:, :, :5].reshape(-1, 5).mean(axis=0)
    check_std  = X_norm[:, :, :5].reshape(-1, 5).std(axis=0)
    print(f"\nVérification après normalisation :")
    print(f"  mean ≈ 0  : {np.allclose(check_mean, 0, atol=1e-3)}")
    print(f"  std  ≈ 1  : {np.allclose(check_std,  1, atol=1e-3)}")
    print(f"  NaN       : {np.isnan(X_norm).sum()}")
    print(f"  Inf       : {np.isinf(X_norm).sum()}")

    # ── Sauvegarder ──────────────────────────────────────────────────────────
    norm_params = np.stack([
        np.append(mean, 0.0),    # mean  — feature 5 (valid_ratio) non normalisée
        np.append(std,  1.0),    # std
    ])

    x_path    = out_dir / "X_pretrain_v1_enrichi.npy"
    norm_path = out_dir / "norm_params_enrichi.npy"
    meta_path = out_dir / "meta_enrichi.csv"

    np.save(x_path,    X_norm)
    np.save(norm_path, norm_params)
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)

    size_mb = x_path.stat().st_size / 1e6
    print(f"\n{'='*60}")
    print(f"✅ X_pretrain_v1_enrichi.npy : {X.shape}  {size_mb:.1f} MB")
    print(f"✅ norm_params_enrichi.npy   : {norm_params.shape}")
    print(f"✅ meta_enrichi.csv          : {len(meta_rows)} lignes")
    print(f"\nSauvegardé dans : {out_dir}")

    # ── Distribution par instrument ──────────────────────────────────────────
    df_meta = pd.DataFrame(meta_rows)
    print(f"\nDistribution par instrument :")
    print(df_meta.groupby("instrument").size().to_string())
    print(f"\nDistribution par session :")
    print(df_meta.groupby("session").size().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="C:/ICEMS",
                        help="Racine du projet ICEMS (contient pipeline_output, etc.)")
    parser.add_argument("--out",  default="C:/ICEMS/data",
                        help="Dossier de sortie pour les .npy")
    main(parser.parse_args())
