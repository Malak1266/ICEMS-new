"""
build_mae_dataset_enrichi.py
============================
Construit X_pretrain_4ch.npy en fusionnant toutes les sessions Atracsys :
  - 2026-04-24  (move / rotate / static — pipeline_output)
  - 2026-04-30  (cycle — pipeline_output_v3)
  - 2026-05-06  (cycle — pipeline_output_new)
  - 2026-05-13  (slow / hesitant / continuous — pipeline_output_new13)

Chaque séquence est découpée en fenêtres glissantes de WIN_LEN=32 frames.
Les features brutes (T, 6) proviennent des fichiers features_6ch.npy
déjà produits par tracking_hungarian.py, dans l'ordre :
    [velocity, accel, jerk, spread, axis_angle, valid_ratio]

⚠️  Décision post-réunion (cf. PROJECT_PROGRESSION.md §2.2) :
    On conserve uniquement 4 canaux invariants à la pose caméra :
        [velocity, accel, jerk, valid_ratio]
    `spread` (constante physique : sphères vissées rigidement) et
    `axis_angle` (non-invariant + invalide quand N_SPHERES=1) sont supprimés
    via slicing `KEEP_IDX = [0, 1, 2, 5]` — voir constante ci-dessous.

Normalisation globale : z-score sur les 3 canaux cinématiques uniquement.
Le canal `valid_ratio` reste dans [0, 1] et n'est PAS normalisé.
Les fenêtres avec trop de frames invalides sont écartées.

Usage (Windows) :
    python build_mae_dataset_enrichi.py --out C:/ICEMS/data/

Usage (Narval) :
    python build_mae_dataset_enrichi.py --out ~/icems/data/

Sorties :
    X_pretrain_4ch.npy      (N, 32, 4)  float32
    norm_params_4ch.npy     (2, 4)       [mean, std]
    meta_4ch.csv            N lignes     métadonnées par fenêtre
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── Sélection des features (Tâche B Étage 1) ───────────────────────────────
# Indices dans le tenseur d'origine (T, 6) :
#   0 : velocity      ✓ conservé
#   1 : accel         ✓ conservé
#   2 : jerk          ✓ conservé
#   3 : spread        ✗ SUPPRIMÉ (constante physique → variance nulle)
#   4 : axis_angle    ✗ SUPPRIMÉ (non-invariant au repère + invalide si N_SPHERES=1)
#   5 : valid_ratio   ✓ conservé (re-positionné à l'index 3 du tenseur final)
KEEP_IDX            = [0, 1, 2, 5]
FEATURE_NAMES       = ["velocity", "accel", "jerk", "valid_ratio"]
N_FEATURES          = len(KEEP_IDX)               # = 4
KIN_SLICE           = slice(0, 3)                  # 3 canaux cinématiques à normaliser
VALID_RATIO_IDX_NEW = 3                            # nouvel index du valid_ratio (post-slicing)
VALID_RATIO_IDX_RAW = 5                            # index dans le tenseur 6ch source

# ── Paramètres ─────────────────────────────────────────────────────────────
WIN_LEN      = 32       # longueur de fenêtre (frames) — voir Tâche G pour la migration future
STRIDE       = 16       # pas glissant (50% overlap)
MIN_VALID    = 0.10     # fraction minimale de frames valides par fenêtre
MIN_FRAMES   = 50       # séquences trop courtes → ignorées

INSTRUMENTS_LEGACY = ["scissors", "bipolar", "aspirator"]
INSTRUMENTS_GENERIC = ["generic"]
MOTIONS_V1   = ["move", "rotate", "static"]
MOTIONS_NEW  = ["cycle"]

# ── Sources de données ──────────────────────────────────────────────────────
def get_sources(base: Path):
    """Retourne la liste des dossiers pipeline_output à scanner."""
    return [
        # Session 2026-05-25 (1 rigid body générique, 4 sphères, ~54 Hz)
        (base / "pipeline_output_2026-05-25", "2026-05-25", None),
        # Session 2026-04-24 (move/rotate/static)
        (base / "pipeline_output",     "2026-04-24", MOTIONS_V1),
        # Session 2026-04-30 (cycle V3)
        (base / "pipeline_output_v3",  "2026-04-30", MOTIONS_NEW),
        # Session 2026-05-06 (cycle nouveau)
        (base / "pipeline_output_new", "2026-05-06", MOTIONS_NEW),
        # Session 2026-05-13 (slow/hesitant/continuous)
        (base / "pipeline_output_new13", "2026-05-13", ["slow", "hesitant", "continuous"]),
    ]


def _instruments_in(pipeline_dir: Path):
    """Instruments présents dans un dossier pipeline_output."""
    found = []
    for instr in INSTRUMENTS_GENERIC + INSTRUMENTS_LEGACY:
        if (pipeline_dir / instr).is_dir():
            found.append(instr)
    return found


def _motions_in(instr_dir: Path, motions):
    """Liste de motions à scanner (None = auto-découverte)."""
    if motions is None:
        return sorted(
            p.name for p in instr_dir.iterdir()
            if p.is_dir() and not p.name.startswith("_")
        )
    return motions


def load_features(pipeline_dir: Path, motions: list, session: str):
    """Charge tous les features_6ch.npy d'un dossier pipeline."""
    records = []
    for instr in _instruments_in(pipeline_dir):
        instr_dir = pipeline_dir / instr
        for motion in _motions_in(instr_dir, motions):
            motion_dir = instr_dir / motion
            if not motion_dir.is_dir():
                continue
            for rep_dir in sorted(motion_dir.iterdir()):
                if not rep_dir.is_dir() or not rep_dir.name.startswith("R"):
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
    Découpe une séquence (T, 6) en fenêtres (win_len, N_FEATURES).

    Pipeline :
      1. Calcul du valid_ratio sur le tenseur 6ch d'origine (index VALID_RATIO_IDX_RAW = 5)
         pour filtrer les fenêtres trop occluses.
      2. Slicing colonne via KEEP_IDX → conserve [velocity, accel, jerk, valid_ratio].

    Retourne array (N_win, win_len, N_FEATURES) ou None si aucune fenêtre retenue.
    """
    T = feat.shape[0]
    windows = []
    for start in range(0, T - win_len + 1, stride):
        w_full = feat[start:start + win_len]              # (win_len, 6)
        valid_ratio = w_full[:, VALID_RATIO_IDX_RAW].mean()
        if valid_ratio >= min_valid:
            w_kept = w_full[:, KEEP_IDX]                   # (win_len, N_FEATURES)
            windows.append(w_kept)
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
    X = np.concatenate(all_windows, axis=0)   # (N, WIN_LEN, N_FEATURES)
    assert X.shape[2] == N_FEATURES, (
        f"Shape inattendue : {X.shape}. Attendu (N, {WIN_LEN}, {N_FEATURES}). "
        f"Vérifier KEEP_IDX et la sortie de sliding_windows()."
    )
    print(f"\n{'='*60}")
    print(f"Total séquences  : {total_seq}")
    print(f"Total fenêtres   : {X.shape[0]}")
    print(f"Shape X          : {X.shape}  (N, WIN_LEN={WIN_LEN}, FEATURES={N_FEATURES})")
    print(f"Features         : {FEATURE_NAMES}")
    print(f"dtype            : {X.dtype}")

    # ── Normalisation z-score sur les canaux cinématiques uniquement ─────────
    # KIN_SLICE = slice(0, 3) → velocity, accel, jerk
    # Le canal valid_ratio (index VALID_RATIO_IDX_NEW = 3) reste dans [0, 1].
    n_kin = KIN_SLICE.stop - KIN_SLICE.start    # = 3
    mean = X[:, :, KIN_SLICE].reshape(-1, n_kin).mean(axis=0)
    std  = X[:, :, KIN_SLICE].reshape(-1, n_kin).std(axis=0)
    std  = np.where(std < 1e-8, 1.0, std)       # éviter division par zéro

    X_norm = X.copy()
    X_norm[:, :, KIN_SLICE] = (X[:, :, KIN_SLICE] - mean) / std

    # Vérification
    kin_names = FEATURE_NAMES[KIN_SLICE]
    print(f"\nNormalisation (canaux cinématiques) :")
    for i, name in enumerate(kin_names):
        print(f"  {name:12s} : mean={mean[i]:8.3f}  std={std[i]:8.3f}")
    print(f"  {FEATURE_NAMES[VALID_RATIO_IDX_NEW]:12s} : non normalisé (reste dans [0, 1])")

    check_mean = X_norm[:, :, KIN_SLICE].reshape(-1, n_kin).mean(axis=0)
    check_std  = X_norm[:, :, KIN_SLICE].reshape(-1, n_kin).std(axis=0)
    print(f"\nVérification après normalisation :")
    print(f"  mean ≈ 0  : {np.allclose(check_mean, 0, atol=1e-3)}")
    print(f"  std  ≈ 1  : {np.allclose(check_std,  1, atol=1e-3)}")
    print(f"  NaN       : {np.isnan(X_norm).sum()}")
    print(f"  Inf       : {np.isinf(X_norm).sum()}")

    # ── Sauvegarder ──────────────────────────────────────────────────────────
    # norm_params shape = (2, N_FEATURES) :
    #   ligne 0 : mean (0.0 pour valid_ratio)
    #   ligne 1 : std  (1.0 pour valid_ratio)
    norm_params = np.stack([
        np.append(mean, 0.0),   # mean  : 3 cinématiques + 0 pour valid_ratio
        np.append(std,  1.0),   # std   : 3 cinématiques + 1 pour valid_ratio
    ])

    # Suffixe `_4ch` pour préserver les anciens fichiers v1_enrichi (rollback possible).
    x_path    = out_dir / "X_pretrain_4ch.npy"
    norm_path = out_dir / "norm_params_4ch.npy"
    meta_path = out_dir / "meta_4ch.csv"

    np.save(x_path,    X_norm)
    np.save(norm_path, norm_params)
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)

    size_mb = x_path.stat().st_size / 1e6
    print(f"\n{'='*60}")
    print(f"✅ {x_path.name:30s} : {X_norm.shape}  {size_mb:.1f} MB")
    print(f"✅ {norm_path.name:30s} : {norm_params.shape}")
    print(f"✅ {meta_path.name:30s} : {len(meta_rows)} lignes")
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
