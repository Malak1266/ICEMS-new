"""
convert_srytrack.py
===================
Convertit les fichiers CSV exportés par SryTrack GUI v1.14.1
en format fiducials_raw.csv + fiducials_clean.csv compatibles
avec tracking_hungarian.py.

Usage :
    # Convertir une séquence R01
    python pipeline/convert_srytrack.py --seq dataset/2026-05-06/scissors/cycle/R01

    # Convertir toute une session
    python pipeline/convert_srytrack.py --all dataset/2026-05-06
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paramètres de nettoyage (identiques à tracking_hungarian.py) ──────────
PROB_MIN     = 0.9
TRI_ERR_MAX  = 15.0
COORD_MAX    = 800.0   # |x|,|y|,|z| max en mm


def convert_one(seq_path: str):
    """
    Cherche le fichier Export_*_Fiducial.csv dans seq_path,
    le convertit en fiducials_raw.csv + fiducials_clean.csv.
    """
    seq = Path(seq_path)

    # ── Trouver le fichier Fiducial exporté par SryTrack ──────────────────
    candidates = list(seq.glob("*_Fiducial.csv"))
    if not candidates:
        print(f"  ✗ Aucun fichier *_Fiducial.csv dans {seq}")
        return False

    src = candidates[0]
    print(f"\n{'='*60}")
    print(f"Conversion : {src.name}")

    # ── Lire le CSV SryTrack (séparateur ;) ───────────────────────────────
    df = pd.read_csv(src, sep=';')

    # ── Renommer les colonnes vers l'ancien format ─────────────────────────
    df = df.rename(columns={
        'CoordX':        'x',
        'CoordY':        'y',
        'CoordZ':        'z',
        'Triangulation': 'triangulation_error',
        'Epipolar':      'epipolar_error',
        'Probability':   'probability',
        'Index':         'fiducial_idx',
    })

    # ── Convertir Timestamp hex → frame_idx entier ────────────────────────
    timestamps_sorted = sorted(df['Timestamp'].unique())
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps_sorted)}
    df['frame_idx'] = df['Timestamp'].map(ts_to_idx)

    # ── Convertir timestamp hex → nanosecondes (t_pc_ns) ─────────────────
    df['t_pc_ns'] = df['Timestamp'].apply(lambda x: int(x, 16))

    # ── Ajouter colonnes manquantes ────────────────────────────────────────
    df['n_fiducials'] = df.groupby('frame_idx')['fiducial_idx'].transform('count')
    df['index']       = df['fiducial_idx']
    df['valid']       = True
    df['status']      = 'ok'

    # ── Colonnes finales dans le bon ordre ────────────────────────────────
    cols = ['frame_idx', 't_pc_ns', 'fiducial_idx', 'x', 'y', 'z',
            'n_fiducials', 'index', 'valid', 'probability',
            'triangulation_error', 'epipolar_error', 'status']
    df_raw = df[cols].copy()

    # ── Sauvegarder fiducials_raw.csv ─────────────────────────────────────
    raw_path = seq / "fiducials_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    print(f"  ✓ fiducials_raw.csv  ({len(df_raw)} lignes, {df_raw['frame_idx'].nunique()} frames)")

    # ── Appliquer le filtre clean ──────────────────────────────────────────
    mask = (
        (df_raw['probability']          >= PROB_MIN)     &
        (df_raw['triangulation_error']  <  TRI_ERR_MAX)  &
        (df_raw['x'].abs()              <  COORD_MAX)     &
        (df_raw['y'].abs()              <  COORD_MAX)     &
        (df_raw['z'].abs()              <  COORD_MAX)
    )
    df_clean = df_raw[mask].copy().reset_index(drop=True)

    # Ajouter colonnes supplémentaires du format clean
    df_clean['track_rank']       = 0
    df_clean['n_fiducials_raw']  = df_clean.groupby('frame_idx')['fiducial_idx'].transform('count')
    df_clean['n_fiducials_kept'] = df_clean.groupby('frame_idx')['fiducial_idx'].transform('count')

    # ── Sauvegarder fiducials_clean.csv ───────────────────────────────────
    clean_path = seq / "fiducials_clean.csv"
    df_clean.to_csv(clean_path, index=False)

    pct_kept = 100 * len(df_clean) / len(df_raw) if len(df_raw) > 0 else 0
    print(f"  ✓ fiducials_clean.csv ({len(df_clean)} lignes propres, {pct_kept:.1f}% conservées)")

    # ── Écrire meta.txt ───────────────────────────────────────────────────
    n_frames  = df_raw['frame_idx'].nunique()
    fps_obs   = n_frames / max(1, (df_raw['t_pc_ns'].max() - df_raw['t_pc_ns'].min()) / 1e9)
    meta_path = seq / "meta.txt"
    with open(meta_path, 'w') as f:
        f.write(f"session_dir={seq}\n")
        f.write(f"source_file={src.name}\n")
        f.write(f"frames={n_frames}\n")
        f.write(f"fps_observed={fps_obs:.2f}\n")
        f.write(f"raw_points={len(df_raw)}\n")
        f.write(f"clean_points={len(df_clean)}\n")
        f.write(f"prob_min={PROB_MIN}\n")
        f.write(f"tri_err_max={TRI_ERR_MAX}\n")
        f.write(f"coord_max={COORD_MAX}\n")
    print(f"  ✓ meta.txt écrit ({n_frames} frames, {fps_obs:.1f} fps)")

    return True


def convert_all(dataset_path: str):
    """Convertit toutes les séquences d'une session.

    Découverte dynamique :
      - instruments : tous les dossiers enfants de `dataset_path` (incluant
                      `generic` pour la nouvelle collecte 1-rigid-body)
      - motions    : tous les dossiers enfants de chaque instrument (catalogue
                     libre : translation_slow, figure_8, suture_pattern, etc.)
    """
    base = Path(dataset_path)
    total = 0
    ok    = 0

    for instr_dir in sorted(base.iterdir()):
        if not instr_dir.is_dir() or instr_dir.name.startswith('_'):
            continue
        for motion_dir in sorted(instr_dir.iterdir()):
            if not motion_dir.is_dir() or motion_dir.name.startswith('_'):
                continue
            for rep in sorted(motion_dir.iterdir()):
                if rep.is_dir() and rep.name.startswith('R'):
                    total += 1
                    if convert_one(str(rep)):
                        ok += 1

    print(f"\n{'='*60}")
    print(f"Conversion terminée : {ok}/{total} séquences converties")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', help='Chemin d\'une séquence unique')
    parser.add_argument('--all', help='Chemin d\'une session complète')
    args = parser.parse_args()

    if args.seq:
        convert_one(args.seq)
    elif args.all:
        convert_all(args.all)
    else:
        print("Usage : --seq <chemin>  ou  --all <session>")
