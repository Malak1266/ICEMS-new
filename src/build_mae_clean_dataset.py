"""
build_mae_clean_dataset.py
============================
Dataset MAE 4ch filtré pour pré-entraînement (sessions propres, qualité élevée).

Filtres appliqués :
  - Sessions sélectionnables (défaut : 2026-05-25 uniquement)
  - valid_ratio fenêtre >= min_valid (défaut 0.30)
  - valid_ratio séquence >= min_seq_valid (défaut 0.30)
  - clip vitesse sur le canal velocity brut (défaut 700 mm/s)

Sorties :
  X_pretrain_clean_4ch.npy
  norm_params_clean_4ch.npy
  meta_clean_4ch.csv

Usage :
    python src/build_mae_clean_dataset.py --base ~/icems --out ~/icems/data
    python src/build_mae_clean_dataset.py --base C:/ICEMS --sessions 2026-05-25 2026-05-13
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from build_mae_dataset_enrichi import (  # noqa: E402
    FEATURE_NAMES,
    KIN_SLICE,
    N_FEATURES,
    VALID_RATIO_IDX_NEW,
    VALID_RATIO_IDX_RAW,
    WIN_LEN,
    get_sources,
    load_features,
    sliding_windows,
)


def clip_velocity(feat6: np.ndarray, max_vel: float) -> np.ndarray:
    """Plafonne velocity (index 0) pour éliminer les artefacts de tracking."""
    out = feat6.copy()
    out[:, 0] = np.clip(out[:, 0], 0.0, max_vel)
    return out


def filter_sources(sources, sessions: list[str] | None):
    if not sessions:
        return sources
    allowed = set(sessions)
    return [(p, s, m) for p, s, m in sources if s in allowed]


def main():
    ap = argparse.ArgumentParser(description="Construit X_pretrain_clean_4ch.npy (filtré).")
    ap.add_argument("--base", default="~/icems", help="Racine ICEMS (pipeline_output*).")
    ap.add_argument("--out", default="~/icems/data", help="Dossier de sortie.")
    ap.add_argument(
        "--sessions",
        nargs="*",
        default=["2026-05-25"],
        help="Sessions à inclure (défaut : 2026-05-25 seule). Vide = toutes.",
    )
    ap.add_argument("--win-len", type=int, default=WIN_LEN)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--min-valid", type=float, default=0.30,
                    help="valid_ratio min par fenêtre.")
    ap.add_argument("--min-seq-valid", type=float, default=0.30,
                    help="valid_ratio moyen min sur la séquence entière.")
    ap.add_argument("--max-velocity", type=float, default=700.0,
                    help="Clip velocity (mm/s) avant fenêtrage.")
    ap.add_argument("--min-frames", type=int, default=50)
    args = ap.parse_args()

    base = Path(args.base).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = args.sessions if args.sessions else None
    sources = filter_sources(get_sources(base), sessions)

    all_windows = []
    meta_rows = []
    n_seq_in, n_seq_out = 0, 0

    print("=" * 60)
    print("  MAE dataset PROPRE (filtré)")
    print("=" * 60)
    print(f"  Sessions     : {sessions or 'toutes'}")
    print(f"  min_valid    : {args.min_valid}")
    print(f"  max_velocity : {args.max_velocity} mm/s")
    print("=" * 60)

    for pipeline_dir, session, motions in sources:
        if not pipeline_dir.exists():
            print(f"  ⚠️  absent : {pipeline_dir}")
            continue
        records = load_features(pipeline_dir, motions, session)
        print(f"\n{session} : {len(records)} séquences brutes")

        for rec in records:
            n_seq_in += 1
            feat = clip_velocity(rec["features"], args.max_velocity)
            seq_valid = float(feat[:, VALID_RATIO_IDX_RAW].mean())
            if seq_valid < args.min_seq_valid:
                continue

            wins = sliding_windows(feat, args.win_len, args.stride, args.min_valid)
            if wins is None or len(wins) == 0:
                continue

            all_windows.append(wins)
            n_seq_out += 1
            for i in range(len(wins)):
                meta_rows.append({
                    "window_idx": len(meta_rows),
                    "session": rec["session"],
                    "instrument": rec["instrument"],
                    "motion": rec["motion"],
                    "rep": rec["rep"],
                    "win_in_seq": i,
                    "T_seq": rec["T"],
                    "seq_valid_ratio": seq_valid,
                })
            print(f"  ✓ {rec['instrument']}/{rec['motion']}/{rec['rep']} "
                  f"valid={seq_valid:.2f} → {len(wins)} fenêtres")

    if not all_windows:
        print("\n❌ Aucune fenêtre — assouplir --min-valid ou vérifier --base.")
        sys.exit(1)

    X = np.concatenate(all_windows, axis=0).astype(np.float32)
    n_kin = KIN_SLICE.stop - KIN_SLICE.start
    mean = X[:, :, KIN_SLICE].reshape(-1, n_kin).mean(axis=0)
    std = X[:, :, KIN_SLICE].reshape(-1, n_kin).std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    X_norm = X.copy()
    X_norm[:, :, KIN_SLICE] = (X[:, :, KIN_SLICE] - mean) / std

    norm_params = np.stack([
        np.append(mean, 0.0),
        np.append(std, 1.0),
    ])

    x_path = out_dir / "X_pretrain_clean_4ch.npy"
    norm_path = out_dir / "norm_params_clean_4ch.npy"
    meta_path = out_dir / "meta_clean_4ch.csv"

    np.save(x_path, X_norm)
    np.save(norm_path, norm_params)
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"  Séquences retenues : {n_seq_out}/{n_seq_in}")
    print(f"  Fenêtres           : {X_norm.shape[0]}  shape={X_norm.shape}")
    print(f"  ✅ {x_path}")
    print(f"  ✅ {norm_path}")
    print(f"  ✅ {meta_path}")
    kin_names = FEATURE_NAMES[KIN_SLICE]
    for i, name in enumerate(kin_names):
        print(f"     {name}: mean={mean[i]:.3f} std={std[i]:.3f}")
    print(f"     {FEATURE_NAMES[VALID_RATIO_IDX_NEW]}: non normalisé")


if __name__ == "__main__":
    main()
