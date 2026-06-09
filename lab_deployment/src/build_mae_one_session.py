"""
build_mae_one_session.py
========================
Construit X_pretrain_4ch.npy à partir d'UN SEUL dossier pipeline_output.

Usage (Narval) :
    python build_mae_one_session.py \
        --pipeline ~/icems/data/pipeline_output_2026-05-25 \
        --session 2026-05-25 \
        --out ~/icems/data/atracsys_mae_v2
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

KEEP_IDX = [0, 1, 2, 5]
FEATURE_NAMES = ["velocity", "accel", "jerk", "valid_ratio"]
N_FEATURES = 4
KIN_SLICE = slice(0, 3)
VALID_RATIO_IDX_RAW = 5
VALID_RATIO_IDX_NEW = 3

WIN_LEN = 32
STRIDE = 16
MIN_VALID = 0.10
MIN_FRAMES = 50


def load_generic_features(pipeline_dir: Path, session: str):
    records = []
    generic = pipeline_dir / "generic"
    if not generic.is_dir():
        print(f"✗ Dossier generic/ absent dans {pipeline_dir}")
        return records

    for motion_dir in sorted(generic.iterdir()):
        if not motion_dir.is_dir() or motion_dir.name.startswith("_"):
            continue
        for rep_dir in sorted(motion_dir.iterdir()):
            if not rep_dir.is_dir() or not rep_dir.name.startswith("R"):
                continue
            npy = rep_dir / "features_6ch.npy"
            if not npy.exists():
                print(f"  ⚠ skip (pas de features) : {motion_dir.name}/{rep_dir.name}")
                continue
            feat = np.load(npy)
            if feat.shape[0] < MIN_FRAMES:
                print(f"  ⚠ skip (T={feat.shape[0]} < {MIN_FRAMES}) : {motion_dir.name}/{rep_dir.name}")
                continue
            records.append({
                "features": feat,
                "session": session,
                "motion": motion_dir.name,
                "rep": rep_dir.name,
                "T": feat.shape[0],
            })
            print(f"  ✓ {motion_dir.name}/{rep_dir.name}  T={feat.shape[0]}")
    return records


def sliding_windows(feat, win_len, stride, min_valid):
    windows = []
    T = feat.shape[0]
    for start in range(0, T - win_len + 1, stride):
        w = feat[start:start + win_len]
        if w[:, VALID_RATIO_IDX_RAW].mean() >= min_valid:
            windows.append(w[:, KEEP_IDX])
    return np.array(windows, dtype=np.float32) if windows else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", required=True, help="Dossier pipeline_output_YYYY-MM-DD")
    p.add_argument("--session", default="2026-05-25")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    pipeline = Path(args.pipeline).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Pipeline : {pipeline}")
    print(f"Session  : {args.session}")
    print("=" * 60)

    records = load_generic_features(pipeline, args.session)
    if not records:
        print("\n❌ Aucune séquence chargée.")
        return

    all_windows, meta_rows = [], []
    for rec in records:
        wins = sliding_windows(rec["features"], WIN_LEN, STRIDE, MIN_VALID)
        if wins is None or len(wins) == 0:
            continue
        all_windows.append(wins)
        for i in range(len(wins)):
            meta_rows.append({
                "window_idx": len(meta_rows),
                "session": rec["session"],
                "motion": rec["motion"],
                "rep": rec["rep"],
                "win_in_seq": i,
                "T_seq": rec["T"],
            })

    X = np.concatenate(all_windows, axis=0)
    mean = X[:, :, KIN_SLICE].reshape(-1, 3).mean(axis=0)
    std = X[:, :, KIN_SLICE].reshape(-1, 3).std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    X_norm = X.copy()
    X_norm[:, :, KIN_SLICE] = (X[:, :, KIN_SLICE] - mean) / std

    norm_params = np.stack([np.append(mean, 0.0), np.append(std, 1.0)])

    np.save(out_dir / "X_pretrain_4ch.npy", X_norm)
    np.save(out_dir / "norm_params_4ch.npy", norm_params)
    pd.DataFrame(meta_rows).to_csv(out_dir / "meta_4ch.csv", index=False)

    print(f"\n{'=' * 60}")
    print(f"✅ Séquences   : {len(records)}")
    print(f"✅ Fenêtres    : {X.shape[0]:,}")
    print(f"✅ Shape       : {X.shape}  (N, {WIN_LEN}, {N_FEATURES})")
    print(f"✅ Sauvegardé  : {out_dir}")
    print(f"\nPar motion :")
    df = pd.DataFrame(meta_rows)
    print(df.groupby("motion").size().to_string())


if __name__ == "__main__":
    main()
