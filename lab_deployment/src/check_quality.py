"""
check_quality.py
================
Outil de contrôle qualité rapide pour la collecte Atracsys.

À lancer après chaque rep (ou en mode batch sur une session entière) pour
décider de garder / re-faire l'enregistrement avant de passer au suivant.

Critères GO :
  - pct_valid               ≥ 70 %   (sphères correctement détectées)
  - mean_n_fiducials_kept   ≥ 3.0    (au moins 3 sphères vues en moyenne)
  - n_frames_total          ≥ 100    (pas une rep tronquée)

Usage :
  # Une seule rep
  python src/check_quality.py "C:/path/to/R01"

  # Toute une session
  python src/check_quality.py --all "C:/path/to/2026-MM-DD/generic"
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# ── Critères de qualité ─────────────────────────────────────────────────────
MIN_PCT_VALID         = 70.0
MIN_MEAN_FIDUCIALS    = 3.0
MIN_FRAMES            = 100
TARGET_FPS            = 120.0
FPS_TOLERANCE         = 0.20   # ±20 %


def check_one(rep_dir: Path, verbose: bool = True) -> dict:
    """Évalue la qualité d'une seule rep."""
    result = {
        "rep": str(rep_dir),
        "ok":  False,
        "reasons": [],
    }

    clean_csv = rep_dir / "fiducials_clean.csv"
    if not clean_csv.exists():
        result["reasons"].append("fiducials_clean.csv absent — convert_srytrack.py non lancé ?")
        if verbose:
            print(f"  ✗ {rep_dir.name}: pas de fiducials_clean.csv")
        return result

    df = pd.read_csv(clean_csv)
    if len(df) == 0:
        result["reasons"].append("CSV vide")
        if verbose:
            print(f"  ✗ {rep_dir.name}: CSV vide")
        return result

    # Statistiques par frame
    n_frames_unique = df["frame_idx"].nunique()
    if "n_fiducials_kept" in df.columns:
        per_frame_kept = df.groupby("frame_idx")["n_fiducials_kept"].first()
    else:
        per_frame_kept = df.groupby("frame_idx").size()

    n_frames_expected = df["frame_idx"].max() + 1
    pct_valid = 100.0 * (per_frame_kept >= 3).mean()
    mean_kept = float(per_frame_kept.mean())

    fps = float("nan")
    if "t_pc_ns" in df.columns and len(df) > 1:
        t_min = df["t_pc_ns"].min()
        t_max = df["t_pc_ns"].max()
        duration_s = (t_max - t_min) / 1e9
        if duration_s > 0:
            fps = n_frames_unique / duration_s

    result.update({
        "n_frames":      int(n_frames_unique),
        "pct_valid":     round(pct_valid, 1),
        "mean_kept":     round(mean_kept, 2),
        "fps":           round(fps, 1) if not np.isnan(fps) else None,
    })

    # Évaluation
    if pct_valid < MIN_PCT_VALID:
        result["reasons"].append(f"pct_valid {pct_valid:.1f}% < {MIN_PCT_VALID}%")
    if mean_kept < MIN_MEAN_FIDUCIALS:
        result["reasons"].append(f"mean_kept {mean_kept:.1f} < {MIN_MEAN_FIDUCIALS}")
    if n_frames_unique < MIN_FRAMES:
        result["reasons"].append(f"n_frames {n_frames_unique} < {MIN_FRAMES}")
    if not np.isnan(fps):
        lo = TARGET_FPS * (1 - FPS_TOLERANCE)
        hi = TARGET_FPS * (1 + FPS_TOLERANCE)
        if not (lo <= fps <= hi):
            result["reasons"].append(
                f"fps {fps:.1f} hors plage [{lo:.0f}-{hi:.0f}]"
            )

    result["ok"] = len(result["reasons"]) == 0

    if verbose:
        verdict = "✓ GO " if result["ok"] else "✗ REDO"
        print(
            f"  {verdict} {rep_dir.name:20s} "
            f"frames={result['n_frames']:4d}  "
            f"valid={result['pct_valid']:5.1f}%  "
            f"kept={result['mean_kept']:.2f}  "
            f"fps={result['fps']}"
            f"{'  →  ' + ' ; '.join(result['reasons']) if result['reasons'] else ''}"
        )
    return result


def check_all(session_dir: Path):
    """Évalue toutes les reps d'une session (toutes motions confondues)."""
    session_dir = Path(session_dir)
    print(f"\n{'=' * 80}")
    print(f"  Contrôle qualité de la session : {session_dir}")
    print(f"{'=' * 80}\n")

    results = []
    for motion_dir in sorted(session_dir.iterdir()):
        if not motion_dir.is_dir():
            continue
        print(f"\n[{motion_dir.name}]")
        for rep_dir in sorted(motion_dir.iterdir()):
            if rep_dir.is_dir() and rep_dir.name.startswith("R"):
                res = check_one(rep_dir)
                res["motion"] = motion_dir.name
                results.append(res)

    if not results:
        print("\n⚠ Aucune rep trouvée.")
        return

    df = pd.DataFrame(results)
    n_total = len(df)
    n_ok = int(df["ok"].sum())
    pct_ok = 100.0 * n_ok / n_total

    print(f"\n{'=' * 80}")
    print(f"  BILAN GLOBAL")
    print(f"{'=' * 80}")
    print(f"  Reps OK            : {n_ok}/{n_total}  ({pct_ok:.1f}%)")
    print(f"  pct_valid moyen    : {df['pct_valid'].mean():.1f}%")
    print(f"  mean_kept moyen    : {df['mean_kept'].mean():.2f}")
    if df["fps"].notna().any():
        print(f"  fps moyen          : {df['fps'].mean():.1f} Hz")

    bad = df[~df["ok"]]
    if len(bad) > 0:
        print(f"\n  Reps à refaire ({len(bad)}) :")
        for _, row in bad.iterrows():
            reasons = " ; ".join(row["reasons"])
            rep_short = Path(row["rep"]).name
            print(f"    - {row['motion']}/{rep_short}  →  {reasons}")

    out_csv = session_dir / "_quality_report.csv"
    df_save = df.drop(columns=["reasons"]).copy()
    df_save["reasons"] = df["reasons"].apply(lambda lst: " ; ".join(lst))
    df_save.to_csv(out_csv, index=False)
    print(f"\n  Rapport sauvegardé : {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contrôle qualité des reps Atracsys")
    parser.add_argument("path", nargs="?",
                        help="Chemin d'une rep unique (R01) ou d'une session si --all")
    parser.add_argument("--all", action="store_true",
                        help="Évaluer toute une session (sous-dossiers motion → R*)")
    args = parser.parse_args()

    if not args.path:
        parser.print_help()
        sys.exit(1)

    target = Path(args.path)
    if not target.exists():
        print(f"✗ Chemin introuvable : {target}")
        sys.exit(1)

    if args.all:
        check_all(target)
    else:
        check_one(target)
