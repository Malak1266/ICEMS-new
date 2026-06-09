"""
process_session.py
==================
Wrapper "1 commande" à lancer en fin de session de collecte.
Effectue séquentiellement sur toute la session :
  1. Conversion massive : *_Fiducial.csv → fiducials_clean.csv (toutes les reps)
  2. Tracking hongrois + génération features_6ch.npy (toutes les reps)
  3. Rapport qualité global (toutes les reps en GO / REDO)
  4. Création d'une archive .zip prête à transférer

Usage :
    python src/process_session.py "C:\\ICEMS\\dataset\\2026-MM-DD"
    python src/process_session.py "C:\\ICEMS\\dataset\\2026-MM-DD" --fs 120
    python src/process_session.py "C:\\ICEMS\\dataset\\2026-MM-DD" --no-archive
"""
import argparse
import sys
import time
from pathlib import Path

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))

from convert_srytrack import convert_all
from tracking_hungarian import process_all
from check_quality import check_all


def make_archive(session_dir: Path, pipeline_dir: Path):
    """Crée un .zip avec la session brute + la sortie pipeline."""
    import shutil
    parent = session_dir.parent
    zip_base = parent / f"transfer_{session_dir.name}"
    print(f"\n[Archivage] Création de {zip_base}.zip ...")

    tmp = parent / f"_tmp_{session_dir.name}"
    tmp.mkdir(exist_ok=True)
    try:
        shutil.copytree(session_dir,  tmp / "dataset",         dirs_exist_ok=True)
        shutil.copytree(pipeline_dir, tmp / "pipeline_output", dirs_exist_ok=True)
        shutil.make_archive(str(zip_base), "zip", root_dir=tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    size_mb = zip_base.with_suffix(".zip").stat().st_size / 1e6
    print(f"  ✓ Archive : {zip_base.with_suffix('.zip')} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Traite une session de collecte Atracsys en une seule commande.")
    parser.add_argument("session_dir",
                        help="Dossier de la session, ex: C:\\ICEMS\\dataset\\2026-MM-DD")
    parser.add_argument("--fs", type=float, default=120.0,
                        help="Fréquence d'acquisition réelle (défaut 120 Hz)")
    parser.add_argument("--no-archive", action="store_true",
                        help="Ne pas créer l'archive zip à la fin")
    args = parser.parse_args()

    session = Path(args.session_dir).resolve()
    if not session.exists():
        print(f"✗ Dossier introuvable : {session}")
        sys.exit(2)

    pipeline_out = session.parent / f"pipeline_output_{session.name}"
    print(f"\n{'=' * 70}")
    print(f"  TRAITEMENT SESSION  {session.name}")
    print(f"  Fréquence cible     : {args.fs} Hz")
    print(f"  Pipeline output     : {pipeline_out}")
    print(f"{'=' * 70}")

    # ── Étape 1 : conversion ─────────────────────────────────────────────────
    print("\n[1/3] Conversion SpryTrack → fiducials (toutes les reps)...")
    t0 = time.time()
    convert_all(str(session))
    print(f"  Durée : {time.time() - t0:.1f} s")

    # ── Étape 2 : tracking + features ───────────────────────────────────────
    print(f"\n[2/3] Tracking hongrois + features (fs={args.fs} Hz)...")
    t0 = time.time()
    process_all(str(session), str(pipeline_out), fs=args.fs)
    print(f"  Durée : {time.time() - t0:.1f} s")

    # ── Étape 3 : rapport qualité ───────────────────────────────────────────
    print("\n[3/3] Rapport qualité global...")
    for instr_dir in sorted(session.iterdir()):
        if instr_dir.is_dir() and not instr_dir.name.startswith("_"):
            check_all(instr_dir)

    # ── Archive optionnelle ─────────────────────────────────────────────────
    if not args.no_archive and pipeline_out.exists():
        try:
            make_archive(session, pipeline_out)
        except Exception as e:
            print(f"  ⚠ Archivage échoué : {e}")

    print(f"\n{'=' * 70}")
    print(f"  ✓ SESSION {session.name} TRAITÉE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
