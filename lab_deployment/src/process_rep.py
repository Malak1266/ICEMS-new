"""
process_rep.py
==============
Wrapper "1 commande" à lancer juste après chaque rep enregistrée.
Effectue séquentiellement :
  1. Conversion SpryTrack → fiducials_raw/clean (convert_srytrack)
  2. Évaluation qualité (check_quality)

Usage :
    python src/process_rep.py "C:\\ICEMS\\dataset\\2026-MM-DD\\generic\\translation_slow\\R01"

Retourne code 0 si rep GO, code 1 si REDO.
"""
import argparse
import sys
from pathlib import Path

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ajouter le dossier src au PYTHONPATH pour importer les frères
sys.path.insert(0, str(Path(__file__).parent))

from convert_srytrack import convert_one
from check_quality import check_one


def main():
    parser = argparse.ArgumentParser(
        description="Convertit + vérifie la qualité d'une rep en un seul appel.")
    parser.add_argument("rep_path", help="Chemin du dossier de la rep (ex: .../R01)")
    args = parser.parse_args()

    rep = Path(args.rep_path)
    if not rep.exists() or not rep.is_dir():
        print(f"\n✗ ERREUR : dossier introuvable : {rep}\n")
        sys.exit(2)

    print(f"\n{'=' * 70}")
    print(f"  PROCESS REP : {rep.name}")
    print(f"  Chemin     : {rep}")
    print(f"{'=' * 70}\n")

    # ── Étape 1 : conversion ─────────────────────────────────────────────────
    print("[1/2] Conversion SpryTrack → fiducials...")
    ok_conv = convert_one(str(rep))
    if not ok_conv:
        print("\n✗ ÉCHEC conversion. Vérifier qu'un fichier *_Fiducial.csv "
              "est bien présent dans le dossier.\n")
        sys.exit(2)

    # ── Étape 2 : contrôle qualité ───────────────────────────────────────────
    print("\n[2/2] Contrôle qualité...")
    result = check_one(rep, verbose=False)

    print()
    if result["ok"]:
        print(f"  ✓✓✓ GO — rep validée ({rep.name})")
        print(f"      n_frames = {result['n_frames']}  "
              f"valid = {result['pct_valid']}%  "
              f"kept = {result['mean_kept']:.2f}  "
              f"fps = {result['fps']}")
        print("\n  → Passe à la rep suivante.\n")
        sys.exit(0)
    else:
        print(f"  ✗✗✗ REDO — rep à refaire ({rep.name})")
        print(f"      n_frames = {result['n_frames']}  "
              f"valid = {result['pct_valid']}%  "
              f"kept = {result['mean_kept']:.2f}  "
              f"fps = {result['fps']}")
        print("\n  Raisons :")
        for r in result["reasons"]:
            print(f"    - {r}")
        print("\n  → Supprime ce dossier et refais la rep.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
