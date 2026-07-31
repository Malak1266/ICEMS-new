#!/usr/bin/env python3
"""
run_extremes_comparison.py
==========================
Script principal du protocole "extrêmes" EVICEMS.

Entraîne CausalGRU ET Hybrid1EVICEMS dans des conditions STRICTEMENT identiques
(même pkl, même seq_len, même seed, même normalisation, mêmes hyperparamètres),
puis compare leur validité prédictive sur les niveaux intermédiaires jamais vus.

Contraintes respectées :
  - Aucune modification de CausalGRUScorer / run_corrected_lopo / run_lopo_corrected.py
  - Seeds fixes partout (un seul `seed` propagé aux deux modèles)
  - Par défaut, tous les extrêmes sont utilisés en TRAIN ; --use-val-split restaure
    l'ancien GroupShuffleSplit strict 70/15/15.

Usage :
    python run_extremes_comparison.py --pkl data/continuous_per_trial.pkl
    python run_extremes_comparison.py --seq-len 800 --seed 42 --epochs 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Rendre src/ importable (exécution depuis la racine du projet).
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from eval.extremes_protocol import run_extremes_protocol  # noqa: E402
from eval.predictive_validity import compute_predictive_validity  # noqa: E402

MODELS = ["causal_gru", "hybrid1_evicems"]
MODEL_LABELS = {"causal_gru": "CausalGRU", "hybrid1_evicems": "Hybrid1EVICEMS"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Comparaison extrêmes CausalGRU vs Hybrid1EVICEMS.")
    ap.add_argument("--pkl", type=Path, default=Path("data/continuous_per_trial.pkl"))
    ap.add_argument("--out", type=Path, default=Path("results/extremes"))
    ap.add_argument("--seq-len", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--no_augment", dest="augment", action="store_false", default=True,
                    help="Désactive l'augmentation Option D du pool TRAIN (activée par défaut).")
    ap.add_argument("--multiplier", type=int, default=3,
                    help="Multiplicateur global de l'augmentation (étape 2). Défaut=3.")
    ap.add_argument(
        "--use-val-split",
        action="store_true",
        default=False,
        help="Split val/test interne 70/15/15 sur pool extrêmes. "
             "Défaut False = TOUS les 22 participants extrêmes en TRAIN, "
             "epochs fixe sans early stopping.",
    )
    ap.add_argument(
        "--auto-seq-len",
        action="store_true",
        default=False,
        help="Ajuste seq_len par modèle : CausalGRU→seq_len=4000 crop=start, "
             "Hybrid1EVICEMS→seq_len=2000 crop=center.",
    )
    args = ap.parse_args()

    if not args.pkl.exists():
        sys.exit(f"Fichier pkl introuvable : {args.pkl}")
    args.out.mkdir(parents=True, exist_ok=True)

    comparison = {}
    for model_name in MODELS:
        print("=" * 70)
        print(f" Protocole extrêmes — {MODEL_LABELS[model_name]}")
        print("=" * 70)
        model_out = args.out / model_name

        # Mêmes paramètres de protocole ; auto_seq_len peut adapter causal_gru à 4000.
        proto = run_extremes_protocol(
            pkl_path=args.pkl,
            model_name=model_name,
            seq_len=args.seq_len,
            seed=args.seed,
            output_dir=model_out,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            augment=args.augment,
            global_multiplier=args.multiplier,
            use_val_split=args.use_val_split,
            auto_seq_len=args.auto_seq_len,
        )
        print(f"  Splits : train={proto['n_train']} val={proto['n_val']} "
              f"test_interne={proto['n_test_internal']} | "
              f"participants milieu testés={proto['n_test_middle_participants']}")
        mae_label = "test interne (ms/staff held-out)" if proto["use_val_split"] else "train extrêmes"
        print(f"  MAE {mae_label} : {proto['train_internal_mae']:.4f}")
        print(f"  seq_len effectif={proto['seq_len']} crop_mode={proto['crop_mode']}")

        validity = compute_predictive_validity(proto["json_path"], model_out)
        print(f"  Validité prédictive : R²={validity['r2']:.3f}  "
              f"pente={validity['slope']:.3f}  MAE={validity['mae']:.3f}")

        comparison[model_name] = {
            "r2": validity["r2"],
            "slope": validity["slope"],
            "intercept": validity["intercept"],
            "mae": validity["mae"],
            "mse": validity["mse"],
            "r2_ci95": validity["r2_ci95"],
            "slope_ci95": validity["slope_ci95"],
            "train_internal_mae": proto["train_internal_mae"],
            "json_path": proto["json_path"],
            "seq_len_effective": proto["seq_len"],
            "crop_mode": proto["crop_mode"],
            "use_val_split": proto["use_val_split"],
        }

    # Tableau comparatif JSON
    comparison_payload = {
        "params": {
            "pkl": str(args.pkl), "seq_len": args.seq_len, "seed": args.seed,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "patience": args.patience,
            "augment": args.augment, "global_multiplier": args.multiplier,
            "use_val_split": args.use_val_split, "auto_seq_len": args.auto_seq_len,
        },
        "models": comparison,
    }
    comp_path = args.out / "comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison_payload, f, indent=2, ensure_ascii=False)

    # Affichage console
    print("\n" + "=" * 78)
    print(" COMPARAISON — validité prédictive (niveaux intermédiaires jamais vus)")
    print("=" * 78)
    header = f"| {'Modèle':<15} | {'R²':>6} | {'Pente':>6} | {'MAE':>6} | {'MSE':>6} | {'IC95 R²':>16} |"
    print(header)
    print("|" + "-" * 17 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 8 + "|"
          + "-" * 8 + "|" + "-" * 18 + "|")
    for model_name in MODELS:
        m = comparison[model_name]
        ci = f"[{m['r2_ci95'][0]:.2f}, {m['r2_ci95'][1]:.2f}]"
        print(f"| {MODEL_LABELS[model_name]:<15} | {m['r2']:>6.3f} | {m['slope']:>6.3f} | "
              f"{m['mae']:>6.3f} | {m['mse']:>6.3f} | {ci:>16} |")

    print(f"\n[Sauvegarde] {comp_path.resolve()}")


if __name__ == "__main__":
    main()
