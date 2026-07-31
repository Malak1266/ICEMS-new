#!/usr/bin/env python3
"""
LOPO nested anti-fuite — stratégie ICEMS (une variable à la fois).

Étape 0 (obligatoire) :
    python src/diagnose_tsne_baseline.py

Approches 1–4 :
    python run_lopo_corrected.py --approach 1 --smoke-test   # baseline + aug globale
    python run_lopo_corrected.py --approach 2 --smoke-test   # hybrid + MSE
    python run_lopo_corrected.py --approach 3 --full         # hybrid + Focal
    python run_lopo_corrected.py --approach 4 --full         # hybrid + multitask

Usage manuel :
    python run_lopo_corrected.py --full --aug-mode global --model hybrid --loss mse
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from icems_strategy import APPROACHES, get_approach  # noqa: E402
from step_B_classification import (  # noqa: E402
    BASELINE_METRICS,
    N_INTERNAL_VAL_PARTICIPANTS,
    _normalize_dataset,
    corrected_lopo_output_dir,
    print_corrected_lopo_metrics,
    print_early_stopping_summary,
    run_corrected_lopo,
    save_corrected_lopo_results,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


def approach_output_dir(approach_id: int, base: Path | None = None) -> Path:
    from datetime import date
    cfg = get_approach(approach_id)
    root = base or Path("results/strategy")
    return root / f"approach_{approach_id}_{cfg.name}" / date.today().isoformat()


def print_approach_banner(approach_id: int, cfg, smoke: bool) -> None:
    print("=" * 60)
    print(f" STRATÉGIE ICEMS — Approche {approach_id} : {cfg.name}")
    print("=" * 60)
    print(f"  {cfg.description}")
    print(f"  Modèle     : {cfg.model}")
    print(f"  Loss       : {cfg.loss}")
    print(f"  Augmentation : {cfg.aug_mode}")
    print(f"  Référence  : r_participant ≥ {BASELINE_METRICS['r_participant']:.3f}")
    print(f"  Mode       : {'SMOKE TEST' if smoke else 'RUN COMPLET LOPO'}")
    print("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="LOPO corrigé — stratégie ICEMS (approches 1–4, une variable à la fois).",
    )
    ap.add_argument("--data", type=Path, default=Path("data/continuous_per_trial.pkl"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--approach", type=int, default=1, choices=sorted(APPROACHES.keys()))
    ap.add_argument("--aug-mode", type=str, default=None, choices=["none", "v2", "global"])
    ap.add_argument("--model", type=str, default=None, choices=["causal_gru", "hybrid"])
    ap.add_argument("--loss", type=str, default=None,
                    choices=["mse", "class_weights", "focal", "multitask"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--n-inner-val", type=int, default=N_INTERNAL_VAL_PARTICIPANTS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mc-passes", type=int, default=None)
    ap.add_argument("--aug-target", type=int, default=None)
    ap.add_argument("--no-inline-aug", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--focal-alpha", type=float, default=0.1)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    args = ap.parse_args()

    if args.approach == 0:
        print("Approche 0 = diagnostic t-SNE. Lancez : python src/diagnose_tsne_baseline.py")
        sys.exit(0)

    cfg = get_approach(args.approach)
    aug_mode = args.aug_mode or cfg.aug_mode
    model_name = args.model or cfg.model
    loss_mode = args.loss or cfg.loss

    smoke = args.smoke_test or not args.full
    epochs = args.epochs if args.epochs is not None else (3 if smoke else 40)
    max_folds = args.max_folds if args.max_folds is not None else (2 if smoke else None)
    mc_passes = args.mc_passes if args.mc_passes is not None else (3 if smoke else 30)

    out_dir = args.out or approach_output_dir(args.approach)

    if not args.data.exists():
        raise FileNotFoundError(
            f"{args.data} introuvable. Lancez : python src/build_continuous_dataset.py"
        )

    import pickle

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.data, "rb") as f:
        raw = pickle.load(f)
    dataset = _normalize_dataset(raw)

    print_approach_banner(args.approach, cfg, smoke)
    print(f"\n[Dataset] {len(dataset)} trials réels, "
          f"{len({k[0] for k in dataset})} participants")
    print(f"[Sortie]  {out_dir.resolve()}")
    print(f"[Device]  {'cuda' if torch.cuda.is_available() else 'cpu'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds_df, metrics, all_preds, aux = run_corrected_lopo(
        dataset,
        device,
        epochs=epochs,
        max_folds=max_folds,
        patience=args.patience,
        batch_size=args.batch_size,
        mc_passes=mc_passes,
        n_inner_val=args.n_inner_val,
        seed=args.seed,
        aug_target=args.aug_target,
        no_inline_aug=args.no_inline_aug,
        verbose_split_debug=smoke,
        aug_mode=aug_mode,
        model_name=model_name,
        loss_mode=loss_mode,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )

    print_early_stopping_summary(aux.get("fold_train_info", []), epochs)
    print_corrected_lopo_metrics(metrics)

    r_part = metrics.get("r_participant", float("nan"))
    baseline_r = BASELINE_METRICS["r_participant"]
    if not np.isnan(r_part):
        if r_part >= baseline_r:
            print(f"\n✅ r_participant={r_part:.3f} ≥ baseline ({baseline_r:.3f}) — GARDER")
        else:
            print(f"\n❌ r_participant={r_part:.3f} < baseline ({baseline_r:.3f}) — REJETER")

    pred_path, metrics_path = save_corrected_lopo_results(
        out_dir, preds_df, metrics, all_preds, aux,
    )

    strategy_meta = {
        "approach": args.approach,
        "approach_name": cfg.name,
        "model": model_name,
        "loss": loss_mode,
        "aug_mode": aug_mode,
        "smoke_test": smoke,
        "baseline_r_participant": baseline_r,
        "metrics": metrics,
    }
    meta_path = out_dir / "strategy_run.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(strategy_meta, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[Sauvegarde]")
    print(f"  {pred_path.resolve()}")
    print(f"  {metrics_path.resolve()}")
    print(f"  {meta_path.resolve()}")

    if smoke:
        print("\n✅ Smoke test terminé — assertions anti-fuite OK.")
        print(f"   Prochaine étape : --full --approach {args.approach}")
    else:
        print(f"\n✅ Approche {args.approach} terminée ({metrics.get('n_folds', '?')} folds).")


if __name__ == "__main__":
    main()
