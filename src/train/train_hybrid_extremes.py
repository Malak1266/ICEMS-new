"""
train_hybrid_extremes.py
========================
Entraînement Hybrid LSTM-Transformer + HOEL — train extrêmes, test milieu.

Protocole (Uthamacumaran et al.) :
  - TRAIN : Medical Student + Staff
  - TEST  : PGY1…PGY6 + Fellow (jamais vus à l'entraînement)
  - VAL   : held-out participants parmi les extrêmes (early stopping)

Anti-fuite :
  - split par sous-niveau (pas LOPO)
  - validation interne au niveau participant
  - scaler fit sur inner_train réels uniquement
  - augmentation sur inner_train uniquement
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import List, Sequence, Set, Tuple

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.hybrid_lstm_transformer import HybridConfig

from train.train_hybrid_lopo import (
    NESTED_LOPO_SEED,
    Trial,
    TrainConfig,
    load_dataset_icems,
    make_hoel,
    predict_trials,
    train_one_fold,
)

SUBLEVEL_TO_CLASS4_EXTREMES: dict[str, str] = {
    "ms": "student",
    "pgy1": "junior", "pgy2": "junior", "pgy3": "junior",
    "pgy4": "junior", "pgy5": "junior",
    "pgy6": "senior", "fellow": "senior",
    "staff": "expert",
}


def sublevel_to_class4(sublevel: str) -> str:
    key = sublevel.strip().lower()
    if key not in SUBLEVEL_TO_CLASS4_EXTREMES:
        raise KeyError(f"Sous-niveau inconnu pour class_4 : {sublevel!r}")
    return SUBLEVEL_TO_CLASS4_EXTREMES[key]


def split_by_extremes(
    trials: Sequence[Trial],
    train_sublevels: Set[str],
    test_sublevels: Set[str],
) -> Tuple[List[Trial], List[Trial]]:
    """Split trials into train (extremes) and test (middle)."""
    train = [t for t in trials if t.sublevel in train_sublevels]
    test = [t for t in trials if t.sublevel in test_sublevels]
    return train, test


def make_inner_val(
    train_trials: List[Trial],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Trial], List[Trial]]:
    """
    Held-out participants (pas trials) pour early stopping.
    Split au niveau PARTICIPANT pour éviter le leakage.
    """
    participants = sorted({t.participant for t in train_trials})
    rng = np.random.RandomState(seed)
    rng.shuffle(participants)
    n_val = max(1, int(len(participants) * val_fraction))
    val_participants = set(participants[:n_val])

    inner_train = [t for t in train_trials if t.participant not in val_participants]
    inner_val = [t for t in train_trials if t.participant in val_participants]
    return inner_train, inner_val


def augment_trial_extremes(x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    x: (T, 13) features cinématiques d'un trial.
    Retourne une version augmentée, même shape.
    """
    x_aug = x.copy()

    feature_std = x_aug.std(axis=0, keepdims=True) + 1e-8
    jitter = rng.normal(0, 0.015, size=x_aug.shape) * feature_std
    x_aug = x_aug + jitter

    scale_factors = rng.uniform(0.92, 1.08, size=(1, x_aug.shape[1]))
    x_aug = x_aug * scale_factors

    warp_factor = rng.uniform(0.9, 1.1)
    T = x_aug.shape[0]
    T_warped = max(int(T * warp_factor), 10)
    t_orig = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, T_warped)
    x_warped = np.stack([
        np.interp(t_new, t_orig, x_aug[:, f]) for f in range(x_aug.shape[1])
    ], axis=1)
    t_back = np.linspace(0, 1, T)
    x_aug = np.stack([
        np.interp(t_back, t_new, x_warped[:, f]) for f in range(x_warped.shape[1])
    ], axis=1)

    n_mask = int(0.05 * T)
    if n_mask > 0:
        safe_lo = int(0.10 * T)
        safe_hi = int(0.90 * T)
        if safe_hi > safe_lo:
            mask_idx = rng.choice(
                np.arange(safe_lo, safe_hi),
                size=min(n_mask, safe_hi - safe_lo),
                replace=False,
            )
            for idx in mask_idx:
                prev_idx = max(idx - 1, 0)
                next_idx = min(idx + 1, T - 1)
                x_aug[idx] = 0.5 * (x_aug[prev_idx] + x_aug[next_idx])

    return x_aug.astype(np.float32)


def augment_train_extremes(
    train_real: List[Trial],
    seed: int = 42,
    multiplier: int = 2,
) -> List[Trial]:
    """
    Chaque trial original + (multiplier - 1) versions augmentées.
    multiplier=2 → 2x le set ; multiplier=3 → 3x le set.
    """
    if multiplier < 1:
        raise ValueError(f"multiplier doit être >= 1, reçu {multiplier}")
    rng = np.random.RandomState(seed)
    out = list(train_real)
    n_copies = multiplier - 1
    for parent in train_real:
        for j in range(n_copies):
            X_new = augment_trial_extremes(parent.X, rng)
            out.append(Trial(
                X=X_new,
                score=parent.score,
                tier=parent.tier,
                rank=parent.rank,
                participant=parent.participant,
                sublevel=parent.sublevel,
                trial_id=f"{parent.trial_id}_aug_{j}",
                is_augmented=True,
            ))
    rng.shuffle(out)
    return out


def assert_no_leakage_extremes(
    inner_train: List[Trial],
    inner_val: List[Trial],
    test: List[Trial],
    val_participants: Set[str],
    test_sublevels: Set[str],
) -> None:
    train_pids = {t.participant for t in inner_train if not t.is_augmented}
    for t in inner_val:
        assert not t.is_augmented, "FUITE: augmenté dans val interne"
        assert t.participant in val_participants
    for t in test:
        assert not t.is_augmented, "FUITE: augmenté dans test"
        assert t.sublevel in test_sublevels
    assert train_pids.isdisjoint(val_participants)
    for t in inner_train:
        if t.is_augmented:
            assert t.participant not in val_participants


def _prediction_rows(trials: Sequence[Trial], preds: List[float]) -> List[dict]:
    rows: List[dict] = []
    for t, pred in zip(trials, preds):
        rows.append({
            "participant": t.participant,
            "trial_id": t.trial_id,
            "sublevel": t.sublevel,
            "class_4": sublevel_to_class4(t.sublevel),
            "tier": t.tier,
            "rank": t.rank,
            "score_true": t.score,
            "score_pred": float(pred),
        })
    return rows


def run_extremes(
    trials: List[Trial],
    train_sublevels: Set[str],
    test_sublevels: Set[str],
    hybrid_cfg: HybridConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    out_dir: Path,
    augment: bool = True,
    augment_factor: int = 2,
    val_fraction: float = 0.2,
) -> dict:
    """Single split : extrêmes → milieu, avec validation interne sur extrêmes."""
    real_trials = [t for t in trials if not t.is_augmented]
    train_extremes, test_middle = split_by_extremes(
        real_trials, train_sublevels, test_sublevels,
    )

    if not train_extremes:
        raise ValueError(f"TRAIN vide pour sublevels={sorted(train_sublevels)}")
    if not test_middle:
        raise ValueError(f"TEST vide pour sublevels={sorted(test_sublevels)}")

    overlap = train_sublevels & test_sublevels
    if overlap:
        raise ValueError(f"Chevauchement train/test sublevels : {overlap}")

    inner_train_real, inner_val = make_inner_val(
        train_extremes, val_fraction=val_fraction, seed=train_cfg.seed,
    )
    val_participants = {t.participant for t in inner_val}

    if not inner_train_real:
        raise ValueError("inner_train vide après hold-out validation.")

    if augment:
        inner_train_aug = augment_train_extremes(
            inner_train_real, seed=train_cfg.seed, multiplier=augment_factor,
        )
    else:
        inner_train_aug = list(inner_train_real)

    assert_no_leakage_extremes(
        inner_train_aug, inner_val, test_middle, val_participants, test_sublevels,
    )

    n_aug = sum(1 for t in inner_train_aug if t.is_augmented)
    print(
        f"[split] train_extremes={len(train_extremes)} trials "
        f"({len({t.participant for t in train_extremes})} participants) | "
        f"inner_train={len(inner_train_real)} | inner_val={len(inner_val)} "
        f"({sorted(val_participants)}) | test_middle={len(test_middle)} | "
        f"augmented={n_aug} (factor={augment_factor if augment else 1})"
    )

    loss_fn = make_hoel()
    model, info = train_one_fold(
        inner_train_aug, inner_val, hybrid_cfg, train_cfg, loss_fn, device,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model_best.pt")

    mean, std = info["norm_mean"], info["norm_std"]

    preds_middle = predict_trials(model, test_middle, mean, std, train_cfg, device)
    preds_extremes = predict_trials(model, train_extremes, mean, std, train_cfg, device)

    rows_middle = _prediction_rows(test_middle, preds_middle)
    rows_extremes = _prediction_rows(train_extremes, preds_extremes)

    with open(out_dir / "predictions_middle.pkl", "wb") as f:
        pickle.dump(rows_middle, f)
    with open(out_dir / "predictions_extremes.pkl", "wb") as f:
        pickle.dump(rows_extremes, f)

    split_info = {
        "train_sublevels": sorted(train_sublevels),
        "test_sublevels": sorted(test_sublevels),
        "n_train_extreme_trials": len(train_extremes),
        "n_inner_train_trials": len(inner_train_real),
        "n_inner_val_trials": len(inner_val),
        "n_test_middle_trials": len(test_middle),
        "val_participants": sorted(val_participants),
        "val_fraction": val_fraction,
        "augment": augment,
        "augment_factor": augment_factor if augment else 1,
        "n_augmented": n_aug,
        "seq_len": hybrid_cfg.seq_len,
        "seed": train_cfg.seed,
    }
    with open(out_dir / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    training_info = {
        k: v for k, v in info.items()
        if k not in ("norm_mean", "norm_std", "history")
    }
    training_info["history"] = info.get("history", [])
    with open(out_dir / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2)

    with open(out_dir / "norm_stats.pkl", "wb") as f:
        pickle.dump({"mean": mean, "std": std}, f)

    return {
        "split": split_info,
        "training": training_info,
        "predictions_middle": rows_middle,
        "predictions_extremes": rows_extremes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Hybrid LSTM-Transformer — train extrêmes, test milieu (HOEL)",
    )
    ap.add_argument("--output", type=Path, default=Path("results/hybrid_extremes"))
    ap.add_argument("--pkl", type=Path, default=Path("data/continuous_per_trial.pkl"))
    ap.add_argument("--csv", type=Path, default=Path("data/Exvivo_trial_Participants(Sheet1).csv"))
    ap.add_argument("--seed", type=int, default=NESTED_LOPO_SEED)
    ap.add_argument("--augment", type=int, default=1, help="1=avec augmentation, 0=sans")
    ap.add_argument(
        "--augment-factor", type=int, default=2,
        help="Facteur d'augmentation (2=original+1 copie, 3=original+2 copies)",
    )
    ap.add_argument("--val-fraction", type=float, default=0.2,
                    help="Fraction de participants extrêmes en validation interne")
    ap.add_argument("--train-sublevels", nargs="+", default=["ms", "staff"])
    ap.add_argument(
        "--test-sublevels", nargs="+",
        default=["pgy1", "pgy2", "pgy3", "pgy4", "pgy5", "pgy6", "fellow"],
    )
    ap.add_argument("--smoke", action="store_true", help="seq_len réduit, 3 epochs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_cfg = TrainConfig(seed=args.seed)
    if args.smoke:
        train_cfg.max_epochs = 3
        train_cfg.patience = 2
        train_cfg.seq_len = 512
        print("[smoke] seq_len=512, max_epochs=3")

    hybrid_cfg = HybridConfig(
        n_features=13,
        seq_len=train_cfg.seq_len,
        lstm_hidden=128,
        d_model=128,
        nhead=4,
        key_dim=32,
        n_transformer_blocks=1,
        dropout=0.30,
        pos_encoding="sinusoidal",
        bidirectional=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_sublevels = {s.strip().lower() for s in args.train_sublevels}
    test_sublevels = {s.strip().lower() for s in args.test_sublevels}

    augment_factor = max(1, min(3, args.augment_factor))

    print(
        f"[device] {device} | loss=HOEL | augment={bool(args.augment)} | "
        f"factor={augment_factor} | "
        f"train={sorted(train_sublevels)} | test={sorted(test_sublevels)}"
    )

    trials = load_dataset_icems(args.pkl, args.csv)
    print(f"[data] {len(trials)} trials, {len({t.participant for t in trials})} participants")

    out_dir = args.output if not args.smoke else args.output / "smoke"
    result = run_extremes(
        trials,
        train_sublevels=train_sublevels,
        test_sublevels=test_sublevels,
        hybrid_cfg=hybrid_cfg,
        train_cfg=train_cfg,
        device=device,
        out_dir=out_dir,
        augment=bool(args.augment),
        augment_factor=augment_factor,
        val_fraction=args.val_fraction,
    )

    print(f"[done] middle preds={len(result['predictions_middle'])} -> {out_dir}")


if __name__ == "__main__":
    main()
