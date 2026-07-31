"""
Entraînement Hybrid LSTM-Transformer — train extrêmes, éval intermédiaires.
"""
from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from augment import augment_train_extremes
from exp_config import (
    EXP_ROOT,
    ExpAugmentConfig,
    ExpTrainConfig,
    N_FEATURES,
    SRC_ROOT,
    output_dir,
)

from sklearn.model_selection import GroupShuffleSplit

from data_split import (
    TrialRecord,
    load_atracsys_trials,
    split_extreme_intermediate,
    split_summary,
)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from losses.hoel import HOEL  # noqa: E402
from models.hybrid_lstm_transformer import HybridConfig, HybridLSTMTransformer  # noqa: E402

SEED = int(os.environ.get("SEED", "0"))

# Noms alignés sur aggregate_seeds.py / audit (Novice < Junior < Senior < Expert).
_SUBLEVEL_TO_GROUP = {
    "ms": "Novice",
    "pgy1": "Junior", "pgy2": "Junior", "pgy3": "Junior",
    "pgy4": "Junior", "pgy5": "Junior",
    "pgy6": "Senior", "fellow": "Senior",
    "staff": "Expert",
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _group_for(sublevel: str) -> str:
    return _SUBLEVEL_TO_GROUP.get(sublevel.strip().lower(), "Junior")


def _export_raw_scores_csv(
    model: HybridLSTMTransformer,
    splits: Sequence[tuple[str, List[TrialRecord]]],
    mean: np.ndarray,
    std: np.ndarray,
    cfg: ExpTrainConfig,
    device: torch.device,
    seed: int,
) -> Path:
    """Exporte REPO_ROOT/runs/seed_{SEED}.csv — scores Tanh bruts, sans calibration."""
    from exp_config import REPO_ROOT

    rows: List[dict] = []
    for split_name, trials in splits:
        preds = predict_trials(model, trials, mean, std, cfg, device)
        for t, raw in zip(trials, preds):
            rows.append({
                "participant": t.participant,
                "group": _group_for(t.sublevel),
                "split": split_name,
                "raw_score": float(raw),
            })

    df = pd.DataFrame(rows)
    assert df["raw_score"].between(-1, 1).all(), "raw_score hors [-1, 1]"

    out_dir = REPO_ROOT / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_{seed}.csv"
    df.to_csv(out_path, index=False)
    # Copie sous l'expérience (tracabilité locale du job)
    exp_dir = EXP_ROOT / "runs"
    exp_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(exp_dir / f"seed_{seed}.csv", index=False)
    print(f"[export] {len(df)} lignes → {out_path}")
    return out_path


def _holdout_extreme_val(
    train_real: List[TrialRecord],
    seed: int,
    val_frac: float = 0.15,
) -> Tuple[List[TrialRecord], List[TrialRecord]]:
    """Validation interne sur participants extrêmes uniquement."""
    groups = np.array([t.participant for t in train_real])
    idx = np.arange(len(train_real))
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    fit_idx, val_idx = next(gss.split(idx, groups=groups))
    fit = [train_real[i] for i in fit_idx]
    val = [train_real[i] for i in val_idx]
    assert {t.participant for t in fit}.isdisjoint({t.participant for t in val})
    return fit, val


def compute_norm_stats(trials: Sequence[TrialRecord]) -> Tuple[np.ndarray, np.ndarray]:
    real = [t for t in trials if not t.is_augmented]
    kin = np.concatenate([t.X.reshape(-1, N_FEATURES) for t in real], axis=0)
    mean = kin.mean(axis=0).astype(np.float32)
    std = (kin.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std


def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def crop_sequence(X: np.ndarray, seq_len: int, mode: str = "start") -> np.ndarray:
    T, F = X.shape
    if T >= seq_len:
        if mode == "start":
            return X[:seq_len].astype(np.float32)
        start = np.random.randint(0, T - seq_len)
        return X[start : start + seq_len].astype(np.float32)
    pad = np.zeros((seq_len - T, F), dtype=np.float32)
    return np.concatenate([X, pad], axis=0)


class HybridTrialDataset(Dataset):
    def __init__(
        self,
        trials: Sequence[TrialRecord],
        mean: np.ndarray,
        std: np.ndarray,
        cfg: ExpTrainConfig,
        train: bool = True,
        use_train_score: bool = True,
    ):
        self.trials = list(trials)
        self.mean = mean
        self.std = std
        self.cfg = cfg
        self.train = train
        self.use_train_score = use_train_score

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx: int):
        t = self.trials[idx]
        x = apply_norm(t.X, self.mean, self.std)
        mode = self.cfg.crop_mode if self.train else "start"
        x = crop_sequence(x, self.cfg.seq_len, mode=mode)
        y = t.train_score if self.use_train_score else t.score
        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(t.tier, dtype=torch.long),
            torch.tensor(t.rank, dtype=torch.long),
        )


def collate_hybrid(batch):
    xs, scores, tiers, ranks = zip(*batch)
    return torch.stack(xs), torch.stack(scores), torch.stack(tiers), torch.stack(ranks)


def make_hoel() -> HOEL:
    return HOEL(
        lambda_asym=0.20,
        lambda_pair=0.30,
        lambda_top=0.40,
        lambda_anchor=0.08,
        tier_weights={0: 1.0, 1: 2.5, 2: 4.0},
    )


def train_model(
    train_trials: List[TrialRecord],
    val_trials: List[TrialRecord],
    hybrid_cfg: HybridConfig,
    train_cfg: ExpTrainConfig,
    device: torch.device,
) -> Tuple[HybridLSTMTransformer, dict]:
    mean, std = compute_norm_stats(train_trials)
    train_ds = HybridTrialDataset(train_trials, mean, std, train_cfg, train=True)
    val_ds = HybridTrialDataset(
        val_trials, mean, std, train_cfg, train=False, use_train_score=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate_hybrid,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate_hybrid,
    )

    model = HybridLSTMTransformer(hybrid_cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay,
    )

    if train_cfg.use_hoel:
        loss_fn: nn.Module = make_hoel()
        use_hoel = True
    else:
        loss_fn = nn.MSELoss()
        use_hoel = False

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    history: List[dict] = []

    for epoch in range(1, train_cfg.max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb, tb, rb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            tb, rb = tb.to(device), rb.to(device)
            opt.zero_grad()
            pred = model(xb)
            if use_hoel:
                loss = loss_fn(pred, yb, tb, rb)["loss"]
            else:
                loss = loss_fn(pred.squeeze(-1), yb)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, tb, rb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                tb, rb = tb.to(device), rb.to(device)
                pred = model(xb)
                if use_hoel:
                    val_losses.append(float(loss_fn(pred, yb, tb, rb)["loss"].item()))
                else:
                    val_losses.append(float(nn.functional.mse_loss(pred.squeeze(-1), yb).item()))

        tr = float(np.mean(train_losses)) if train_losses else float("nan")
        va = float(np.mean(val_losses)) if val_losses else float("nan")
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        print(f"  epoch {epoch}: train={tr:.4f} val={va:.4f}")

        if va < best_val:
            best_val = va
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= train_cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    info = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "history": history,
        "norm_mean": mean,
        "norm_std": std,
    }
    return model, info


@torch.no_grad()
def predict_trials(
    model: HybridLSTMTransformer,
    trials: Sequence[TrialRecord],
    mean: np.ndarray,
    std: np.ndarray,
    cfg: ExpTrainConfig,
    device: torch.device,
) -> List[float]:
    model.eval()
    ds = HybridTrialDataset(trials, mean, std, cfg, train=False, use_train_score=False)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_hybrid)
    preds: List[float] = []
    for xb, _, _, _ in loader:
        out = model(xb.to(device)).squeeze(-1).cpu().numpy()
        preds.extend(out.tolist())
    return preds


def run_single_experiment(
    seed: Optional[int] = None,
    smoke: bool = False,
    out: Optional[Path] = None,
    pkl: Optional[Path] = None,
    csv: Optional[Path] = None,
    use_hoel: Optional[bool] = None,
) -> dict:
    seed = SEED if seed is None else seed
    _set_seed(seed)

    train_cfg = ExpTrainConfig(seed=seed)
    if use_hoel is not None:
        train_cfg.use_hoel = bool(use_hoel)
    aug_cfg = ExpAugmentConfig()
    if smoke:
        train_cfg.seq_len = 512
        train_cfg.max_epochs = 3
        train_cfg.patience = 2
        train_cfg.global_multiplier = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = out or output_dir(seed, smoke=smoke)
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = load_atracsys_trials(pkl, csv)
    train_real, test_real = split_extreme_intermediate(trials)
    train_fit_real, val_extreme = _holdout_extreme_val(train_real, seed=seed)
    summary = split_summary(train_real, test_real)
    summary["n_val_extreme_trials"] = len(val_extreme)
    print(f"[split] {json.dumps(summary, indent=2)}")

    train_aug = augment_train_extremes(
        train_fit_real,
        seed=seed,
        cfg=aug_cfg,
        global_multiplier=train_cfg.global_multiplier,
    )
    n_aug = sum(1 for t in train_aug if t.is_augmented)
    print(f"[aug] train_real={len(train_real)} augmented={n_aug} total={len(train_aug)}")

    hybrid_cfg = HybridConfig(
        n_features=N_FEATURES,
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

    loss_name = "HOEL" if train_cfg.use_hoel else "MSE (flat, no class weighting)"
    print(
        f"[train] device={device} seed={seed} seq_len={train_cfg.seq_len} "
        f"loss={loss_name}"
    )
    model, info = train_model(train_aug, val_extreme, hybrid_cfg, train_cfg, device)
    info["use_hoel"] = bool(train_cfg.use_hoel)
    info["loss"] = "hoel" if train_cfg.use_hoel else "mse"

    csv_path = _export_raw_scores_csv(
        model,
        splits=[
            ("train", train_fit_real),
            ("val", val_extreme),
            ("test", test_real),
        ],
        mean=info["norm_mean"],
        std=info["norm_std"],
        cfg=train_cfg,
        device=device,
        seed=seed,
    )

    preds_raw = predict_trials(model, test_real, info["norm_mean"], info["norm_std"], train_cfg, device)
    predictions = []
    for t, pred in zip(test_real, preds_raw):
        predictions.append({
            "participant": t.participant,
            "trial_id": t.trial_id,
            "sublevel": t.sublevel,
            "subgroup": t.subgroup,
            "tier": t.tier,
            "rank": t.rank,
            "score_true": t.score,
            "score_pred": float(pred),
        })

    from evaluate import run_evaluation

    metrics = run_evaluation(predictions, out_dir)

    torch.save({
        "state_dict": model.state_dict(),
        "hybrid_cfg": asdict(hybrid_cfg),
        "norm_mean": info["norm_mean"],
        "norm_std": info["norm_std"],
        "seed": seed,
    }, out_dir / "model.pt")

    with open(out_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "training_info.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in info.items() if k not in ("norm_mean", "norm_std")}, f, indent=2)

    result = {
        "seed": seed,
        "out_dir": str(out_dir),
        "raw_scores_csv": str(csv_path),
        "split": summary,
        "metrics": metrics,
        "n_augmented": n_aug,
        "use_hoel": bool(train_cfg.use_hoel),
        "loss": "hoel" if train_cfg.use_hoel else "mse",
    }
    with open(out_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
