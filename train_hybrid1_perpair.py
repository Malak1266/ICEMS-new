"""
train_hybrid1_perpair.py
========================
Entrainement per-paire (eApp 2) : 1 Hybrid1 Model A par paire outil-metrique.

Pour chaque paire k in {0..9} :
  - entree univariee (B, L, 1)
  - extremes experts(+1) + novices(-1) uniquement
  - pleine resolution ; trials > max_len -> fenetres non chevauchantes
  - Adam lr=1e-3, MSE reequilibree (fenetres), batch configurable, 50 epochs,
    early stop patience=12 apres min_epochs=20
  - split 70/15/15 stratifie groupe+participant

Sorties : results/hybrid1_perpair/pair{k}/model_A_best.pt, norm_stats.pkl, ...
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_hybrid1 import (
    BUTTERWORTH_ORDER,
    DEFAULT_BUTTERWORTH_FC,
    GROUP4_ORDER,
    PAIR_NAMES,
    TrialDataset,
    build_trials,
    collate_pad,
    compute_norm_stats,
    expand_trials_to_windows,
    load_raw_trials,
    log_gpu_memory,
    resolve_hybrid1_paths,
    select_extremes,
    stratified_trial_split,
    to_pair_trials,
)
from models_hybrid1 import Hybrid1Config, Hybrid1ModelA, count_params

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("train_hybrid1_perpair")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(trials, mean, std, batch_size, shuffle):
    ds = TrialDataset(trials, mean, std, target="label_extreme")
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_pad,
    )


def evaluate_loss(model, loader, device, loss_fn) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for X, mask, y in loader:
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            out = model(X, key_padding_mask=mask)
            total += loss_fn(out, y).item() * X.size(0)
            n += X.size(0)
    return total / max(n, 1)


def train_one_pair(
    pair_idx: int,
    raw: list,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    pair_name = PAIR_NAMES[pair_idx]
    out_dir = args.out / f"pair{pair_idx}_{pair_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"TRAIN pair {pair_idx} ({pair_name}) — Hybrid1 Model A univarie")
    logger.info("=" * 70)

    all_trials = build_trials(
        raw,
        max_len=args.max_len,
        full_resolution=True,
        apply_butterworth=args.butterworth,
        butterworth_fc=DEFAULT_BUTTERWORTH_FC,
    )
    pair_all = to_pair_trials(all_trials, pair_idx)
    extremes = select_extremes(pair_all)
    n_exp = sum(1 for t in extremes if t.label_extreme == 1)
    n_nov = sum(1 for t in extremes if t.label_extreme == -1)
    logger.info(f"[data] {len(extremes)} extremes (expert={n_exp}, novice={n_nov})")

    train_t, val_t, test_t = stratified_trial_split(extremes, seed=args.seed)
    train_stride = args.train_window_stride
    train_t = expand_trials_to_windows(train_t, args.max_len, stride=train_stride)
    val_t = expand_trials_to_windows(val_t, args.max_len)
    test_t = expand_trials_to_windows(test_t, args.max_len)

    def grp_counts(ts):
        d: dict[str, int] = {}
        for t in ts:
            d[t.group4] = d.get(t.group4, 0) + 1
        return {g: d.get(g, 0) for g in GROUP4_ORDER if d.get(g, 0)}

    logger.info(
        f"[split+windows] train={len(train_t)} val={len(val_t)} test={len(test_t)} "
        f"(train_stride={train_stride})"
    )

    mean, std = compute_norm_stats(train_t)
    logger.info(f"[norm] stats train-only, mean={mean.ravel()}, std={std.ravel()}")

    batch_size = args.batch_size
    train_loader = make_loader(train_t, mean, std, batch_size, shuffle=True)
    val_loader = make_loader(val_t, mean, std, batch_size, shuffle=False)
    test_loader = make_loader(test_t, mean, std, batch_size, shuffle=False)

    cfg = Hybrid1Config(n_features=1, max_len=args.max_len, dropout=args.dropout)
    model = Hybrid1ModelA(cfg).to(device)
    logger.info(f"[model] params={count_params(model):,} max_len={args.max_len}")
    log_gpu_memory("after model init")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    y_tr = torch.tensor([float(t.label_extreme) for t in train_t])
    n_pos = int((y_tr > 0).sum())
    n_neg = int((y_tr < 0).sum())
    n_win = n_pos + n_neg
    w_pos = n_win / (2.0 * max(n_pos, 1))
    w_neg = n_win / (2.0 * max(n_neg, 1))
    logger.info(
        f"[balance] fenetres +1={n_pos} -1={n_neg} | w_pos={w_pos:.3f} w_neg={w_neg:.3f} "
        f"| point fixe MSE non ponderee = {(n_pos - n_neg) / max(n_win, 1):+.3f}"
    )

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = torch.where(target > 0, w_pos, w_neg).to(pred.device)
        return (w * (pred - target) ** 2).mean()

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    wait = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        oom_retried = False
        for X, mask, y in train_loader:
            try:
                X, mask, y = X.to(device), mask.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(X, key_padding_mask=mask)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()
                tot += loss.item() * X.size(0)
                n += X.size(0)
            except torch.cuda.OutOfMemoryError:
                if oom_retried:
                    raise
                torch.cuda.empty_cache()
                oom_retried = True
                logger.warning("[OOM] batch trop grand — videz le cache et continuez")
                raise

        train_loss = tot / max(n, 1)
        val_loss = evaluate_loss(model, val_loader, device, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info(f"[epoch {epoch:3d}] train={train_loss:.4f} val={val_loss:.4f}")
        if epoch == 1 or epoch % 10 == 0:
            log_gpu_memory(f"pair{pair_idx} epoch{epoch}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if epoch >= args.min_epochs and wait >= args.patience:
                logger.info(
                    f"[early-stop] patience {args.patience} apres min_epochs={args.min_epochs} "
                    f"(best epoch {best_epoch})"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss = evaluate_loss(model, test_loader, device, loss_fn)
    logger.info(f"[done] best_epoch={best_epoch} val={best_val:.4f} test={test_loss:.4f}")

    torch.save(best_state, out_dir / "model_A_best.pt")
    with open(out_dir / "norm_stats.pkl", "wb") as f:
        pickle.dump({"mean": mean, "std": std, "pair_idx": pair_idx}, f)

    split_ids = {
        "train": [(t.participant, t.trial, t.window_start) for t in train_t],
        "val": [(t.participant, t.trial, t.window_start) for t in val_t],
        "test": [(t.participant, t.trial, t.window_start) for t in test_t],
    }
    with open(out_dir / "split_ids.pkl", "wb") as f:
        pickle.dump(split_ids, f)

    config = {
        "mode": "perpair",
        "data": getattr(args, "data", None),
        "pkl": str(args.pkl),
        "pair_idx": pair_idx,
        "pair_name": pair_name,
        "seed": args.seed,
        "max_len": args.max_len,
        "batch_size": batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
        "min_epochs": args.min_epochs,
        "train_window_stride": train_stride,
        "balance": {"n_pos": n_pos, "n_neg": n_neg, "w_pos": w_pos, "w_neg": w_neg},
        "dropout": args.dropout,
        "weight_decay": 0.0,
        "butterworth": args.butterworth,
        "butterworth_order": BUTTERWORTH_ORDER,
        "butterworth_fc": DEFAULT_BUTTERWORTH_FC,
        "n_train_windows": len(train_t),
        "n_val_windows": len(val_t),
        "n_test_windows": len(test_t),
        "train_groups": grp_counts(train_t),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "n_params": count_params(model),
    }
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return config


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Hybrid1 per-pair (eApp 2)")
    ap.add_argument("--data", choices=["v1", "v2"], default=None,
                    help="Version trial_tensor (fixe pkl + out si non surcharges)")
    ap.add_argument("--pkl", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Papier=32 ; reduire automatiquement si OOM (run_hybrid1_perpair)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--min-epochs", type=int, default=20,
                    help="Early-stop actif seulement apres min_epochs")
    ap.add_argument("--train-window-stride", type=int, default=None,
                    help="Stride fenetres train (defaut: max_len//2 ; val/test non chevauchants)")
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--butterworth", action=argparse.BooleanOptionalAction, default=False,
                    help="Defaut False : v1/v2 deja filtres en amont")
    ap.add_argument("--pairs", type=int, nargs="*", default=None,
                    help="Indices de paires a entrainer (defaut: 0..9)")
    args = ap.parse_args()

    pkl_path, out_dir = resolve_hybrid1_paths(args.data, args.pkl, args.out)
    args.pkl = pkl_path
    args.out = out_dir
    if args.train_window_stride is None:
        args.train_window_stride = args.max_len // 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("HYBRID1 PER-PAIR TRAINING — reproduction fidele eApp 2")
    logger.info("=" * 70)
    logger.info(f"[config] data={args.data or 'legacy'} pkl={args.pkl} out={args.out}")
    logger.info(f"[config] seed={args.seed} device={device} max_len={args.max_len} "
                f"batch={args.batch_size} butterworth={args.butterworth}")
    logger.info(f"[config] fc Butterworth={DEFAULT_BUTTERWORTH_FC} order={BUTTERWORTH_ORDER}")

    if not args.pkl.exists():
        raise FileNotFoundError(f"Donnees manquantes : {args.pkl}")

    raw = load_raw_trials(str(args.pkl))
    pair_indices = args.pairs if args.pairs is not None else list(range(10))

    summary = {"seed": args.seed, "pairs": {}}
    for k in pair_indices:
        if k < 0 or k >= 10:
            raise ValueError(f"pair_idx invalide : {k}")
        summary["pairs"][str(k)] = train_one_pair(k, raw, args, device)
        torch.cuda.empty_cache()

    with open(args.out / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"[save] resume -> {args.out / 'train_summary.json'}")


if __name__ == "__main__":
    main()
