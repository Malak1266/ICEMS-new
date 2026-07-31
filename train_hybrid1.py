"""
train_hybrid1.py — MODE "faithful"
==================================
Entrainement du Modele A (Hybrid1, fidele papier EV-ICEMS) sur les EXTREMES seulement.

Protocole "faithful" :
  - Selection des trials avec label_extreme in {+1 (expert), -1 (novice)}.
  - Split 70/15/15 train/val/test AU NIVEAU ESSAI, stratifie par groupe (+ participant).
    Ce mode "faithful" tolere la fuite inter-essais du papier (un participant peut
    apparaitre dans train et test). Le mode rigoureux (LOPO) viendra en Phase 2.
  - Loss Modele A : MSE(score_essai, label). Adam lr=1e-3, batch=32, epochs=50,
    early stopping patience=8 sur val loss, dropout=0.30. SEED fixee + loggee.

Usage :
    python train_hybrid1.py --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_hybrid1 import (
    GROUP4_ORDER,
    Trial,
    TrialDataset,
    build_trials,
    collate_pad,
    compute_norm_stats,
    load_raw_trials,
    select_extremes,
    stratified_trial_split,
)
from models_hybrid1 import Hybrid1Config, Hybrid1ModelA, count_params

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("train_hybrid1")


def set_full_determinism(seed: int) -> None:
    """Reproductibilité inter-lancements. À appeler tout en haut de main()."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # requis pour cuBLAS déterministe
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # sinon cuDNN autotune = non-déterministe
    try:
        torch.use_deterministic_algorithms(True)  # strict ; peut lever une erreur
        logger.info("[determinism] use_deterministic_algorithms=True (strict)")
    except Exception as e:
        logger.warning(f"[determinism] strict indisponible, repli best-effort: {e}")


def make_loader(trials, mean, std, batch_size, shuffle, seed: int):
    ds = TrialDataset(trials, mean, std, target="label_extreme")
    g = torch.Generator()
    g.manual_seed(seed)

    def _worker_init(worker_id: int) -> None:
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_pad,
        num_workers=0,  # 0 = pas de fork ; déterminisme DataLoader garanti
        worker_init_fn=_worker_init,
        generator=g,
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Hybrid1 Model A (faithful mode)")
    ap.add_argument("--pkl", "--data", type=Path, default=Path("data/trial_tensor_v2.pkl"),
                    dest="pkl",
                    help="tenseur d'essais (alias --data accepté)")
    ap.add_argument("--out", type=Path, default=Path("results/hybrid1_faithful"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.30)
    # Pooling (défaut gap = reproduction A0 ; attn = contribution proposée)
    ap.add_argument("--pool-type", choices=["gap", "max", "attn"], default="gap")
    ap.add_argument("--pool-heads", type=int, default=1)
    ap.add_argument("--pool-tau", type=float, default=1.0)
    ap.add_argument("--pool-d-attn", type=int, default=None,
                    help="dim attention (défaut = d_model=64)")
    ap.add_argument("--pool-att-dropout", type=float, default=0.0)
    ap.add_argument("--pool-time-shuffle", action="store_true",
                    help="contrôle A6 : permute timesteps avant pooling")
    # SWA — condition DISTINCTE (ex. A2+SWA), jamais mélangée à l'ablation A0/A2/A6
    ap.add_argument("--swa", action="store_true",
                    help="Stochastic Weight Averaging (label condition séparée)")
    ap.add_argument("--swa-start", type=int, default=None,
                    help="époque de début SWA (défaut = best_epoch val extrêmes)")
    ap.add_argument("--swa-last-k", type=int, default=5,
                    help="fallback : moyenne des K dernières époques si pas de best")
    args = ap.parse_args()

    # Première chose : seeds + cuDNN/cuBLAS déterministes (avant tout CUDA).
    set_full_determinism(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = args.out / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("TRAIN Hybrid1 Model A — MODE FAITHFUL")
    logger.info("=" * 70)
    logger.info(f"[config] seed={args.seed} device={device} max_len={args.max_len}")
    logger.info(f"[config] lr={args.lr} batch={args.batch_size} epochs={args.epochs} "
                f"patience={args.patience} dropout={args.dropout}")
    logger.info(f"[config] pool_type={args.pool_type} heads={args.pool_heads} "
                f"tau={args.pool_tau} time_shuffle={args.pool_time_shuffle} "
                f"swa={args.swa}")
    logger.info(f"[config] pkl={args.pkl}")

    raw = load_raw_trials(str(args.pkl))
    all_trials = build_trials(raw, max_len=args.max_len)
    extremes = select_extremes(all_trials)
    n_exp = sum(1 for t in extremes if t.label_extreme == 1)
    n_nov = sum(1 for t in extremes if t.label_extreme == -1)
    logger.info(f"[data] {len(all_trials)} trials totaux ; "
                f"{len(extremes)} extremes (expert={n_exp}, novice={n_nov})")

    train_t, val_t, test_t = stratified_trial_split(extremes, seed=args.seed)

    def grp_counts(ts):
        d = {}
        for t in ts:
            d[t.group4] = d.get(t.group4, 0) + 1
        return {g: d.get(g, 0) for g in GROUP4_ORDER if d.get(g, 0)}

    logger.info(f"[split] train={len(train_t)} {grp_counts(train_t)} | "
                f"val={len(val_t)} {grp_counts(val_t)} | "
                f"test={len(test_t)} {grp_counts(test_t)}")

    # Normalisation : stats calculees sur le TRAIN uniquement.
    mean, std = compute_norm_stats(train_t)
    logger.info(f"[norm] mean/std calcules sur {len(train_t)} trials train")

    train_loader = make_loader(
        train_t, mean, std, args.batch_size, shuffle=True, seed=args.seed,
    )
    val_loader = make_loader(
        val_t, mean, std, args.batch_size, shuffle=False, seed=args.seed,
    )
    test_loader = make_loader(
        test_t, mean, std, args.batch_size, shuffle=False, seed=args.seed,
    )

    cfg = Hybrid1Config(
        n_features=10,
        max_len=args.max_len,
        dropout=args.dropout,
        pool_type=args.pool_type,
        pool_heads=args.pool_heads,
        pool_d_attn=args.pool_d_attn,
        pool_tau=args.pool_tau,
        pool_att_dropout=args.pool_att_dropout,
        pool_time_shuffle=args.pool_time_shuffle,
    )
    model = Hybrid1ModelA(cfg).to(device)
    logger.info(f"[model] Hybrid1ModelA params={count_params(model):,} "
                f"pool={args.pool_type}")

    from src.models.swa import offline_average_checkpoints, resolve_start_swa_epoch

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    wait = 0
    history = []
    # Snapshots pour SWA (val extrêmes guide start ; milieu jamais vu)
    epoch_states: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for X, mask, y in train_loader:
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X, key_padding_mask=mask)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            tot += loss.item() * X.size(0)
            n += X.size(0)
        train_loss = tot / max(n, 1)
        val_loss = evaluate_loss(model, val_loader, device, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info(f"[epoch {epoch:3d}] train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f}")

        if args.swa:
            epoch_states.append(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                logger.info(f"[early-stop] patience {args.patience} atteinte "
                            f"(best epoch {best_epoch}, val_loss={best_val:.4f})")
                break

    swa_meta = {"swa": False}
    if args.swa and epoch_states:
        start = resolve_start_swa_epoch(
            best_epoch=best_epoch,
            total_epochs_ran=len(epoch_states),
            swa_start=args.swa_start,
            swa_last_k=args.swa_last_k,
        )
        start = min(start, len(epoch_states))
        selected = epoch_states[start - 1 :]
        swa_state = offline_average_checkpoints(selected)
        best_state = swa_state
        swa_meta = {
            "swa": True,
            "start_swa_epoch": start,
            "n_averaged": len(selected),
            "has_batchnorm": False,
            "update_bn_applied": False,
            "note": "SWA condition distincte ; start sur plateau val EXTRÊMES",
        }
        logger.info(f"[SWA] moyenne de {len(selected)} checkpoints "
                    f"(epochs {start}..{len(epoch_states)}) — pas de BN à maj")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss = evaluate_loss(model, test_loader, device, loss_fn)
    logger.info(f"[done] best_epoch={best_epoch} best_val_loss={best_val:.4f} "
                f"test_loss={test_loss:.4f} swa={swa_meta.get('swa')}")

    # Sauvegardes
    torch.save(best_state, out_dir / "model_A_best.pt")
    with open(out_dir / "norm_stats.pkl", "wb") as f:
        pickle.dump({"mean": mean, "std": std}, f)

    split_ids = {
        "train": [(t.participant, t.trial) for t in train_t],
        "val": [(t.participant, t.trial) for t in val_t],
        "test": [(t.participant, t.trial) for t in test_t],
    }
    with open(out_dir / "split_ids.pkl", "wb") as f:
        pickle.dump(split_ids, f)

    git_sha = None
    try:
        import subprocess
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        pass

    config = {
        "mode": "faithful",
        "seed": args.seed,
        "pkl": str(args.pkl),
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
        "dropout": args.dropout,
        "pool_type": args.pool_type,
        "pool_heads": args.pool_heads,
        "pool_tau": args.pool_tau,
        "pool_d_attn": args.pool_d_attn,
        "pool_att_dropout": args.pool_att_dropout,
        "pool_time_shuffle": args.pool_time_shuffle,
        "determinism": "full",
        "git_sha": git_sha,
        **swa_meta,
        "n_train": len(train_t), "n_val": len(val_t), "n_test": len(test_t),
        "train_groups": grp_counts(train_t),
        "val_groups": grp_counts(val_t),
        "test_groups": grp_counts(test_t),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "n_params": count_params(model),
    }
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info(f"[save] modele + stats + config -> {out_dir}")


if __name__ == "__main__":
    main()
