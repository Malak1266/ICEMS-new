"""
mae_pretrain.py
===============
Pré-entraînement MAE PyTorch (4 canaux) — compatible avec mae_encoder.py / ablation.

Hyperparamètres alignés Narval :
  mask=0.40, embed=64, heads=4, blocks=3, ff=128

Sorties :
  {output}/best_encoder.pt   — pour ablation condition A
  {output}/history.json

Usage :
    python src/mae_pretrain.py --data data/X_pretrain_aug_4ch.npy --output results/mae_aug_run1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from mae_encoder import (  # noqa: E402
    MAEEncoder4ch,
    MAE_EMBED_DIM,
    MAE_FF_DIM,
    MAE_IN_DIM,
    MAE_MAX_LEN,
)

WIN_LEN = MAE_MAX_LEN
IN_DIM = MAE_IN_DIM


class MAEDecoder(nn.Module):
    def __init__(self, embed_dim: int = MAE_EMBED_DIM, out_dim: int = IN_DIM, ff_dim: int = MAE_FF_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MAEModel(nn.Module):
    def __init__(self, mask_ratio: float = 0.4):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.encoder = MAEEncoder4ch()
        self.decoder = MAEDecoder()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, MAE_EMBED_DIM))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x : (B, T, 4)
        Retourne pred, target, mask (bool, True = masqué).
        """
        b, t, c = x.shape
        z = self.encoder.input_proj(x) + self.encoder.pos_emb[:, :t, :]

        n_mask = max(1, int(t * self.mask_ratio))
        mask = torch.zeros(b, t, dtype=torch.bool, device=x.device)
        for i in range(b):
            idx = torch.randperm(t, device=x.device)[:n_mask]
            mask[i, idx] = True

        mask_exp = mask.unsqueeze(-1).expand_as(z)
        z_masked = torch.where(mask_exp, self.mask_token.expand(b, t, -1), z)

        h = z_masked
        for block in self.encoder.blocks:
            h = block(h)
        h = self.encoder.norm(h)
        pred = self.decoder(h)
        return pred, x, mask


def train_epoch(model, loader, opt, device) -> float:
    model.train()
    losses = []
    for (xb,) in loader:
        xb = xb.to(device)
        pred, target, mask = model(xb)
        loss = ((pred - target) ** 2)[mask].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def eval_epoch(model, loader, device) -> float:
    model.eval()
    losses = []
    for (xb,) in loader:
        xb = xb.to(device)
        pred, target, mask = model(xb)
        loss = ((pred - target) ** 2)[mask].mean()
        losses.append(loss.item())
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser(description="Pré-entraînement MAE 4ch (PyTorch).")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("results/mae_aug_run1"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mask", type=float, default=0.40)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X = np.load(args.data).astype(np.float32)
    if X.shape[1:] != (WIN_LEN, IN_DIM):
        raise ValueError(f"Attendu (N, {WIN_LEN}, {IN_DIM}), reçu {X.shape}")

    n = len(X)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[tr_idx])),
        batch_size=args.batch, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[val_idx])),
        batch_size=args.batch, shuffle=False,
    )

    model = MAEModel(mask_ratio=args.mask).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    args.output.mkdir(parents=True, exist_ok=True)
    best_val, best_ep, wait = float("inf"), 0, 0
    history = {"train_loss": [], "val_loss": []}

    print("=" * 60)
    print("  MAE pretrain PyTorch")
    print("=" * 60)
    print(f"  data   : {args.data}  {X.shape}")
    print(f"  train  : {len(tr_idx)}  val : {len(val_idx)}")
    print(f"  device : {device}")

    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, opt, device)
        va_loss = eval_epoch(model, val_loader, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        if va_loss < best_val - 1e-5:
            best_val, best_ep, wait = va_loss, ep, 0
            torch.save(
                {"encoder": model.encoder.state_dict(), "val_loss": best_val, "epoch": ep},
                args.output / "best_encoder.pt",
            )
        else:
            wait += 1

        if ep % 10 == 0 or ep == 1:
            print(f"  epoch {ep:3d}  train={tr_loss:.4f}  val={va_loss:.4f}  best={best_val:.4f}@{best_ep}")

        if wait >= args.patience:
            print(f"  Early stopping @ epoch {ep}")
            break

    meta = {
        "best_val_loss": best_val,
        "best_epoch": best_ep,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(val_idx)),
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    with open(args.output / "history.json", "w", encoding="utf-8") as f:
        json.dump({**meta, **history}, f, indent=2)

    print(f"\n✅ best_encoder.pt → {args.output / 'best_encoder.pt'}  (val_loss={best_val:.4f})")


if __name__ == "__main__":
    main()
