"""
mae_encoder.py
==============
Encodeur MAE 4 canaux (aligné Narval : embed=64, heads=4, blocks=3, ff=128).

Solution C : le même encodeur est appliqué indépendamment à chaque instrument
(bipolar, scissors, cavitron) avant le GRU causal.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Indices cinématiques dans continuous_per_trial.pkl (T, 10)
INSTRUMENT_KIN_SLICES = [
    (0, 1, 2),   # bipolar
    (3, 4, 5),   # scissors
    (6, 7, 8),   # cavitron
]

MAE_IN_DIM = 4
MAE_EMBED_DIM = 64
MAE_N_HEADS = 4
MAE_N_BLOCKS = 3
MAE_FF_DIM = 128
MAE_MAX_LEN = 32


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class MAEEncoder4ch(nn.Module):
    """
    Encodeur transformer 4→64 (sans masquage, mode inférence pour le scorer).
    Architecture identique à scripts/mae_pretrain.py sur Narval.
    """

    def __init__(
        self,
        in_dim: int = MAE_IN_DIM,
        embed_dim: int = MAE_EMBED_DIM,
        n_heads: int = MAE_N_HEADS,
        n_blocks: int = MAE_N_BLOCKS,
        ff_dim: int = MAE_FF_DIM,
        max_len: int = MAE_MAX_LEN,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ff_dim, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

    def _positional_encoding(self, t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Interpole l'embedding positionnel appris (32) vers la longueur T du trial."""
        if t <= self.max_len:
            return self.pos_emb[:, :t, :]
        pos = F.interpolate(
            self.pos_emb.transpose(1, 2),
            size=t,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        return pos.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, 4) → (B, T, embed_dim)."""
        _, t, _ = x.shape
        pos = self._positional_encoding(t, x.device, x.dtype)
        h = self.input_proj(x) + pos
        for block in self.blocks:
            h = block(h)
        return self.norm(h)


def _strip_prefix(state: Dict[str, torch.Tensor], prefixes: Tuple[str, ...]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, val in state.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        out[new_key] = val
    return out


def load_encoder_checkpoint(
    encoder: MAEEncoder4ch,
    ckpt_path: Path,
    freeze: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Charge best_encoder.pt (Narval) avec correspondance souple des clés.
    Retourne (missing_keys, unexpected_keys).
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        if "encoder" in raw and isinstance(raw["encoder"], dict):
            state = raw["encoder"]
        elif "state_dict" in raw:
            state = raw["state_dict"]
        elif "model" in raw and isinstance(raw["model"], dict):
            state = raw["model"]
        else:
            state = raw
    else:
        raise ValueError(f"Format checkpoint non supporté : {ckpt_path}")

    state = _strip_prefix(state, ("encoder.", "module.encoder.", "module."))
    # Ne garder que les tensors compatibles avec MAEEncoder4ch
    enc_state = {k: v for k, v in state.items() if not k.startswith("decoder")}
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)

    if freeze:
        for param in encoder.parameters():
            param.requires_grad = False
    return list(missing), list(unexpected)


def extract_instrument_4ch(
    x: torch.Tensor,
    kin_slice: Tuple[int, int, int],
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Construit (B, T, 4) = [vel, acc, jerk, valid_ratio] pour un instrument."""
    kin = x[..., list(kin_slice)]
    vr = valid_mask.unsqueeze(-1)
    return torch.cat([kin, vr], dim=-1)


class MultiInstrumentMAEFront(nn.Module):
    """Solution C : encodeur MAE partagé sur les 3 instruments."""

    def __init__(self, encoder: MAEEncoder4ch):
        super().__init__()
        self.encoder = encoder
        self.out_dim = encoder.embed_dim * len(INSTRUMENT_KIN_SLICES)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        embs: List[torch.Tensor] = []
        for kin_slice in INSTRUMENT_KIN_SLICES:
            x4 = extract_instrument_4ch(x, kin_slice, valid_mask)
            embs.append(self.encoder(x4))
        return torch.cat(embs, dim=-1)
