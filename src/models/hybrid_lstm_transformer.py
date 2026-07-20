"""
hybrid_lstm_transformer.py
==========================
Hybrid LSTM-Transformer trial-level pour ICEMS LOPO.

    Input (B, T, n_features), T = seq_len
      → LSTM(n_features → lstm_hidden, bidirectional?)
      → Dropout
      → Linear(lstm_out → d_model)
      → + positional encoding (sinusoidal | learned)
      → n_transformer_blocks × TransformerBlock
      → mean(dim=1)
      → MLP head → Tanh
    Output (B, 1) ∈ [-1, 1]
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.hybrid1_evicems import TransformerBlock, _make_sinusoidal_pe


@dataclass
class HybridConfig:
    n_features: int = 13
    seq_len: int = 4000
    lstm_hidden: int = 128
    d_model: int = 128
    nhead: int = 4
    key_dim: int = 32
    n_transformer_blocks: int = 1
    ff_dim: int = 256
    dropout: float = 0.30
    pos_encoding: str = "sinusoidal"  # "sinusoidal" | "learned"
    bidirectional: bool = False


class HybridLSTMTransformer(nn.Module):
    """LSTM (optionnel Bi) + Transformer + pooling global."""

    def __init__(self, cfg: HybridConfig) -> None:
        super().__init__()
        self.cfg = cfg
        lstm_out = cfg.lstm_hidden * (2 if cfg.bidirectional else 1)

        self.lstm = nn.LSTM(
            cfg.n_features,
            cfg.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )
        self.lstm_dropout = nn.Dropout(cfg.dropout)
        self.proj = nn.Linear(lstm_out, cfg.d_model)

        if cfg.pos_encoding == "sinusoidal":
            pe = _make_sinusoidal_pe(cfg.seq_len, cfg.d_model)
            self.register_buffer("pos_embedding", pe)
            self.pos_learned = None
        elif cfg.pos_encoding == "learned":
            self.pos_learned = nn.Parameter(
                torch.zeros(1, cfg.seq_len, cfg.d_model)
            )
            nn.init.trunc_normal_(self.pos_learned, std=0.02)
            self.register_buffer("pos_embedding", None)
        else:
            raise ValueError(f"pos_encoding inconnu : {cfg.pos_encoding!r}")

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=cfg.d_model,
                n_heads=cfg.nhead,
                key_dim=cfg.key_dim,
                ff_dim=cfg.ff_dim,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_transformer_blocks)
        ])

        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, 32),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def _add_pos(self, h: torch.Tensor) -> torch.Tensor:
        if self.pos_learned is not None:
            return h + self.pos_learned
        return h + self.pos_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"attendu (B,T,F), reçu {tuple(x.shape)}"
        B, T, F = x.shape
        assert T == self.cfg.seq_len, f"T={T} != seq_len={self.cfg.seq_len}"
        assert F == self.cfg.n_features, f"F={F} != n_features={self.cfg.n_features}"

        h, _ = self.lstm(x)
        h = self.lstm_dropout(h)
        h = self.proj(h)
        h = self._add_pos(h)
        for block in self.blocks:
            h = block(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)
