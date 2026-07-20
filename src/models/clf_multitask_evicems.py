"""
clf_multitask_evicems.py
========================
Classificateur multi-tache : backbone Hybrid1 (LSTM + Transformer)
avec deux tetes de classification :
    - y4 : expertise (Student / Junior / Senior / Expert)
    - y9 : niveau de formation (ms ... staff)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.hybrid1_evicems import TransformerBlock


class MultiTaskEVICEMS(nn.Module):
    """Backbone partage + deux tetes softmax (y4 et y9)."""

    def __init__(
        self,
        n_features: int = 10,
        seq_len: int = 800,
        n_classes_y4: int = 4,
        n_classes_y9: int = 9,
        d_model: int = 64,
        n_heads: int = 4,
        key_dim: int = 32,
        ff_dim: int = 128,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.n_classes_y4 = n_classes_y4
        self.n_classes_y9 = n_classes_y9

        self.lstm = nn.LSTM(n_features, 128, num_layers=1, batch_first=True)
        self.lstm_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(128, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.transformer = TransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            key_dim=key_dim,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        self.shared = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_y4 = nn.Linear(32, n_classes_y4)
        self.head_y9 = nn.Linear(32, n_classes_y9)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        h = self.lstm_dropout(h)
        h = self.proj(h)
        h = h + self.pos_embedding
        h = self.transformer(h)
        return h.mean(dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.encode(x)
        shared = self.shared(pooled)
        return self.head_y4(shared), self.head_y9(shared)
