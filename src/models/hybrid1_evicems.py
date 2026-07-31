"""
hybrid1_evicems.py
==================
Architecture Hybrid1 du protocole EVICEMS (supplément eAppendix 1, PyTorch pur).

Schéma (ordre STRICT du supplément / eApp 1) :
    Input (B, L, n_features), L <= max_len
      → LSTM(n_features → 128, 1 couche, batch_first)          (B, L, 128)
      → Dropout(0.30)
      → Linear(128 → d_model=64)                                 (B, L, 64)
      → + pos_embedding APPRENABLE (max_len, 64), tronqué à L
      → 1 × TransformerBlock(d_model=64, n_heads=4, key_dim=32, ff_dim=128)
      → Pooling (défaut : Masked GAP ; option : AttentionPooling1D)  (B, 64|H*64)
      → Linear(pool_dim → 32) → ReLU → Dropout(0.30) → Linear(32 → 1) → Tanh
    Output (B, 1) dans [-1, 1]

Pooling (extension méthodologique SPIE / perspectives) :
    pool_type ∈ {"gap", "max", "attn"} — défaut "gap" = reproduction A0 bit-identique
    (appelle masked_mean, aucun paramètre d'attention). Voir attention_pooling.py.

Point d'attention (key_dim=32 != d_model/n_heads=16) :
    L'attention multi-têtes est implémentée MANUELLEMENT (Q/K/V séparés) car la
    dimension interne n_heads*key_dim = 128 diffère de d_model = 64. nn.MultiheadAttention
    imposerait head_dim = d_model/n_heads = 16, ce qui ne reproduirait PAS le supplément
    (key_dim=32 de la couche MultiHeadAttention Keras).

Cette architecture est purement additive : elle ne modifie ni CausalGRUScorer ni
HybridLSTMTransformerScorer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from src.models.attention_pooling import AttentionPooling1D, masked_max, time_shuffle

PoolType = Literal["gap", "max", "attn"]


def _make_sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    """Encodage positionnel sinusoïdal (Vaswani et al.) — shape (1, seq_len, d_model)."""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


@dataclass
class Hybrid1Config:
    n_features: int = 10
    max_len: int = 1024
    lstm_hidden: int = 128
    d_model: int = 64
    nhead: int = 4
    key_dim: int = 32
    ff_dim: int = 128
    dropout: float = 0.30
    # --- pooling (défaut = reproduction A0) ---
    pool_type: PoolType = "gap"
    pool_heads: int = 1
    pool_d_attn: int | None = None  # None → d_model
    pool_tau: float = 1.0
    pool_att_dropout: float = 0.0
    # contrôle A6 : permute les timesteps juste avant le pooling (train+eval)
    pool_time_shuffle: bool = False


class TransformerBlock(nn.Module):
    """Bloc Transformer interne réutilisable (B, T, d_model) → (B, T, d_model).

    Reproduit exactement le bloc du supplément :
      - Attention Q/K/V séparée, dimension interne n_heads*key_dim (!= d_model)
      - Dropout sur la sortie attention, résiduel + LayerNorm (post-LN)
      - FFN Linear(d_model→ff_dim, ReLU)→Linear(ff_dim→d_model)
      - Dropout sur la sortie FFN, résiduel + LayerNorm
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        key_dim: int = 32,
        ff_dim: int = 128,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.key_dim = key_dim
        self.inner_dim = n_heads * key_dim  # 128 — dimension interne attention

        self.q_proj = nn.Linear(d_model, self.inner_dim)
        self.k_proj = nn.Linear(d_model, self.inner_dim)
        self.v_proj = nn.Linear(d_model, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert x.dim() == 3, f"TransformerBlock attend (B, T, d_model), reçu {tuple(x.shape)}"
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.key_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.key_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.key_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.key_dim)

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_dropout(attn)
        ctx = torch.matmul(attn, v)

        ctx = ctx.transpose(1, 2).contiguous().view(B, T, self.inner_dim)
        attn_out = self.attn_dropout(self.out_proj(ctx))

        x = self.norm1(x + attn_out)

        ffn_out = self.ffn_dropout(self.ffn(x))
        x = self.norm2(x + ffn_out)

        assert x.shape == (B, T, D), f"sortie TransformerBlock {tuple(x.shape)} != {(B, T, D)}"
        return x


def masked_mean(
    h: torch.Tensor,
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Global average pooling sur les frames NON paddees."""
    if key_padding_mask is None:
        return h.mean(dim=1)
    valid = (~key_padding_mask).unsqueeze(-1).float()
    summed = (h * valid).sum(dim=1)
    counts = valid.sum(dim=1).clamp_min(1.0)
    return summed / counts


class Hybrid1EVICEMS(nn.Module):
    """Hybride LSTM + (1 bloc) Transformer pour scoring d'expertise trial-level."""

    def __init__(
        self,
        n_features: int | Hybrid1Config | None = None,
        seq_len: int | None = None,
        *,
        d_model: int = 64,
        n_heads: int = 4,
        key_dim: int = 32,
        ff_dim: int = 128,
        dropout: float = 0.30,
        lstm_hidden: int = 128,
    ) -> None:
        super().__init__()
        if isinstance(n_features, Hybrid1Config):
            cfg = n_features
        elif n_features is None:
            cfg = Hybrid1Config(
                d_model=d_model,
                nhead=n_heads,
                key_dim=key_dim,
                ff_dim=ff_dim,
                dropout=dropout,
                lstm_hidden=lstm_hidden,
            )
        else:
            cfg = Hybrid1Config(
                n_features=n_features,
                max_len=seq_len if seq_len is not None else 1024,
                d_model=d_model,
                nhead=n_heads,
                key_dim=key_dim,
                ff_dim=ff_dim,
                dropout=dropout,
                lstm_hidden=lstm_hidden,
            )

        self.cfg = cfg
        self.n_features = cfg.n_features
        self.max_len = cfg.max_len
        self.d_model = cfg.d_model
        self.pool_type: PoolType = cfg.pool_type
        self.pool_time_shuffle = bool(cfg.pool_time_shuffle)

        if cfg.pool_type not in ("gap", "max", "attn"):
            raise ValueError(f"pool_type invalide: {cfg.pool_type!r}")

        self.lstm = nn.LSTM(
            cfg.n_features, cfg.lstm_hidden, num_layers=1, batch_first=True,
        )
        self.lstm_dropout = nn.Dropout(cfg.dropout)
        self.proj = nn.Linear(cfg.lstm_hidden, cfg.d_model)

        self.pos_embedding = nn.Parameter(torch.zeros(cfg.max_len, cfg.d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.transformer = TransformerBlock(
            d_model=cfg.d_model,
            n_heads=cfg.nhead,
            key_dim=cfg.key_dim,
            ff_dim=cfg.ff_dim,
            dropout=cfg.dropout,
        )

        # Attention pooling uniquement si demandé (sinon aucun paramètre ajouté → A0)
        self.attn_pool: AttentionPooling1D | None = None
        if cfg.pool_type == "attn":
            self.attn_pool = AttentionPooling1D(
                d_model=cfg.d_model,
                d_attn=cfg.pool_d_attn,
                heads=cfg.pool_heads,
                temperature=cfg.pool_tau,
                att_dropout=cfg.pool_att_dropout,
            )
            pool_dim = self.attn_pool.out_dim
        else:
            pool_dim = cfg.d_model

        self.head = nn.Sequential(
            nn.Linear(pool_dim, 32),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def _pool(
        self,
        h: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.pool_type == "gap":
            return masked_mean(h, key_padding_mask)
        if self.pool_type == "max":
            return masked_max(h, key_padding_mask)
        assert self.attn_pool is not None
        return self.attn_pool(h, key_padding_mask=key_padding_mask)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert x.dim() == 3, f"Hybrid1EVICEMS attend (B, L, n_features), reçu {tuple(x.shape)}"
        B, L, F = x.shape
        assert L <= self.max_len, f"L={L} > max_len={self.max_len}"
        assert F == self.n_features, (
            f"n_features reçu {F} != n_features attendu {self.n_features}."
        )

        h, _ = self.lstm(x)
        h = self.lstm_dropout(h)
        h = self.proj(h)
        h = h + self.pos_embedding[:L].unsqueeze(0)
        h = self.transformer(h, key_padding_mask=key_padding_mask)

        # Contrôle A6 : détruit l'ordre temporel juste avant le pooling
        if self.pool_time_shuffle:
            h, key_padding_mask = time_shuffle(h, key_padding_mask)

        pooled = self._pool(h, key_padding_mask)
        out = self.head(pooled)

        assert out.shape == (B, 1), f"sortie {tuple(out.shape)} != {(B, 1)}"
        return out


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


Hybrid1ModelA = Hybrid1EVICEMS


if __name__ == "__main__":
    cfg_perpair = Hybrid1Config(n_features=1, max_len=5000, dropout=0.30)
    model = Hybrid1EVICEMS(cfg_perpair)
    print(f"Paramètres per-pair (n_features=1, max_len=5000) : {count_params(model):,}")

    B, L = 4, 200
    x = torch.randn(B, L, 1)
    pad = torch.zeros(B, L, dtype=torch.bool)
    pad[0, 150:] = True
    out = model(x, key_padding_mask=pad)
    assert out.shape == (B, 1)
    assert out.abs().max() <= 1.0
    print("Forward test : OK")

    param_names = [n for n, _ in model.named_parameters()]
    assert any("pos_embedding" in n for n in param_names), "PE doit etre apprenable"
    assert model.transformer.key_dim == 32
    print("Positional embedding APPRENABLE : OK")
    print("MHA custom key_dim=32 : OK")

    # Non-régression : pool_type=gap n'ajoute aucun paramètre d'attention
    assert model.attn_pool is None
    assert not any("attn_pool" in n for n in param_names)

    # Attention pooling : shapes + params
    cfg_attn = Hybrid1Config(n_features=1, max_len=5000, pool_type="attn", pool_heads=1)
    m_attn = Hybrid1EVICEMS(cfg_attn)
    out_a = m_attn(x, key_padding_mask=pad)
    assert out_a.shape == (B, 1)
    assert m_attn.attn_pool is not None
    n_extra = count_params(m_attn) - count_params(model)
    assert n_extra == 4224, f"params attention attendus 4224, got {n_extra}"
    print(f"Attention pooling : OK (+{n_extra} params)")

    # Bit-identité GAP : deux modèles gap, mêmes poids → même sortie
    torch.manual_seed(0)
    m1 = Hybrid1EVICEMS(Hybrid1Config(n_features=1, max_len=5000, pool_type="gap"))
    m2 = Hybrid1EVICEMS(Hybrid1Config(n_features=1, max_len=5000, pool_type="gap"))
    m2.load_state_dict(m1.state_dict())
    m1.eval()
    m2.eval()
    with torch.no_grad():
        y1 = m1(x, key_padding_mask=pad)
        y2 = m2(x, key_padding_mask=pad)
    assert torch.equal(y1, y2), "pool_type=gap non-régression échouée"
    print("Non-régression pool_type=gap : OK")
