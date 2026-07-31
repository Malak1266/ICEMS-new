"""
attention_pooling.py
====================
Pooling temporel à attention apprise (Bahdanau, mono- ou multi-tête).

Remplace le Global Average Pooling (masked_mean) dans Hybrid1EVICEMS.

Convention de masque (identique au reste du pipeline Hybrid1) :
  key_padding_mask : (B, L) bool, True = timestep paddé / à ignorer.

Les poids α sont stockés dans `last_alpha` pour visualisation SEULEMENT
(poids de pooling, PAS d'explication causale — Jain & Wallace 2019).
Les claims causaux restent sur l'occlusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling1D(nn.Module):
    """(B, L, D) → (B, heads * D). Attention additive Bahdanau."""

    def __init__(
        self,
        d_model: int = 64,
        d_attn: int | None = None,
        heads: int = 1,
        temperature: float = 1.0,
        att_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError(f"heads must be >= 1, got {heads}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.d_model = d_model
        self.d_attn = int(d_attn) if d_attn is not None else d_model
        self.heads = heads
        self.temperature = float(temperature)

        self.proj = nn.Linear(d_model, self.d_attn)  # W_a, b_a
        self.v = nn.Linear(self.d_attn, heads, bias=False)  # requête(s)
        self.drop = nn.Dropout(att_dropout)

        # (B, L, heads) — affichage only, jamais une claim XAI
        self.last_alpha: torch.Tensor | None = None

    @property
    def out_dim(self) -> int:
        return self.heads * self.d_model

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, D)
        key_padding_mask : (B, L) bool, True = pad
        """
        assert x.dim() == 3, f"AttentionPooling1D attend (B,L,D), reçu {tuple(x.shape)}"
        B, L, D = x.shape
        assert D == self.d_model, f"D={D} != d_model={self.d_model}"

        # u = tanh(H W_a + b_a) → (B, L, d_attn)
        u = torch.tanh(self.proj(x))
        # e = u · v → (B, L, heads)
        e = self.v(u)

        if key_padding_mask is not None:
            # True = pad → -inf avant softmax
            e = e.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf"))

        alpha = torch.softmax(e / self.temperature, dim=1)  # (B, L, heads)
        alpha = torch.nan_to_num(alpha, nan=0.0)  # lignes entièrement paddées
        alpha = self.drop(alpha)
        self.last_alpha = alpha.detach()

        # c_h = Σ_t α_{t,h} h_t → (B, heads, D) → (B, heads*D)
        ctx = torch.einsum("blh,bld->bhd", alpha, x)
        return ctx.reshape(B, self.heads * D)


def masked_max(
    h: torch.Tensor,
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Max-pooling sur les frames NON paddées — contrôle A1 (ablation)."""
    if key_padding_mask is None:
        return h.max(dim=1).values
    filled = h.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf"))
    out = filled.max(dim=1).values
    # si une ligne est entièrement paddée : -inf → 0
    return torch.nan_to_num(out, nan=0.0, neginf=0.0)


def time_shuffle(
    h: torch.Tensor,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Permute les timesteps (contrôle A6). Le masque suit la permutation
    pour que le pooling voie le même contenu, dans un ordre détruit.
    Retourne (h_shuffled, mask_shuffled).
    """
    B, L, _ = h.shape
    idx = torch.stack([torch.randperm(L, device=h.device) for _ in range(B)], dim=0)
    gather_idx = idx.unsqueeze(-1).expand_as(h)
    h_shuf = torch.gather(h, 1, gather_idx)
    if key_padding_mask is None:
        return h_shuf, None
    mask_shuf = torch.gather(key_padding_mask, 1, idx)
    return h_shuf, mask_shuf


# ---------------------------------------------------------------------------
# Sanity / non-régression (exécutable : python -m src.models.attention_pooling)
# ---------------------------------------------------------------------------
def _self_test() -> None:
    torch.manual_seed(0)
    B, L, D = 4, 32, 64
    x = torch.randn(B, L, D)
    pad = torch.zeros(B, L, dtype=torch.bool)
    pad[0, 20:] = True
    pad[1, 10:] = True

    pool = AttentionPooling1D(d_model=D, d_attn=64, heads=1, temperature=1.0)
    c = pool(x, key_padding_mask=pad)
    assert c.shape == (B, D), f"shape {tuple(c.shape)}"
    assert pool.last_alpha is not None
    a = pool.last_alpha
    assert a.shape == (B, L, 1)
    # Σα ≈ 1 sur frames valides
    sums = a.sum(dim=1).squeeze(-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5), sums
    # α = 0 sur padding
    assert (a[0, 20:, 0] == 0).all()
    assert (a[1, 10:, 0] == 0).all()

    # multi-tête
    pool4 = AttentionPooling1D(d_model=D, heads=4)
    c4 = pool4(x, key_padding_mask=pad)
    assert c4.shape == (B, 4 * D)

    # max pooling
    m = masked_max(x, pad)
    assert m.shape == (B, D)

    n_params = sum(p.numel() for p in pool.parameters())
    # W(64×64)+b(64)+v(64) = 4224
    assert n_params == 64 * 64 + 64 + 64, n_params
    print(f"AttentionPooling1D self-test OK (params mono-tête={n_params})")


if __name__ == "__main__":
    _self_test()
