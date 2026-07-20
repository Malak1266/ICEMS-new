"""
hierarchical_expertise_loss.py
==============================
Loss hiérarchique pour le scoring d'expertise continu.

Poids par tier (PGY / Fellow / Expert) + pénalité faux-Expert
(quadratique ou multiplicative) sur les prédictions trop élevées
pour les tiers non-Expert.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HierLossConfig:
    w_pgy: float = 1.0
    w_fellow: float = 2.0
    w_expert: float = 3.0
    lambda_false_expert: float = 0.0
    false_expert_mode: str = "quadratic"  # "quadratic" | "multiplicative"
    expert_tier: int = 2
    false_expert_threshold: float = 0.75


class HierarchicalExpertiseLoss(nn.Module):
    """MSE pondérée par tier + pénalité faux-Expert optionnelle."""

    def __init__(self, cfg: HierLossConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.false_expert_mode not in ("quadratic", "multiplicative"):
            raise ValueError(
                f"false_expert_mode inconnu : {cfg.false_expert_mode!r}"
            )

    def _tier_weights(self, tier: torch.Tensor) -> torch.Tensor:
        w = torch.empty_like(tier, dtype=torch.float32)
        w[tier == 0] = self.cfg.w_pgy
        w[tier == 1] = self.cfg.w_fellow
        w[tier == 2] = self.cfg.w_expert
        return w

    def _false_expert_penalty(
        self,
        pred: torch.Tensor,
        tier: torch.Tensor,
        base_mse: torch.Tensor,
    ) -> torch.Tensor:
        lam = self.cfg.lambda_false_expert
        if lam <= 0:
            return pred.new_zeros(())

        p = pred.squeeze(-1) if pred.dim() > 1 else pred
        non_expert = tier < self.cfg.expert_tier
        excess = torch.clamp(p - self.cfg.false_expert_threshold, min=0.0)
        mask = non_expert & (excess > 0)

        if not mask.any():
            return pred.new_zeros(())

        if self.cfg.false_expert_mode == "quadratic":
            per_sample = torch.zeros_like(p)
            per_sample[mask] = excess[mask] ** 2
        else:
            per_sample = torch.zeros_like(base_mse)
            per_sample[mask] = base_mse[mask] * (1.0 + excess[mask] ** 2)

        return lam * per_sample[mask].mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        tier: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : (B, 1) ou (B,)
        target : (B,)
        tier : (B,) int64 — 0=PGY, 1=Fellow, 2=Expert
        """
        p = pred.squeeze(-1) if pred.dim() > 1 else pred
        t = target.float()
        tier = tier.long()

        w = self._tier_weights(tier)
        per_mse = w * (p - t) ** 2
        loss_mse = per_mse.mean()
        penalty = self._false_expert_penalty(pred, tier, per_mse)
        return loss_mse + penalty
