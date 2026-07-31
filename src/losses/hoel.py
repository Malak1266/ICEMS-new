"""
HOEL — Hierarchical Ordinal Expertise Loss
============================================
5-term composite loss for continuous surgical expertise scoring.

Terms:
  1. L_mse   : Tier-weighted MSE (base regression signal)
  2. L_asym  : Asymmetric directional penalty (protects experts from under-estimation)
  3. L_pair  : Pairwise monotonicity (prevents rank inversions across all levels)
  4. L_top   : Top-tier reinforced ranking (extra margin for PGY5-Staff zone)
  5. L_anchor: Extreme anchoring (pushes MS low, Fellow/Staff high)

Usage:
    criterion = HOEL()
    loss_dict = criterion(preds, targets, tiers, ranks)
    loss_dict["loss"].backward()

Author: ICEMS project — Malek Kamel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HOEL(nn.Module):
    """Hierarchical Ordinal Expertise Loss — 5 terms."""

    def __init__(
        self,
        # --- Lambda coefficients (from the plan) ---
        lambda_asym: float = 0.20,
        lambda_pair: float = 0.30,
        lambda_top: float = 0.40,
        lambda_anchor: float = 0.08,
        # --- Tier weights for MSE ---
        tier_weights: dict = None,
        # --- Pairwise margins ---
        pair_margin_per_rank: float = 0.05,
        top_margin: float = 0.15,
        top_min_rank: int = 5,  # ranks >= 5 = {pgy5, pgy6, fellow, staff}
        # --- Asymmetric coefficients ---
        asym_under: dict = None,   # {rank: alpha} for under-estimation
        asym_over_alpha: float = 1.0,
        asym_under_threshold: float = 0.50,   # target >= this triggers under-est penalty
        asym_over_threshold: float = -0.50,   # target <= this triggers over-est penalty
        # --- Anchor thresholds ---
        anchor_ms_max: float = -0.75,     # MS should stay below this
        anchor_fellow_min: float = 0.70,  # Fellow should stay above this
        anchor_staff_min: float = 0.85,   # Staff should stay above this
    ):
        super().__init__()

        # Lambdas
        self.lambda_asym = lambda_asym
        self.lambda_pair = lambda_pair
        self.lambda_top = lambda_top
        self.lambda_anchor = lambda_anchor

        # Tier weights: {tier_int: weight_float}
        self.tier_weights = tier_weights or {0: 1.0, 1: 2.5, 2: 4.0}

        # Pairwise
        self.pair_margin_per_rank = pair_margin_per_rank
        self.top_margin = top_margin
        self.top_min_rank = top_min_rank

        # Asymmetric under-estimation weights per rank
        # Only ranks with target >= asym_under_threshold are affected
        self.asym_under = asym_under or {6: 1.5, 7: 3.5, 8: 5.0}
        self.asym_over_alpha = asym_over_alpha
        self.asym_under_threshold = asym_under_threshold
        self.asym_over_threshold = asym_over_threshold

        # Anchors
        self.anchor_ms_max = anchor_ms_max
        self.anchor_fellow_min = anchor_fellow_min
        self.anchor_staff_min = anchor_staff_min

    def forward(
        self,
        preds: torch.Tensor,    # (B,) or (B,1) — model predictions
        targets: torch.Tensor,  # (B,) or (B,1) — regression targets
        tiers: torch.Tensor,    # (B,) — integer tier {0, 1, 2}
        ranks: torch.Tensor,    # (B,) — integer rank 0..8
    ) -> dict:
        """
        Returns dict with 'loss' (total, for backward) and individual terms (detached).
        """
        preds = preds.view(-1)
        targets = targets.view(-1)
        tiers = tiers.view(-1).long()
        ranks = ranks.view(-1).long()
        B = preds.size(0)

        # ================================================================
        # Term 1: Tier-weighted MSE
        # ================================================================
        residuals_sq = (preds - targets).pow(2)

        # Build per-sample tier weight tensor
        w = torch.ones(B, device=preds.device, dtype=preds.dtype)
        for tier_val, tier_w in self.tier_weights.items():
            mask = (tiers == tier_val)
            w[mask] = tier_w

        l_mse = (w * residuals_sq).mean()

        # ================================================================
        # Term 2: Asymmetric directional penalty
        # ================================================================
        l_asym = torch.tensor(0.0, device=preds.device, dtype=preds.dtype)

        # 2a: Under-estimation of experts (target >= threshold, pred too low)
        under_mask = (targets >= self.asym_under_threshold)
        if under_mask.any():
            under_error = F.relu(targets - preds).pow(2)  # only when pred < target
            # Build alpha per sample based on rank
            alpha_under = torch.zeros(B, device=preds.device, dtype=preds.dtype)
            for rank_val, alpha_val in self.asym_under.items():
                alpha_under[ranks == rank_val] = alpha_val
            l_asym_under = (under_mask.float() * alpha_under * under_error).sum() / under_mask.float().sum().clamp(min=1.0)
        else:
            l_asym_under = torch.tensor(0.0, device=preds.device)

        # 2b: Over-estimation of juniors (target <= threshold, pred too high)
        over_mask = (targets <= self.asym_over_threshold)
        if over_mask.any():
            over_error = F.relu(preds - targets).pow(2)  # only when pred > target
            l_asym_over = (over_mask.float() * self.asym_over_alpha * over_error).sum() / over_mask.float().sum().clamp(min=1.0)
        else:
            l_asym_over = torch.tensor(0.0, device=preds.device)

        l_asym = l_asym_under + l_asym_over

        # ================================================================
        # Term 3: Pairwise monotonicity (all pairs in batch)
        # ================================================================
        # For pairs (i,j) where rank_i > rank_j:
        #   penalize if pred_i < pred_j + margin * (rank_i - rank_j)
        rank_diff = ranks.unsqueeze(0) - ranks.unsqueeze(1)    # (B, B), [i,j] = rank_i - rank_j
        pred_diff = preds.unsqueeze(0) - preds.unsqueeze(1)    # (B, B), [i,j] = pred_i - pred_j

        # Mask: pairs where rank_i > rank_j (positive rank_diff)
        pair_mask = (rank_diff > 0).float()
        n_pairs = pair_mask.sum().clamp(min=1.0)

        # Required margin scales with rank difference
        required_margin = self.pair_margin_per_rank * rank_diff.float()  # (B, B)

        # Violation: pred_diff < required_margin
        pair_violations = F.relu(required_margin - pred_diff).pow(2)

        l_pair = (pair_mask * pair_violations).sum() / n_pairs

        # ================================================================
        # Term 4: Top-tier reinforced ranking (ranks >= top_min_rank only)
        # ================================================================
        top_mask_i = (ranks >= self.top_min_rank)  # (B,)
        # Build a mask for pairs where BOTH i and j are top-tier and rank_i > rank_j
        top_pair_mask = (
            top_mask_i.unsqueeze(0).float()         # i is top-tier
            * top_mask_i.unsqueeze(1).float()        # j is top-tier
            * (rank_diff > 0).float()                # rank_i > rank_j
        )
        n_top_pairs = top_pair_mask.sum().clamp(min=1.0)

        top_violations = F.relu(self.top_margin - pred_diff).pow(2)

        l_top = (top_pair_mask * top_violations).sum() / n_top_pairs

        # ================================================================
        # Term 5: Anchor constraints
        # ================================================================
        l_anchor = torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
        anchor_count = 0.0

        # MS (rank=0): penalize if pred > anchor_ms_max
        ms_mask = (ranks == 0)
        if ms_mask.any():
            ms_penalty = F.relu(preds[ms_mask] - self.anchor_ms_max).pow(2)
            l_anchor = l_anchor + ms_penalty.sum()
            anchor_count += ms_mask.float().sum().item()

        # Fellow (rank=7): penalize if pred < anchor_fellow_min
        fellow_mask = (ranks == 7)
        if fellow_mask.any():
            fellow_penalty = F.relu(self.anchor_fellow_min - preds[fellow_mask]).pow(2)
            l_anchor = l_anchor + fellow_penalty.sum()
            anchor_count += fellow_mask.float().sum().item()

        # Staff (rank=8): penalize if pred < anchor_staff_min
        staff_mask = (ranks == 8)
        if staff_mask.any():
            staff_penalty = F.relu(self.anchor_staff_min - preds[staff_mask]).pow(2)
            l_anchor = l_anchor + staff_penalty.sum()
            anchor_count += staff_mask.float().sum().item()

        if anchor_count > 0:
            l_anchor = l_anchor / anchor_count

        # ================================================================
        # Total loss
        # ================================================================
        total = (
            l_mse
            + self.lambda_asym * l_asym
            + self.lambda_pair * l_pair
            + self.lambda_top * l_top
            + self.lambda_anchor * l_anchor
        )

        return {
            "loss": total,
            "mse": l_mse.detach(),
            "asym": l_asym.detach(),
            "pair": l_pair.detach(),
            "top": l_top.detach(),
            "anchor": l_anchor.detach(),
        }

    def __repr__(self):
        return (
            f"HOEL(\n"
            f"  lambda_asym={self.lambda_asym}, lambda_pair={self.lambda_pair},\n"
            f"  lambda_top={self.lambda_top}, lambda_anchor={self.lambda_anchor},\n"
            f"  tier_weights={self.tier_weights},\n"
            f"  pair_margin_per_rank={self.pair_margin_per_rank}, top_margin={self.top_margin},\n"
            f"  asym_under={self.asym_under}\n"
            f")"
        )
