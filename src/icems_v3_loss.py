"""
HierarchicalSurgicalLoss pour ICEMS V3.

3 composantes calibrées :
1. MSE continu       (85-90% du signal)
2. Focal 4 classes   (8% max)
3. CrossEntropy fine (5% max, masquée pour n<3)

Règle d'or : ratio_mse doit rester > 0.80 à tout moment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EXCLUDE_FROM_FINE_LOSS = {3, 4}  # PGY3 (n=2), PGY4 (n=1)


class HierarchicalSurgicalLoss(nn.Module):
    def __init__(self, alpha_focal=0.08, alpha_fine=0.05, gamma=2.0):
        super().__init__()
        self.alpha_focal = alpha_focal
        self.alpha_fine = alpha_fine
        self.gamma = gamma

    def focal_loss(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

    def fine_loss(self, logits_9, y9):
        """CrossEntropy sur 9 sous-niveaux. Exclut PGY3 et PGY4 (n insuffisant)."""
        exclude = torch.tensor(list(EXCLUDE_FROM_FINE_LOSS), device=y9.device)
        mask = ~torch.isin(y9, exclude)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits_9.device)
        return F.cross_entropy(logits_9[mask], y9[mask])

    def forward(self, score_agg, logits_4, logits_9, y_reg, y4, y9):
        mse = F.mse_loss(score_agg, y_reg)
        focal = self.focal_loss(logits_4, y4)
        fine = self.fine_loss(logits_9, y9)

        total = mse + self.alpha_focal * focal + self.alpha_fine * fine

        return total, {
            "mse": mse.item(),
            "focal": focal.item(),
            "fine": fine.item(),
            "ratio_mse": mse.item() / max(total.item(), 1e-8),
        }
