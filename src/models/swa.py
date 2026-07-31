"""
swa.py
======
Stochastic Weight Averaging (Izmailov et al. 2018) pour Hybrid1.

Règles ICEMS :
  - start_swa décidé sur le plateau du val EXTRÊMES uniquement (jamais le milieu)
  - condition rapportée distincte : "A2+SWA" ≠ "A2"
  - JAMAIS mélangée à l'ablation A0/A2/A6
  - Hybrid1EVICEMS n'a pas de BatchNorm → update_bn() inutile (no-op documenté)

Usage typique (dans train_hybrid1) :
    helper = SWAHelper(model, start_epoch=...)
    ...
    helper.update_if_ready(epoch)   # après optimizer.step()
    ...
    final_state = helper.finalize()  # state_dict moyenné
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


def model_has_batchnorm(model: nn.Module) -> bool:
    return any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
               for m in model.modules())


class SWAHelper:
    """
    Snapshot / moyenne des poids à partir de `start_epoch` (1-indexé, inclusif).

    Deux modes :
      - 'average' : AveragedModel (moyenne uniforme) — défaut SWA classique
      - 'ema'     : EMA optionnelle (multi_avg_fn) si besoin futur
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        start_epoch: int,
        device: torch.device | None = None,
        mode: str = "average",
    ) -> None:
        if start_epoch < 1:
            raise ValueError(f"start_epoch must be >= 1, got {start_epoch}")
        self.start_epoch = int(start_epoch)
        self.n_averaged = 0
        self.has_bn = model_has_batchnorm(model)
        self.device = device or next(model.parameters()).device

        if mode == "average":
            self.avg_model = AveragedModel(model, device=self.device)
        elif mode == "ema":
            self.avg_model = AveragedModel(
                model, device=self.device, multi_avg_fn=get_ema_multi_avg_fn(0.999),
            )
        else:
            raise ValueError(f"mode inconnu: {mode}")

    def ready(self, epoch: int) -> bool:
        return epoch >= self.start_epoch

    def update_if_ready(self, model: nn.Module, epoch: int) -> bool:
        """Appeler en fin d'époque (après early-stop check ou avant). Retourne True si snapshot."""
        if not self.ready(epoch):
            return False
        self.avg_model.update_parameters(model)
        self.n_averaged += 1
        return True

    def update_bn_if_needed(
        self,
        loader,
        device: torch.device,
    ) -> None:
        """
        Hybrid1 n'a pas de BN → no-op. Conservé pour compatibilité API /
        futurs modèles avec BN.
        """
        if not self.has_bn:
            return
        # torch.optim.swa_utils.update_bn attend un loader de tenseurs seuls ;
        # adapter si un jour Hybrid1 gagne du BN.
        from torch.optim.swa_utils import update_bn

        def _tensor_loader():
            for X, mask, _y in loader:
                yield X.to(device)

        update_bn(_tensor_loader(), self.avg_model, device=device)

    def state_dict(self) -> dict[str, Any]:
        """state_dict du modèle moyenné (clés sans préfixe 'module.')."""
        sd = self.avg_model.module.state_dict()
        return {k: v.detach().cpu().clone() for k, v in sd.items()}

    def finalize(self, model: nn.Module | None = None) -> dict[str, Any] | None:
        """
        Retourne le state_dict SWA si au moins 1 snapshot, sinon None
        (l'appelant garde alors best_state early-stop classique).
        """
        if self.n_averaged < 1:
            return None
        return self.state_dict()

    def meta(self) -> dict[str, Any]:
        return {
            "swa": True,
            "start_swa_epoch": self.start_epoch,
            "n_averaged": self.n_averaged,
            "has_batchnorm": self.has_bn,
            "update_bn_applied": False,  # Hybrid1 : pas de BN
        }


def resolve_start_swa_epoch(
    *,
    best_epoch: int,
    total_epochs_ran: int,
    swa_start: int | None,
    swa_last_k: int = 5,
) -> int:
    """
    Décide start_swa sur la trajectoire val EXTRÊMES.

    Priorité :
      1) swa_start explicite (CLI)
      2) sinon max(1, best_epoch) — démarre au plateau early-stop
      3) fallback : total_epochs_ran - K + 1 (K dernières époques)
    """
    if swa_start is not None:
        return max(1, int(swa_start))
    if best_epoch is not None and best_epoch > 0:
        return int(best_epoch)
    return max(1, int(total_epochs_ran) - int(swa_last_k) + 1)


def offline_average_checkpoints(
    state_dicts: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Moyenne uniforme de K state_dicts déjà collectés (utile hors boucle)."""
    if not state_dicts:
        raise ValueError("aucun checkpoint")
    keys = state_dicts[0].keys()
    out = {}
    for k in keys:
        stacked = torch.stack([sd[k].float() for sd in state_dicts], dim=0)
        out[k] = stacked.mean(dim=0).to(state_dicts[0][k].dtype)
    return out
