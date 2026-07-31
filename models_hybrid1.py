"""
models_hybrid1.py
=================
Réexporte l'architecture Hybrid1 canonique (src/models/hybrid1_evicems.py).

Point d'entrée pour train_hybrid1_perpair.py, eval_hybrid1_perpair.py, etc.
"""
from src.models.hybrid1_evicems import (
    Hybrid1Config,
    Hybrid1EVICEMS,
    Hybrid1ModelA,
    TransformerBlock,
    count_params,
    masked_mean,
)
from src.models.attention_pooling import AttentionPooling1D, masked_max

__all__ = [
    "Hybrid1Config",
    "Hybrid1EVICEMS",
    "Hybrid1ModelA",
    "TransformerBlock",
    "count_params",
    "masked_mean",
    "AttentionPooling1D",
    "masked_max",
]
