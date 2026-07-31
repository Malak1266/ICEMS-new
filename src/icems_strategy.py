"""
icems_strategy.py
=================
Composants partagés pour la stratégie ICEMS (une variable à la fois).

  - Augmentation globale équilibrée (DBA + jitter + time-warp [+ magnitude])
  - Architecture LSTM + Transformer hybride
  - Focal loss + poids de classe
  - Configurations des 4 approches expérimentales
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import CubicSpline, interp1d

CLASS4_NAMES = ["Student", "Junior", "Senior", "Expert"]

# Synthétiques par trial réel (hors original) — Partie 2 stratégie
AUGMENTATION_GLOBALE: Dict[str, Dict[str, int]] = {
    "Student": {"dba": 2, "jitter": 2, "timewarp": 1},
    "Junior": {"dba": 2, "jitter": 2, "timewarp": 1},
    "Senior": {"dba": 3, "jitter": 2, "timewarp": 2},
    "Expert": {"dba": 4, "jitter": 3, "timewarp": 2},
}

CLASS_WEIGHTS_INVERSE_FREQ = {
    "Student": 1.0,
    "Junior": 1.0,
    "Senior": 1.3,
    "Expert": 1.9,
}

CENTROID_DISTANCE_THRESHOLD = 5.0

TIME_WARP_RANGE = (0.8, 1.2)
MAGNITUDE_WARP_KNOTS = 4
MAGNITUDE_WARP_SIGMA = 0.2


@dataclass(frozen=True)
class ApproachConfig:
    """Une approche = une seule modification vs baseline r_participant=0.870."""

    id: int
    name: str
    model: str  # "causal_gru" | "hybrid"
    loss: str  # "mse" | "class_weights" | "focal" | "multitask"
    aug_mode: str  # "none" | "v2" | "global"
    description: str


APPROACHES: Dict[int, ApproachConfig] = {
    0: ApproachConfig(
        0, "diagnostic_tsne", "causal_gru", "mse", "none",
        "t-SNE baseline — aucun entraînement",
    ),
    1: ApproachConfig(
        1, "baseline_aug_global", "causal_gru", "mse", "global",
        "CausalGRU + MSE + augmentation globale équilibrée",
    ),
    2: ApproachConfig(
        2, "hybrid_mse", "hybrid", "mse", "global",
        "BiLSTM+Transformer + MSE + augmentation globale",
    ),
    3: ApproachConfig(
        3, "hybrid_focal", "hybrid", "focal", "global",
        "BiLSTM+Transformer + MSE+Focal + augmentation globale",
    ),
    4: ApproachConfig(
        4, "hybrid_multitask", "hybrid", "multitask", "global",
        "BiLSTM+Transformer + MSE+CE + augmentation globale",
    ),
}


def get_approach(approach_id: int) -> ApproachConfig:
    if approach_id not in APPROACHES:
        raise ValueError(f"Approche inconnue : {approach_id}. Valides : {sorted(APPROACHES)}")
    return APPROACHES[approach_id]


# ── Augmentation temporelle ──────────────────────────────────────────────────

def time_warp_features(
    feats: np.ndarray,
    rng: np.random.Generator,
    lo: float = TIME_WARP_RANGE[0],
    hi: float = TIME_WARP_RANGE[1],
) -> np.ndarray:
    """Déformation non-linéaire de l'axe temporel sur (C, T)."""
    c, t = feats.shape
    if t <= 1:
        return feats.copy()
    scale = float(rng.uniform(lo, hi))
    new_t = max(2, int(round(t * scale)))
    x_old = np.linspace(0.0, 1.0, t)
    x_new = np.linspace(0.0, 1.0, new_t)
    out = np.zeros((c, new_t), dtype=np.float64)
    for i in range(c):
        out[i, :] = interp1d(x_old, feats[i, :], kind="linear", fill_value="extrapolate")(x_new)
    return out


def magnitude_warp_features(
    feats: np.ndarray,
    rng: np.random.Generator,
    n_knots: int = MAGNITUDE_WARP_KNOTS,
    sigma: float = MAGNITUDE_WARP_SIGMA,
) -> np.ndarray:
    """Multiplication par une courbe lisse aléatoire (C, T)."""
    c, t = feats.shape
    if t <= 1:
        return feats.copy()
    knot_x = np.linspace(0.0, 1.0, n_knots)
    knot_y = 1.0 + rng.normal(0.0, sigma, size=n_knots)
    curve = CubicSpline(knot_x, knot_y)(np.linspace(0.0, 1.0, t))
    return feats * curve[np.newaxis, :]


def apply_jitter_features(
    feats: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
    epsilon: float = 1e-8,
) -> np.ndarray:
    out = feats.copy()
    for i in range(out.shape[0]):
        sigma = alpha * (np.std(out[i, :]) + epsilon)
        out[i, :] += rng.normal(0.0, sigma, size=out.shape[1])
    return out


# ── Embeddings trial-level (diagnostic t-SNE) ────────────────────────────────

def trial_kinematic_embedding(X: np.ndarray, valid_col: int = 9) -> np.ndarray:
    """
    Embedding baseline sans entraînement : mean + std des features cinématiques
    sur frames valides. Shape (2 * n_kin,).
    """
    valid = X[:, valid_col] > 0
    if not valid.any():
        valid = np.ones(X.shape[0], dtype=bool)
    kin = X[valid, :valid_col]
    return np.concatenate([kin.mean(axis=0), kin.std(axis=0)], axis=0).astype(np.float64)


def senior_expert_centroid_distance(
    embeddings: np.ndarray,
    y4_labels: np.ndarray,
    senior_y4: int = 2,
    expert_y4: int = 3,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Distance L2 entre centroïdes Senior et Expert (espace embedding standardisé)."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    emb_std = scaler.fit_transform(embeddings)
    senior = emb_std[y4_labels == senior_y4]
    expert = emb_std[y4_labels == expert_y4]
    if len(senior) == 0 or len(expert) == 0:
        return float("nan"), np.array([]), np.array([])
    c_senior = senior.mean(axis=0)
    c_expert = expert.mean(axis=0)
    dist = float(np.linalg.norm(c_senior - c_expert))
    return dist, c_senior, c_expert


def interpret_centroid_distance(dist: float, threshold: float = CENTROID_DISTANCE_THRESHOLD) -> str:
    if np.isnan(dist):
        return "indéterminé (classe manquante)"
    if dist < threshold:
        return "structurel — gestures Senior/Expert similaires, publier baseline"
    return "exploitable — signal séparable, poursuivre les approches 1–4"


# ── Loss ─────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss sur logits 4 classes (alpha, gamma=2)."""

    def __init__(self, alpha: float = 0.1, gamma: float = 2.0, n_classes: int = 4):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1.0 - pt) ** self.gamma * ce).mean()


def class_weight_tensor(device: torch.device) -> torch.Tensor:
    weights = [CLASS_WEIGHTS_INVERSE_FREQ[name] for name in CLASS4_NAMES]
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ── Architecture hybride ───────────────────────────────────────────────────────

class HybridLSTMTransformerScorer(nn.Module):
    """
    BiLSTM (local) + Transformer (global) + GRU causal (scorer).
    Compatible avec l'interface CausalGRUScorer (forward, trial_staff_logit).
    """

    def __init__(
        self,
        n_features: int,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_transformer_layers: int = 3,
        gru_hidden: int = 64,
        n_classes: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.proj = nn.Linear(n_features + 1, embed_dim)

        self.bilstm = nn.LSTM(
            embed_dim, embed_dim // 2, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.scorer_gru = nn.GRU(
            embed_dim, gru_hidden, num_layers=2,
            batch_first=True, dropout=dropout,
        )
        self.score_head = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        self.class_head = nn.Linear(gru_hidden, n_classes)
        self.staff_head = nn.Sequential(
            nn.Linear(gru_hidden, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def _encode(self, x: torch.Tensor, valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([x, valid_mask.unsqueeze(-1)], dim=-1)
        z = F.gelu(self.proj(inp))
        lstm_out, _ = self.bilstm(z)
        key_pad = valid_mask <= 0
        tr_out = self.transformer(z, src_key_padding_mask=key_pad)
        fused = self.fusion(torch.cat([lstm_out, tr_out], dim=-1))
        gru_out, _ = self.scorer_gru(fused)
        return gru_out, valid_mask

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        gru_out, mask = self._encode(x, valid_mask)
        return self.score_head(gru_out).squeeze(-1)

    def trial_class_logits(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        gru_out, mask = self._encode(x, valid_mask)
        w = mask.clamp(min=0)
        denom = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        trial_emb = (gru_out * w.unsqueeze(-1)).sum(dim=1) / denom
        return self.class_head(trial_emb)

    def trial_embedding(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        gru_out, mask = self._encode(x, valid_mask)
        w = mask.clamp(min=0)
        denom = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return (gru_out * w.unsqueeze(-1)).sum(dim=1) / denom

    def trial_staff_logit(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        emb = self.trial_embedding(x, valid_mask)
        return self.staff_head(emb).squeeze(-1)


def build_scorer(
    model_name: str,
    n_features: int,
    dropout: float = 0.15,
    seq_len: int = 800,
) -> nn.Module:
    """Factory : causal_gru (import lazy), hybrid, ou hybrid1_evicems.

    Parameters
    ----------
    model_name : str    "causal_gru" | "hybrid" | "hybrid1_evicems".
    n_features : int     Nombre de canaux d'entrée.
    dropout : float      Taux de dropout.
    seq_len : int        Longueur temporelle fixe (utilisée uniquement par
                         hybrid1_evicems, ignorée par les autres). Défaut 800.
    """
    if model_name == "hybrid":
        return HybridLSTMTransformerScorer(n_features=n_features, dropout=dropout)
    if model_name == "causal_gru":
        from step_B_classification import CausalGRUScorer
        return CausalGRUScorer(n_features=n_features, dropout=dropout)
    if model_name == "hybrid1_evicems":
        try:
            from models.hybrid1_evicems import Hybrid1EVICEMS
        except ImportError:
            from src.models.hybrid1_evicems import Hybrid1EVICEMS
        return Hybrid1EVICEMS(n_features=n_features, seq_len=seq_len, dropout=dropout)
    raise ValueError(f"Modèle inconnu : {model_name}")
