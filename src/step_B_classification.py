"""
step_B_classification.py
========================
Classification ordinale 9 sous-niveaux (y9) + agrégation 4 classes + LOPO.

Cibles d'entraînement : Y9_TO_REG (MS → Staff).
Loss : MSE + séparation y9 + ancrage Fellow/Staff + marge top-tier + tête Staff (BCE).
Features : 13 canaux (10 base + vel, jerk/vel, valid).

Usage (depuis ICEMS-main) :
    python src/step_B_classification.py --data data/continuous_per_trial.pkl --out results/sublevel_run_v2
    python src/step_B_classification.py --max_folds 3 --epochs 30 --mc-passes 10
    python src/step_B_classification.py --plot-only --out results/sublevel_run
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from tslearn.barycenters import dtw_barycenter_averaging

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# ── Configuration centralisée (mapping sublevel → score) ─────────────────────
# Import compatible exécution directe (python src/step_B_classification.py)
# ET import comme module (from src.step_B_classification import ...).
try:
    from config import (  # noqa: E402
        SUBLEVEL_TO_SCORE,
        SUBLEVEL_ORDER,
        Y9_TO_REG,
        sublevel_score,
        assert_all_sublevels_known,
        log_y_distribution,
    )
except ImportError:
    from src.config import (  # noqa: E402
        SUBLEVEL_TO_SCORE,
        SUBLEVEL_ORDER,
        Y9_TO_REG,
        sublevel_score,
        assert_all_sublevels_known,
        log_y_distribution,
    )

# ── Mapping 4 classes (aligné step_A / barème utilisateur) ───────────────────
Y9_TO_Y4 = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 3}
Y4_TO_REG = np.array([-1.0, -0.33, 0.33, 1.0], dtype=np.float32)
CLASS4_NAMES = ["Student", "Junior", "Senior", "Expert"]
CLASS4_COLORS = {0: "#d62728", 1: "#ff7f0e", 2: "#1f77b4", 3: "#2ca02c"}

# 9 sous-niveaux cliniques (aligné build_continuous_dataset / PROJECT_PROGRESSION §4.3)
SUBLEVEL_NAMES = [
    "Medical student", "PGY1", "PGY2", "PGY3", "PGY4",
    "PGY5", "PGY6", "Fellow", "Staff",
]
# Y9_TO_REG est importé depuis src/config.py (mapping non-linéaire SUBLEVEL_TO_SCORE).
# Ne pas redéfinir ici — toute modification doit se faire dans config.py uniquement.

SUBLEVEL_SHORT = ["MS", "PGY1", "PGY2", "PGY3", "PGY4", "PGY5", "PGY6", "Fellow", "Staff"]

N_BASE_FEATURES = 10
N_DERIVED_FEATURES = 3   # vel_mag, jerk/vel, valid_ratio (copie)
N_FEATURES = N_BASE_FEATURES + N_DERIVED_FEATURES  # = 13
VALID_COL = 9
MC_PASSES = 30
TRAIN_CROP_LEN = 800
INFER_MAX_LEN = 2500

TRAIN_DROPOUT = 0.15
MC_DROPOUT = 0.3
LOPO_PREDICTIONS_FILE = "lopo_predictions.pkl"

# Nested LOPO : validation interne tirée du pool train (jamais le participant tenu out).
N_INTERNAL_VAL_PARTICIPANTS = 4  # Option 1 — inner LOPO à K participants fixes
NESTED_LOPO_SEED = 42
CORRECTED_LOPO_RESULTS_ROOT = Path("results/lopo_corrected")
LOPO_V2_RESULTS_ROOT = Path("results/lopo_v2")

# Métriques baseline (run 9 juin — ablation_aug/comparison/_tmp_A)
BASELINE_METRICS = {
    "r_trial": 0.825,
    "r_participant": 0.870,
    "spearman_rho": 0.887,
    "mae": 0.352,
    "accuracy": 0.596,
    "expert_recall": 0.0,
    "mean_expert_score": 0.149,
    "mean_senior_score": 0.130,
    "senior_expert_diff": 0.019,
    "max_pred_score": 0.506,
}

# 9 sous-niveaux cliniques (TABLE I papier) — noms publication
PAPER_SUBLEVEL_BY_Y9 = [
    "Medical Student",
    "Resident PGY1",
    "Resident PGY2",
    "Resident PGY3",
    "Resident PGY4",
    "Resident PGY5",
    "Resident PGY6",
    "Fellow",
    "Neurosurgeon",
]

# ── DBA + jitter inline (par fold LOPO) ──────────────────────────────────────
DBA_N_PARENTS = 6
DBA_ALPHA = 0.03
MAX_DBA_FRAMES = 500
FEATURE_ROWS = list(range(2, 8))
LABEL_ROWS = (0, 1)
DBA_EPSILON = 1e-8


def trial_y4_reg(rec: dict) -> Tuple[int, float]:
    if "y4" in rec:
        y4 = int(rec["y4"])
        return y4, float(rec.get("y4_reg", Y4_TO_REG[y4]))
    y4 = Y9_TO_Y4[int(rec["y9"])]
    return y4, float(Y4_TO_REG[y4])


def trial_y9_reg(rec: dict) -> Tuple[int, float]:
    """Cible ordinale fine : score depuis SUBLEVEL_TO_SCORE (src/config.py).

    Priorité de résolution :
      1. Champ ``sublevel`` (ex : "Resident PGY1", "Fellow") via sublevel_score()
      2. Champ ``level``    (ex : "Medical student", "Staff") via sublevel_score()
      3. Fallback           : Y9_TO_REG[y9] (dérivé de SUBLEVEL_TO_SCORE)

    Note : la valeur pré-calculée ``y_reg`` stockée dans les fichiers .pkl n'est
    jamais utilisée, car elle peut provenir d'un ancien mapping. La cohérence avec
    le mapping courant est ainsi garantie quel que soit l'âge du fichier de données.
    """
    y9 = int(rec["y9"])
    raw_sl: str = rec.get("sublevel") or rec.get("level", "")
    if raw_sl:
        try:
            return y9, sublevel_score(str(raw_sl))
        except KeyError:
            pass
    return y9, float(Y9_TO_REG[y9])


def enrich_kinematic_features(X: np.ndarray) -> np.ndarray:
    """Modif 4 : descripteurs dérivés frame-wise (efficacité / fluidité)."""
    vel = np.linalg.norm(X[:, [0, 3, 6]], axis=1).astype(np.float32)
    jerk = np.linalg.norm(X[:, [2, 5, 8]], axis=1).astype(np.float32)
    valid = X[:, VALID_COL].astype(np.float32)
    derived = np.stack([vel, jerk / (vel + 1e-6), valid], axis=1)
    return np.concatenate([X[:, :N_BASE_FEATURES], derived], axis=1).astype(np.float32)


def frame_valid_mask(X: np.ndarray, thresh: float = 0.0) -> np.ndarray:
    """1 si frame présente (tracking valide), 0 sinon."""
    return (X[:, VALID_COL] > thresh).astype(np.float32)


def score_to_class(score: float) -> int:
    return int(np.argmin(np.abs(Y4_TO_REG - score)))


def score_to_y9(score: float) -> int:
    return int(np.argmin(np.abs(Y9_TO_REG - score)))


# Marge renforcée entre PGY6 / Fellow / Staff (frontière Senior–Expert).
TOP_TIER_Y9 = (6, 7, 8)
TOP_TIER_MARGIN = 0.12


class CausalGRUScorer(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, dropout: float = TRAIN_DROPOUT):
        super().__init__()
        self.proj = nn.Linear(n_features + 1, 64)
        self.gru = nn.GRU(64, 64, num_layers=2, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        # Modif 3 : tête binaire Staff vs non-Staff (sur embedding trial)
        self.staff_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def _gru_forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, valid_mask.unsqueeze(-1)], dim=-1)
        z = torch.nn.functional.gelu(self.proj(inp))
        out, _ = self.gru(z)
        return out

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self._gru_forward(x, valid_mask)).squeeze(-1)

    def trial_embedding(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        out = self._gru_forward(x, valid_mask)
        w = valid_mask.clamp(min=0)
        denom = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return (out * w.unsqueeze(-1)).sum(dim=1) / denom

    def trial_staff_logit(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self.staff_head(self.trial_embedding(x, valid_mask)).squeeze(-1)


def set_dropout_rate(model: nn.Module, p: float) -> None:
    if hasattr(model, "gru"):
        model.gru.dropout = p
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = p


def aggregate_score(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Moyenne pondérée par le masque de validité (B, T) → (B,)."""
    w = mask.clamp(min=0)
    denom = w.sum(dim=1).clamp(min=1e-6)
    return (scores * w).sum(dim=1) / denom


def separation_loss(
    agg_scores: torch.Tensor,
    rank_classes: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """Force score(i) > score(j) + margin quand rank(i) > rank(j)."""
    loss = torch.tensor(0.0, device=agg_scores.device)
    count = 0

    for i in range(len(agg_scores)):
        for j in range(len(agg_scores)):
            ci = int(rank_classes[i])
            cj = int(rank_classes[j])
            if ci > cj:
                violation = torch.clamp(
                    agg_scores[j] - agg_scores[i] + margin,
                    min=0.0,
                )
                loss = loss + violation
                count += 1

    if count > 0:
        loss = loss / count
    return loss


def top_tier_boundary_loss(
    agg_scores: torch.Tensor,
    y9_classes: torch.Tensor,
    margin: float = TOP_TIER_MARGIN,
) -> torch.Tensor:
    """
    Marge renforcée entre PGY6 (6), Fellow (7) et Staff (8).
    Cible la confusion Senior/Expert (Fellow+PGY6 vs Staff).
    """
    loss = torch.tensor(0.0, device=agg_scores.device)
    count = 0
    for i in range(len(agg_scores)):
        for j in range(len(agg_scores)):
            yi = int(y9_classes[i])
            yj = int(y9_classes[j])
            if yi in TOP_TIER_Y9 and yj in TOP_TIER_Y9 and yi > yj:
                violation = torch.clamp(
                    agg_scores[j] - agg_scores[i] + margin,
                    min=0.0,
                )
                loss = loss + violation
                count += 1
    if count > 0:
        loss = loss / count
    return loss


class ExpertAwareLoss(nn.Module):
    """
    MSE + pénalité asymétrique sur la compression des extrêmes.

    Composantes :
    1. MSE de base
    2. Pénalité de sous-estimation (quand y_pred < y_true pour les hauts scores)
    3. Pénalité de sur-estimation (quand y_pred > y_true pour les bas scores)
    4. Terme de séparation : pousse Expert loin de Senior
    """

    def __init__(self, lambda_asym: float = 0.03, lambda_sep: float = 0.05, margin: float = 0.25):
        super().__init__()
        self.lambda_asym = lambda_asym
        self.lambda_sep = lambda_sep
        self.margin = margin

    def forward(self, y_pred, y_true):
        mse = F.mse_loss(y_pred, y_true)

        high_mask = (y_true >= 0.33).float()
        low_mask = (y_true <= -0.33).float()

        under_penalty = high_mask * F.relu(y_true - y_pred)
        over_penalty = low_mask * F.relu(y_pred - y_true)
        asym_loss = (under_penalty + over_penalty).mean()

        expert_mask = (y_true >= 0.9).float()
        senior_mask = ((y_true >= 0.2) & (y_true <= 0.5)).float()

        sep_loss = torch.tensor(0.0, device=y_pred.device)
        if expert_mask.sum() > 0 and senior_mask.sum() > 0:
            expert_scores = (y_pred * expert_mask).sum() / expert_mask.sum()
            senior_scores = (y_pred * senior_mask).sum() / senior_mask.sum()
            sep_loss = F.relu(self.margin - (expert_scores - senior_scores))

        total = mse + self.lambda_asym * asym_loss + self.lambda_sep * sep_loss
        return total, {
            "mse": mse.item(),
            "asym": asym_loss.item(),
            "sep": sep_loss.item(),
        }


def anchor_loss(
    agg_scores: torch.Tensor,
    y9_classes: torch.Tensor,
) -> torch.Tensor:
    """
    Ancrage des extrêmes et du haut du spectre :
      MS (y9=0)     → score < -0.8
      Fellow (y9=7) → score > +0.75
      Staff  (y9=8) → score > +0.90
    """
    n = len(agg_scores)
    if n == 0:
        return torch.tensor(0.0, device=agg_scores.device)

    loss = torch.tensor(0.0, device=agg_scores.device)
    for i, score in enumerate(agg_scores):
        y9 = int(y9_classes[i])
        if y9 == 0:
            loss = loss + nn.functional.relu(score + 0.8)
        elif y9 == 7:
            loss = loss + nn.functional.relu(0.75 - score)
        elif y9 == 8:
            loss = loss + nn.functional.relu(0.90 - score)
    return loss / n


def ordinal_loss(
    agg: torch.Tensor,
    y_score: torch.Tensor,
    y9_classes: torch.Tensor,
    sep_weight: float = 0.5,
    margin: float = 0.3,
    anchor_weight: float = 0.0,
    top_tier_weight: float = 0.0,
) -> torch.Tensor:
    """MSE sur cibles y9 + séparation ordinale + ancrage Fellow/Staff."""
    mse = nn.functional.mse_loss(agg, y_score)
    sep = separation_loss(agg, y9_classes, margin=margin)
    anchor = anchor_loss(agg, y9_classes)
    top = top_tier_boundary_loss(agg, y9_classes)
    return (
        mse
        + sep_weight * sep
        + anchor_weight * anchor
        + top_tier_weight * top
    )


def staff_binary_loss(
    staff_logit: torch.Tensor,
    y9_classes: torch.Tensor,
    pos_weight: float = 5.0,
) -> torch.Tensor:
    """Modif 3 : BCE Staff (y9=8) vs reste — compense la rareté relative."""
    target = (y9_classes == 8).float()
    weight = torch.tensor(pos_weight, device=staff_logit.device)
    return nn.functional.binary_cross_entropy_with_logits(
        staff_logit, target, pos_weight=weight,
    )


def combined_loss(
    model: CausalGRUScorer,
    xs: torch.Tensor,
    masks: torch.Tensor,
    y_score: torch.Tensor,
    y9_classes: torch.Tensor,
    sep_weight: float,
    margin: float,
    anchor_weight: float,
    top_tier_weight: float,
    staff_weight: float,
    staff_pos_weight: float,
) -> torch.Tensor:
    scores = model(xs, masks)
    agg = aggregate_score(scores, masks)
    loss = ordinal_loss(
        agg, y_score, y9_classes,
        sep_weight=sep_weight, margin=margin,
        anchor_weight=anchor_weight, top_tier_weight=top_tier_weight,
    )
    if staff_weight > 0:
        staff_log = model.trial_staff_logit(xs, masks)
        loss = loss + staff_weight * staff_binary_loss(
            staff_log, y9_classes, pos_weight=staff_pos_weight,
        )
    return loss


def adjust_score_with_staff(score: float, staff_prob: float) -> float:
    """Fusion score régression + probabilité Staff."""
    if staff_prob <= 0.5:
        return score
    return float(max(score, staff_prob * 0.95))


class TrialDataset(Dataset):
    def __init__(self, items: List[Tuple[np.ndarray, float, int, int]], crop_len: int = TRAIN_CROP_LEN):
        self.items = items
        self.crop_len = crop_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        x, y_reg, y4, y9 = self.items[idx]
        if x.shape[0] > self.crop_len:
            start = np.random.randint(0, x.shape[0] - self.crop_len)
            x = x[start : start + self.crop_len]
        vm = frame_valid_mask(x)
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(vm), float(y_reg), y4, y9


def collate_trials(batch):
    lengths = [b[0].shape[0] for b in batch]
    T_max = max(lengths)
    B = len(batch)
    xs = torch.zeros(B, T_max, N_FEATURES)
    masks = torch.zeros(B, T_max)
    y = torch.zeros(B)
    y4 = torch.zeros(B, dtype=torch.long)
    y9 = torch.zeros(B, dtype=torch.long)
    for i, (x, vm, yr, y4_i, y9_i) in enumerate(batch):
        L = x.shape[0]
        xs[i, :L] = x
        masks[i, :L] = vm
        y[i] = yr
        y4[i] = y4_i
        y9[i] = y9_i
    return xs, masks, y, y4, y9


def compute_norm_stats(trials: Dict) -> Tuple[np.ndarray, np.ndarray]:
    kin = np.concatenate(
        [enrich_kinematic_features(rec["X"]).reshape(-1, N_FEATURES) for rec in trials.values()],
        axis=0,
    )
    mean = kin.mean(axis=0)
    std = kin.std(axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = enrich_kinematic_features(X)
    out = (out - mean) / std
    return out.astype(np.float32)


def build_stratified_items(items: List, shuffle_classes: bool = True) -> List:
    """Intercale les trials par sous-niveau y9 pour des batches équilibrés."""
    by_y9 = {c: [] for c in range(9)}
    for t in items:
        by_y9[t[3]].append(t)

    if shuffle_classes:
        for c in range(9):
            np.random.shuffle(by_y9[c])

    stratified = []
    max_len = max(len(v) for v in by_y9.values()) if by_y9 else 0
    for i in range(max_len):
        for c in range(9):
            if i < len(by_y9[c]):
                stratified.append(by_y9[c][i])
    return stratified


def trials_to_items(trials: Dict, mean: np.ndarray, std: np.ndarray) -> List:
    items = []
    for rec in trials.values():
        if rec["X"].shape[0] < 2:
            continue
        x = apply_norm(rec["X"], mean, std)
        y9, y_reg = trial_y9_reg(rec)
        y4 = int(rec.get("y4", Y9_TO_Y4[y9]))
        items.append((x, y_reg, y4, y9))
    return items


def _maybe_truncate(x: np.ndarray, max_len: int = INFER_MAX_LEN) -> np.ndarray:
    if x.shape[0] <= max_len:
        return x
    idx = np.linspace(0, x.shape[0] - 1, max_len, dtype=int)
    return x[idx]


@torch.no_grad()
def predict_trial_mc(
    model: CausalGRUScorer,
    x: np.ndarray,
    device: torch.device,
    n_passes: int = MC_PASSES,
    agg_percentile: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    x = _maybe_truncate(x)
    set_dropout_rate(model, MC_DROPOUT)
    model.train()
    xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
    vm = torch.from_numpy(frame_valid_mask(x)).unsqueeze(0).to(device)

    scores_list = []
    staff_probs = []
    for _ in range(n_passes):
        s = model(xt, vm).squeeze(0).cpu().numpy()
        scores_list.append(s)
        slog = model.trial_staff_logit(xt, vm)
        staff_probs.append(torch.sigmoid(slog).item())
    scores_arr = np.stack(scores_list, axis=0)
    mean_t = scores_arr.mean(axis=0)
    std_t = scores_arr.std(axis=0)
    staff_prob = float(np.mean(staff_probs))

    valid = frame_valid_mask(x) > 0
    frame_scores = mean_t[valid] if valid.any() else mean_t
    if agg_percentile is not None and len(frame_scores) > 0:
        trial_score = float(np.percentile(frame_scores, agg_percentile))
    else:
        trial_score = float(np.median(frame_scores))
    trial_score = adjust_score_with_staff(trial_score, staff_prob)
    trial_std = float(np.median(std_t[valid])) if valid.any() else float(np.median(std_t))
    return mean_t, std_t, trial_score, trial_std, staff_prob


def train_fold(
    train_trials: Dict,
    val_trials: Dict,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    batch_size: int,
    sep_weight: float = 0.5,
    margin: float = 0.3,
    anchor_weight: float = 0.0,
    top_tier_weight: float = 0.0,
    staff_weight: float = 1.0,
    staff_pos_weight: float = 5.0,
    model_name: str = "causal_gru",
    loss_mode: str = "mse",
    focal_alpha: float = 0.1,
    focal_gamma: float = 2.0,
) -> Tuple[nn.Module, dict]:
    from icems_strategy import FocalLoss, build_scorer, class_weight_tensor

    mean, std = compute_norm_stats(train_trials)
    train_items = trials_to_items(train_trials, mean, std)
    val_items = trials_to_items(val_trials, mean, std)

    val_loader = DataLoader(
        TrialDataset(val_items),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_trials,
    )

    model = build_scorer(model_name, n_features=N_FEATURES, dropout=TRAIN_DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_criterion = nn.MSELoss(reduction="none")
    y4_reg_tensor = torch.tensor(Y4_TO_REG, dtype=torch.float32, device=device)
    class_weights = class_weight_tensor(device)
    focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma).to(device)
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_val, best_state, wait = float("inf"), None, 0
    best_epoch, stopped_epoch = 0, epochs
    epoch_loss_log: List[dict] = []

    def _batch_loss(
        xs: torch.Tensor,
        masks: torch.Tensor,
        y4_classes: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        y4_reg = y4_reg_tensor[y4_classes]
        scores = model(xs, masks)
        agg = aggregate_score(scores, masks)
        if loss_mode == "class_weights":
            w = class_weights[y4_classes]
            mse = (mse_criterion(agg, y4_reg) * w).mean()
            return mse, {"mse": mse.item(), "focal": 0.0, "ce": 0.0}
        if loss_mode == "focal" and hasattr(model, "trial_class_logits"):
            mse = mse_criterion(agg, y4_reg).mean()
            logits = model.trial_class_logits(xs, masks)
            fl = focal(logits, y4_classes)
            total = mse + fl
            return total, {"mse": mse.item(), "focal": fl.item(), "ce": 0.0}
        if loss_mode == "multitask" and hasattr(model, "trial_class_logits"):
            mse = mse_criterion(agg, y4_reg).mean()
            logits = model.trial_class_logits(xs, masks)
            ce = ce_criterion(logits, y4_classes)
            total = mse + ce
            return total, {"mse": mse.item(), "focal": 0.0, "ce": ce.item()}
        mse = mse_criterion(agg, y4_reg).mean()
        return mse, {"mse": mse.item(), "focal": 0.0, "ce": 0.0}

    for ep in range(1, epochs + 1):
        model.train()
        set_dropout_rate(model, TRAIN_DROPOUT)
        stratified_items = build_stratified_items(train_items)
        loader = DataLoader(
            TrialDataset(stratified_items),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_trials,
        )
        train_comps = {"mse": [], "focal": [], "ce": []}
        for xs, masks, y, y4_classes, y9_classes in loader:
            xs = xs.to(device)
            masks = masks.to(device)
            y4_classes = y4_classes.to(device)
            loss, comps = _batch_loss(xs, masks, y4_classes)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k in train_comps:
                train_comps[k].append(comps[k])

        model.eval()
        val_losses = []
        val_comps = {"mse": [], "focal": [], "ce": []}
        with torch.no_grad():
            for xs, masks, y, y4_val, y9_val in val_loader:
                xs, masks = xs.to(device), masks.to(device)
                y4_val = y4_val.to(device)
                vloss, vcomps = _batch_loss(xs, masks, y4_val)
                val_losses.append(vloss.item())
                for k in val_comps:
                    val_comps[k].append(vcomps[k])
        vloss = float(np.mean(val_losses)) if val_losses else float("inf")
        ep_log = {
            "epoch": ep,
            "train_mse": float(np.mean(train_comps["mse"])) if train_comps["mse"] else None,
            "train_focal": float(np.mean(train_comps["focal"])) if train_comps["focal"] else None,
            "train_ce": float(np.mean(train_comps["ce"])) if train_comps["ce"] else None,
            "val_mse": float(np.mean(val_comps["mse"])) if val_comps["mse"] else None,
            "val_focal": float(np.mean(val_comps["focal"])) if val_comps["focal"] else None,
            "val_ce": float(np.mean(val_comps["ce"])) if val_comps["ce"] else None,
            "val_loss": vloss,
        }
        epoch_loss_log.append(ep_log)

        if vloss < best_val - 1e-5:
            best_val = vloss
            best_epoch = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                stopped_epoch = ep
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model._norm_mean = mean
    model._norm_std = std
    train_info = {
        "stopped_epoch": stopped_epoch,
        "best_epoch": best_epoch,
        "early_stopped": stopped_epoch < epochs,
        "best_val_loss": best_val,
        "loss_epochs": epoch_loss_log,
        "model_name": model_name,
        "loss_mode": loss_mode,
    }
    if epoch_loss_log:
        best_log = epoch_loss_log[best_epoch - 1]
        train_info["loss_components"] = {
            "mse": best_log.get("val_mse"),
            "focal": best_log.get("val_focal"),
            "ce": best_log.get("val_ce"),
        }
    return model, train_info


def participants_of(dataset: Dict) -> List[str]:
    return sorted({k[0] for k in dataset.keys() if not str(k[0]).startswith("synth_")})


def _participant_y4_map(dataset: Dict, pids: List[str]) -> Dict[str, int]:
    """y4 par participant (premier trial disponible)."""
    out: Dict[str, int] = {}
    for pid in pids:
        pid = str(pid)
        keys = [k for k in dataset if str(k[0]) == pid]
        if not keys:
            continue
        rec = dataset[keys[0]]
        y9 = int(rec["y9"])
        out[pid] = int(rec.get("y4", Y9_TO_Y4[y9]))
    return out


def select_stratified_val_participants(
    train_pool_pids: List[str],
    dataset: Dict,
    k: int = N_INTERNAL_VAL_PARTICIPANTS,
    seed: int = NESTED_LOPO_SEED,
    fold_idx: int = 0,
) -> List[str]:
    """Option 1 — inner LOPO à K participants fixes, stratifiés par classe y4.

    Au moins 1 participant par grand groupe (Student/Junior/Senior/Expert) quand
    possible ; Expert limité à 1 participant en validation interne.
    """
    # Option 2 (plus coûteuse) — inner k-fold à 3 folds sur le pool train :
    #   from sklearn.model_selection import GroupKFold
    #   gkf = GroupKFold(n_splits=3)
    #   ... entraîner 3 modèles par fold externe, early stop sur val interne moyenne
    pool = sorted({str(p) for p in train_pool_pids})
    if len(pool) < k + 1:
        raise ValueError(
            f"Fold {fold_idx}: nested LOPO requiert au moins {k + 1} participants "
            f"dans le pool train (hors held-out), seulement {len(pool)} disponibles."
        )

    pid_y4 = _participant_y4_map(dataset, pool)
    by_class: Dict[int, List[str]] = defaultdict(list)
    for p in pool:
        by_class[pid_y4[p]].append(p)

    rng = np.random.default_rng(seed + fold_idx)
    chosen: List[str] = []

    for y4 in range(4):
        if len(chosen) >= k:
            break
        if y4 == 3 and any(pid_y4.get(c) == 3 for c in chosen):
            continue
        candidates = [p for p in by_class[y4] if p not in chosen]
        if not candidates:
            continue
        chosen.append(str(rng.choice(candidates)))

    remaining = [p for p in pool if p not in chosen]
    n_needed = k - len(chosen)
    if n_needed > 0 and remaining:
        extra = rng.choice(remaining, size=min(n_needed, len(remaining)), replace=False)
        chosen.extend(str(p) for p in np.atleast_1d(extra))

    return sorted(chosen)


def select_internal_val_participants(
    train_pool_pids: List[str],
    fold_idx: int,
    n_val: int = N_INTERNAL_VAL_PARTICIPANTS,
    seed: int = NESTED_LOPO_SEED,
    dataset: Optional[Dict] = None,
) -> List[str]:
    """Tire n_val participants du pool d'entraînement (stratifié si dataset fourni)."""
    if dataset is not None:
        return select_stratified_val_participants(
            train_pool_pids, dataset, k=n_val, seed=seed, fold_idx=fold_idx,
        )
    pool = sorted({str(p) for p in train_pool_pids})
    if len(pool) < n_val + 1:
        raise ValueError(
            f"Fold {fold_idx}: nested LOPO requiert au moins {n_val + 1} participants "
            f"dans le pool train (hors held-out), seulement {len(pool)} disponibles."
        )
    rng = np.random.default_rng(seed + fold_idx)
    chosen = rng.choice(pool, size=n_val, replace=False)
    return sorted(str(p) for p in chosen)


def _real_participants_of_trials(trials: Dict) -> Set[str]:
    return {
        str(k[0]) for k, v in trials.items()
        if not v.get("is_augmented", False) and not str(k[0]).startswith("synth_")
    }


def _aug_source_participants(rec: dict) -> Set[str]:
    return {str(p) for p in rec.get("aug_source_participants", [])}


def filter_synth_by_excluded_sources(synth_dict: Dict, exclude_pids: Set[str]) -> Dict:
    """Exclut les synthétiques dont au moins un parent ∈ exclude_pids."""
    exclude = {str(p) for p in exclude_pids}
    out = {}
    for k, v in synth_dict.items():
        sources = _aug_source_participants(v)
        if sources & exclude:
            continue
        out[k] = v
    return out


def build_train_trials(
    train_fit_real: Dict,
    synth_dict: Dict,
    exclude_pids: Set[str],
) -> Dict:
    """TRAIN = trials réels hors val/test + synthétiques sans parent val/test."""
    synth_filtered = filter_synth_by_excluded_sources(synth_dict, exclude_pids)
    return {**train_fit_real, **synth_filtered}


def assert_no_leakage(
    train_trials: Dict,
    val_trials: Dict,
    test_trials: Dict,
    p_test: str,
    val_participants: List[str],
) -> None:
    """Assertions anti-fuite obligatoires (disjonction participant + augmentation)."""
    val_set = {str(p) for p in val_participants}
    test_set = {str(p_test)}
    exclude = val_set | test_set

    train_p = _real_participants_of_trials(train_trials)
    val_p = _real_participants_of_trials(val_trials)
    test_p = _real_participants_of_trials(test_trials)

    assert test_p == test_set, f"FUITE: test participants {test_p} != {{{p_test}}}"
    assert val_p == val_set, f"FUITE: val participants {val_p} != {val_set}"
    assert train_p.isdisjoint(val_p), "FUITE: train/val partagent un participant"
    assert train_p.isdisjoint(test_p), "FUITE: train/test partagent un participant"
    assert val_p.isdisjoint(test_p), "FUITE: val/test partagent un participant"

    for rec in val_trials.values():
        assert not rec.get("is_augmented", False), "FUITE: augmenté dans val"
    for rec in test_trials.values():
        assert not rec.get("is_augmented", False), "FUITE: augmenté dans test"

    aug_train_sources: Set[str] = set()
    for k, rec in train_trials.items():
        if not rec.get("is_augmented", False):
            continue
        sources = _aug_source_participants(rec)
        assert sources, f"FUITE: synthétique {k} sans aug_source_participants"
        assert not (sources & exclude), (
            f"FUITE: augmentation issue de val/test présente dans train ({k}, sources={sources})"
        )
        aug_train_sources |= sources

    assert aug_train_sources.isdisjoint(exclude), (
        "FUITE: augmentation issue de val/test présente dans train"
    )


def print_fold_split_debug(
    fold_idx: int,
    train_trials: Dict,
    val_trials: Dict,
    test_trials: Dict,
    p_test: str,
    val_participants: List[str],
) -> None:
    """Smoke test : tailles des splits et absence d'augmentation en val/test."""
    train_p = _real_participants_of_trials(train_trials)
    val_p = _real_participants_of_trials(val_trials)
    test_p = _real_participants_of_trials(test_trials)
    n_aug_train = sum(1 for v in train_trials.values() if v.get("is_augmented", False))
    print(
        f"  [split debug fold {fold_idx}] "
        f"train_p={len(train_p)} val_p={len(val_p)} test_p={len(test_p)} | "
        f"trials train={len(train_trials)} (aug={n_aug_train}) "
        f"val={len(val_trials)} test={len(test_trials)}"
    )
    print(
        f"    val participants   : {sorted(val_participants)}"
    )
    print(
        f"    is_augmented val/test: "
        f"{any(v.get('is_augmented') for v in val_trials.values())}/"
        f"{any(v.get('is_augmented') for v in test_trials.values())} (attendu 0/0)"
    )
    aug_sources = set()
    for rec in train_trials.values():
        if rec.get("is_augmented", False):
            aug_sources |= _aug_source_participants(rec)
    overlap = aug_sources & (val_p | test_p)
    print(f"    aug sources ∩ (val∪test) : {sorted(overlap) or '∅ (OK)'}")


def assert_disjoint_participant_splits(
    external_test_pid: str,
    internal_val_pids: List[str],
    train_fit_pids: List[str],
) -> None:
    """Lève si un participant apparaît dans plusieurs splits du fold."""
    splits = {
        "external_test": {str(external_test_pid)},
        "internal_val": {str(p) for p in internal_val_pids},
        "train_fit": {str(p) for p in train_fit_pids},
    }
    for name_a, set_a in splits.items():
        for name_b, set_b in splits.items():
            if name_a >= name_b:
                continue
            overlap = set_a & set_b
            if overlap:
                raise ValueError(
                    f"Participant leak: {sorted(overlap)} présents à la fois dans "
                    f"'{name_a}' et '{name_b}'"
                )


def _data_to_array(data) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"data doit être 2D (C, T), reçu shape={arr.shape}")
    return arr


def _entry_data_to_X(data) -> np.ndarray:
    """Convertit data (C, T) vers X (T, 10) pour le scorer."""
    arr = np.asarray(data, dtype=np.float32)
    t = arr.shape[1]
    x = np.zeros((t, N_FEATURES), dtype=np.float32)
    x[:, 0] = arr[3, :]
    x[:, 1] = arr[4, :]
    x[:, 2] = arr[5, :]
    x[:, 3] = arr[7, :]
    x[:, 6] = arr[6, :]
    x[:, 9] = 1.0
    return x


def _rec_to_entry(key: Tuple, rec: dict) -> dict:
    participant, trial = key
    x = np.asarray(rec["X"], dtype=np.float64)
    y9 = int(rec["y9"])
    y4 = int(rec.get("y4", Y9_TO_Y4[y9]))
    t = x.shape[0]
    label_exp = np.full(t, y4, dtype=np.float64)
    label_lvl = np.full(t, y9, dtype=np.float64)
    pos_mag = np.linalg.norm(x[:, [0, 3, 6]], axis=1)
    data = np.vstack([label_exp, label_lvl, pos_mag, x[:, 0], x[:, 1], x[:, 2], x[:, 6], x[:, 3]])
    return {
        "name": f"{participant}_{trial}",
        "participant": str(participant),
        "trial": str(trial),
        "expertise": CLASS4_NAMES[y4],
        "expertise_idx": y4,
        "level_idx": y9,
        "level": str(rec.get("level", "")),
        "is_augmented": False,
        "data": data.tolist(),
    }


def _entry_to_train_rec(entry: dict) -> dict:
    y4 = int(entry["expertise_idx"])
    y9 = int(entry.get("level_idx", 0))
    x = _entry_data_to_X(entry["data"])
    return {
        "X": x,
        "y9": y9,
        "y4": y4,
        "y4_reg": float(Y4_TO_REG[y4]),
        "y_reg": float(Y9_TO_REG[y9]),
        "level": entry.get("level", SUBLEVEL_NAMES[y9]),
        "T": x.shape[0],
        "fs": float(entry.get("fs", 10.0)),
        "is_augmented": bool(entry.get("is_augmented", False)),
        "aug_source_participants": [
            str(p) for p in entry.get("aug_source_participants", [])
        ],
    }


def _extract_features(data) -> np.ndarray:
    return _data_to_array(data)[FEATURE_ROWS, :]


def _resample_features(feats: np.ndarray, target_len: int) -> np.ndarray:
    c, t = feats.shape
    if t == target_len:
        return feats
    if t <= 1:
        return np.tile(feats, (1, target_len))[:, :target_len]
    x_old = np.linspace(0.0, 1.0, t)
    x_new = np.linspace(0.0, 1.0, target_len)
    out = np.zeros((c, target_len), dtype=np.float64)
    for i in range(c):
        out[i, :] = interp1d(x_old, feats[i, :], kind="linear", fill_value="extrapolate")(x_new)
    return out


def _pad_feature_group(seqs: List[np.ndarray]) -> Tuple[np.ndarray, int]:
    t_target = min(min(s.shape[1] for s in seqs), MAX_DBA_FRAMES)
    aligned = [_resample_features(s, t_target) for s in seqs]
    padded = np.zeros((len(aligned), t_target, aligned[0].shape[0]), dtype=np.float64)
    for i, seq in enumerate(aligned):
        padded[i, :, :] = seq.T
    return padded, t_target


def _run_dba_on_group(parent_features: List[np.ndarray], max_iter: int = 30) -> np.ndarray:
    group_array, _ = _pad_feature_group(parent_features)
    bary = dtw_barycenter_averaging(group_array, max_iter=max_iter)
    return bary.T


def _rebuild_data_matrix(parent_data: np.ndarray, feature_block: np.ndarray) -> np.ndarray:
    t_bary = feature_block.shape[1]
    out = np.zeros((parent_data.shape[0], t_bary), dtype=np.float64)
    for row in LABEL_ROWS:
        out[row, :] = parent_data[row, 0] if parent_data.shape[1] > 0 else 0.0
    out[FEATURE_ROWS, :] = feature_block
    return out


def _apply_jitter(data: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    out = data.copy()
    for row in FEATURE_ROWS:
        channel = out[row, :]
        sigma = alpha * (np.std(channel) + DBA_EPSILON)
        out[row, :] = channel + rng.normal(0.0, sigma, size=channel.shape)
    out[LABEL_ROWS, :] = data[LABEL_ROWS, :]
    return out


def _synth_entry_from_data(
    template: dict,
    data_matrix: np.ndarray,
    aug_type: str,
    fold_i: int,
    synth_i: int,
    class_tag: str,
    aug_sources: Optional[List[str]] = None,
) -> Tuple[dict, int]:
    """Crée un trial synthétique avec participant source pour filtre anti-fuite."""
    pid = f"synth_{fold_i:03d}_{synth_i:03d}"
    synth_i += 1
    trial = f"{aug_type}_{class_tag.lower()}_{synth_i}"
    y9 = int(template["level_idx"])
    y4 = int(template["expertise_idx"])
    sources = aug_sources or [str(template["participant"])]
    entry = {
        "name": f"{pid}_{trial}",
        "participant": pid,
        "trial": trial,
        "expertise": CLASS4_NAMES[y4],
        "expertise_idx": y4,
        "level_idx": y9,
        "level": template.get("level", SUBLEVEL_NAMES[y9]),
        "is_augmented": True,
        "aug_type": aug_type,
        "aug_source_participants": sorted(set(sources)),
        "data": data_matrix.tolist(),
    }
    return {(pid, trial): _entry_to_train_rec(entry)}, synth_i


def _generate_synth_entry(
    pool: List[dict],
    aug_type: str,
    fold_i: int,
    synth_i: int,
    rng: np.random.Generator,
    n_parents: int = DBA_N_PARENTS,
    alpha: float = DBA_ALPHA,
    class_tag: str = "mixed",
) -> Tuple[Dict, int]:
    """Génère un trial synthétique (dba | jitter | timewarp | magnitude)."""
    from icems_strategy import (
        apply_jitter_features,
        magnitude_warp_features,
        time_warp_features,
    )

    if not pool:
        return {}, synth_i

    if aug_type == "dba":
        n_parents_eff = min(n_parents, len(pool))
        if len(pool) < 2:
            return {}, synth_i
        parents = list(rng.choice(pool, size=n_parents_eff, replace=False))
        parent_arrays = [_data_to_array(p["data"]) for p in parents]
        parent_feats = [_extract_features(p["data"]) for p in parents]
        bary_feats = _run_dba_on_group(parent_feats)
        full = _rebuild_data_matrix(parent_arrays[0], bary_feats)
        sources = [str(p["participant"]) for p in parents]
        return _synth_entry_from_data(
            parents[0], full, "dba", fold_i, synth_i, class_tag, sources,
        )

    parent = pool[int(rng.integers(0, len(pool)))]
    parent_data = _data_to_array(parent["data"])
    feats = _extract_features(parent["data"])

    if aug_type == "jitter":
        warped = apply_jitter_features(feats, alpha, rng)
    elif aug_type == "timewarp":
        warped = time_warp_features(feats, rng)
    elif aug_type == "magnitude":
        warped = magnitude_warp_features(feats, rng)
    else:
        raise ValueError(f"aug_type inconnu : {aug_type}")

    full = _rebuild_data_matrix(parent_data, warped)
    return _synth_entry_from_data(
        parent, full, aug_type, fold_i, synth_i, class_tag,
        [str(parent["participant"])],
    )


def _count_trials_by_y4(trials: Dict) -> Dict[int, int]:
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for rec in trials.values():
        y4, _ = trial_y4_reg(rec)
        counts[y4] += 1
    return counts


def _dba_jitter_augment(
    trials_dict: Dict,
    ratio: float,
    fold_i: int,
    seed: int = 42,
    n_parents: int = DBA_N_PARENTS,
    alpha: float = DBA_ALPHA,
    synth_start_i: int = 0,
    class_tag: str = "mixed",
) -> Tuple[Dict, int]:
    """Génère ``ratio × n_real`` trials synthétiques DBA+jitter depuis trials_dict."""
    if not trials_dict or ratio <= 0:
        return {}, synth_start_i

    rng = np.random.default_rng(seed + fold_i * 1000 + hash(class_tag) % 9973)
    entries = [_rec_to_entry(k, v) for k, v in trials_dict.items()]
    n_real = len(entries)
    n_needed = max(0, int(round(ratio * n_real)))
    if n_needed <= 0:
        return {}, synth_start_i

    pool = entries
    n_parents_eff = min(n_parents, len(pool))
    if len(pool) < 2:
        print(
            f"  [aug] {class_tag}: pool={len(pool)} < 2, "
            "augmentation ignorée pour ce groupe"
        )
        return {}, synth_start_i

    synth_dict: Dict = {}
    synth_i = synth_start_i

    for _ in range(n_needed):
        parents = list(rng.choice(pool, size=n_parents_eff, replace=False))
        parent_arrays = [_data_to_array(p["data"]) for p in parents]
        parent_feats = [_extract_features(p["data"]) for p in parents]
        bary_feats = _run_dba_on_group(parent_feats)
        full = _rebuild_data_matrix(parent_arrays[0], bary_feats)
        jittered = _apply_jitter(full, alpha, rng)

        template = parents[0]
        pid = f"synth_{fold_i:03d}_{synth_i:03d}"
        synth_i += 1
        trial = f"dba_jitter_{class_tag.lower().replace(' ', '_')}_{synth_i}"
        y9 = int(template["level_idx"])
        y4 = int(template["expertise_idx"])
        aug_sources = sorted({str(p["participant"]) for p in parents})
        entry = {
            "name": f"{pid}_{trial}",
            "participant": pid,
            "trial": trial,
            "expertise": CLASS4_NAMES[y4],
            "expertise_idx": y4,
            "level_idx": y9,
            "level": template.get("level", SUBLEVEL_NAMES[y9]),
            "is_augmented": True,
            "aug_type": "dba+jitter",
            "aug_source_participants": aug_sources,
            "data": jittered.tolist(),
        }
        key = (pid, trial)
        synth_dict[key] = _entry_to_train_rec(entry)

    return synth_dict, synth_i


def _augment_train_fold_inline(
    train_real: Dict,
    fold_i: int,
    n_parents: int = DBA_N_PARENTS,
    alpha: float = DBA_ALPHA,
    seed: int = 42,
    aug_target: Optional[int] = None,
    aug_by_y9: bool = True,
    aug_top_tier_only: bool = True,
) -> Dict:
    """DBA + jitter avec sur-représentation Expert (train_fit_real uniquement).

    Ratio par classe (synthétiques par trial réel) :
      Student/Junior : ×1.0  |  Senior : ×1.5  |  Expert : ×3.0
    """
    del aug_target, aug_by_y9, aug_top_tier_only  # legacy args — ratios fixes V2

    expert_trials = {k: v for k, v in train_real.items() if trial_y4_reg(v)[0] == 3}
    senior_trials = {k: v for k, v in train_real.items() if trial_y4_reg(v)[0] == 2}
    other_trials = {k: v for k, v in train_real.items() if trial_y4_reg(v)[0] < 2}

    real_counts = _count_trials_by_y4(train_real)
    synth: Dict = {}
    synth_i = 0

    for trials, ratio, tag in (
        (other_trials, 1.0, "student_junior"),
        (senior_trials, 1.5, "senior"),
        (expert_trials, 3.0, "expert"),
    ):
        part, synth_i = _dba_jitter_augment(
            trials, ratio=ratio, fold_i=fold_i, seed=seed,
            n_parents=n_parents, alpha=alpha, synth_start_i=synth_i,
            class_tag=tag,
        )
        synth.update(part)

    aug_counts = _count_trials_by_y4({**train_real, **synth})
    print(
        f"[Fold {fold_i}] Augmentation : "
        f"Expert {real_counts[3]}→{aug_counts[3]} trials | "
        f"Senior {real_counts[2]}→{aug_counts[2]} | "
        f"Student {real_counts[0]}→{aug_counts[0]}"
    )
    if synth:
        print(f"  [aug] +{len(synth)} synthétiques (ratios Expert×3 / Senior×1.5 / autres×1)")
    return synth


def _augment_train_fold_global(
    train_real: Dict,
    fold_i: int,
    n_parents: int = DBA_N_PARENTS,
    alpha: float = DBA_ALPHA,
    seed: int = 42,
    use_magnitude_warp: bool = False,
) -> Dict:
    """Augmentation globale équilibrée — DBA + jitter + time-warp par classe y4."""
    from icems_strategy import AUGMENTATION_GLOBALE

    by_y4: Dict[int, List[dict]] = defaultdict(list)
    for k, v in train_real.items():
        entry = _rec_to_entry(k, v)
        by_y4[int(entry["expertise_idx"])].append(entry)

    real_counts = _count_trials_by_y4(train_real)
    synth: Dict = {}
    synth_i = 0

    for y4, class_name in enumerate(CLASS4_NAMES):
        cfg = AUGMENTATION_GLOBALE[class_name]
        pool = by_y4.get(y4, [])
        if not pool:
            continue
        rng = np.random.default_rng(seed + fold_i * 1000 + y4 * 31)
        tag = class_name.lower()

        for aug_type, count_key in (
            ("dba", "dba"),
            ("jitter", "jitter"),
            ("timewarp", "timewarp"),
        ):
            for _ in range(cfg.get(count_key, 0)):
                part, synth_i = _generate_synth_entry(
                    pool, aug_type, fold_i, synth_i, rng,
                    n_parents=n_parents, alpha=alpha, class_tag=tag,
                )
                synth.update(part)

        if use_magnitude_warp:
            for _ in range(cfg.get("magnitude", 0)):
                part, synth_i = _generate_synth_entry(
                    pool, "magnitude", fold_i, synth_i, rng,
                    n_parents=n_parents, alpha=alpha, class_tag=tag,
                )
                synth.update(part)

    aug_counts = _count_trials_by_y4({**train_real, **synth})
    print(
        f"[Fold {fold_i}] Augmentation GLOBALE : "
        f"Expert {real_counts[3]}→{aug_counts[3]} | "
        f"Senior {real_counts[2]}→{aug_counts[2]} | "
        f"Junior {real_counts[1]}→{aug_counts[1]} | "
        f"Student {real_counts[0]}→{aug_counts[0]}"
    )
    if synth:
        print(f"  [aug] +{len(synth)} synthétiques (DBA+jitter+timewarp par classe)")
    return synth


def _resolve_augmentation(
    train_fit_real: Dict,
    fold_i: int,
    aug_mode: str,
    seed: int,
    aug_target: Optional[int] = None,
    aug_by_y9: bool = True,
    aug_top_tier_only: bool = True,
) -> Dict:
    """Route l'augmentation selon le mode (none | v2 | global)."""
    if aug_mode == "none":
        return {}
    if aug_mode == "global":
        return _augment_train_fold_global(train_fit_real, fold_i=fold_i, seed=seed)
    return _augment_train_fold_inline(
        train_fit_real, fold_i=fold_i, aug_target=aug_target,
        aug_by_y9=aug_by_y9, aug_top_tier_only=aug_top_tier_only, seed=seed,
    )


def _normalize_dataset(dataset) -> Dict:
    """Trials réels uniquement — pas de synthétiques pré-générés."""
    if isinstance(dataset, list):
        real = [e for e in dataset if not e.get("is_augmented", False)]
        out: Dict = {}
        for entry in real:
            pid = str(entry["participant"])
            trial = str(entry.get("trial", entry["name"]))
            out[(pid, trial)] = _entry_to_train_rec(entry)
        return out
    return {
        k: v for k, v in dataset.items()
        if not str(k[0]).startswith("synth_") and not v.get("is_augmented", False)
    }


def run_lopo(
    dataset: Dict,
    device: torch.device,
    epochs: int,
    max_folds: Optional[int],
    patience: int,
    batch_size: int,
    mc_passes: int = MC_PASSES,
    aug_target: Optional[int] = None,
    no_inline_aug: bool = False,
    aug_mode: str = "v2",
    aug_by_y9: bool = True,
    aug_top_tier_only: bool = True,
    sep_weight: float = 0.5,
    margin: float = 0.3,
    anchor_weight: float = 0.0,
    top_tier_weight: float = 0.0,
    staff_weight: float = 1.0,
    staff_pos_weight: float = 5.0,
    agg_percentile: Optional[float] = None,
    model_name: str = "causal_gru",
    loss_mode: str = "mse",
    focal_alpha: float = 0.1,
    focal_gamma: float = 2.0,
    seed: int = NESTED_LOPO_SEED,
) -> Tuple[List[dict], Dict]:
    """LOPO nested : early stopping sur validation interne ; held-out réservé à predict()."""
    dataset = _normalize_dataset(dataset)
    by_pid = defaultdict(list)
    for k in dataset:
        by_pid[k[0]].append(k)

    pids = sorted(by_pid.keys())
    if max_folds is not None:
        pids = pids[:max_folds]

    all_preds = []
    curves = []
    fold_train_info: List[dict] = []

    for fold_i, held_pid in enumerate(pids):
        held_keys = set(by_pid[held_pid])
        train_pool_real = {k: v for k, v in dataset.items() if k not in held_keys}
        train_pool_pids = sorted({k[0] for k in train_pool_real})
        int_val_pids = select_internal_val_participants(
            train_pool_pids, fold_i, dataset=dataset,
        )
        int_val_set = set(int_val_pids)
        train_fit_pids = [p for p in train_pool_pids if p not in int_val_set]
        assert_disjoint_participant_splits(held_pid, int_val_pids, train_fit_pids)

        train_fit_real = {
            k: v for k, v in train_pool_real.items() if k[0] not in int_val_set
        }
        internal_val = {
            k: v for k, v in train_pool_real.items() if k[0] in int_val_set
        }
        test_trials = {k: v for k, v in dataset.items() if k in held_keys}

        print(
            f"\n[LOPO fold {fold_i + 1}/{len(pids)}] held out : {held_pid} | "
            f"internal val : {int_val_pids} | train fit : {len(train_fit_pids)} participants"
        )

        effective_aug = "none" if no_inline_aug else aug_mode
        synth = _resolve_augmentation(
            train_fit_real, fold_i=fold_i, aug_mode=effective_aug, seed=seed,
            aug_target=aug_target, aug_by_y9=aug_by_y9, aug_top_tier_only=aug_top_tier_only,
        )
        exclude = int_val_set | {str(held_pid)}
        train = build_train_trials(train_fit_real, synth, exclude)
        assert_no_leakage(train, internal_val, test_trials, held_pid, int_val_pids)

        model, train_info = train_fold(
            train, internal_val, device, epochs=epochs,
            lr=1e-3, weight_decay=1e-4, patience=patience,
            batch_size=batch_size,
            sep_weight=sep_weight,
            margin=margin,
            anchor_weight=anchor_weight,
            top_tier_weight=top_tier_weight,
            staff_weight=staff_weight,
            staff_pos_weight=staff_pos_weight,
            model_name=model_name,
            loss_mode=loss_mode,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        train_info["participant"] = held_pid
        train_info["internal_val_participants"] = int_val_pids
        train_info["fold"] = fold_i + 1
        fold_train_info.append(train_info)

        if train_info["early_stopped"]:
            print(
                f"  Early stopping à l'epoch {train_info['stopped_epoch']}/{epochs} "
                f"(meilleur modèle : epoch {train_info['best_epoch']}, "
                f"internal val_loss={train_info['best_val_loss']:.5f})"
            )
        else:
            print(
                f"  Entraînement complet jusqu'à l'epoch {epochs}/{epochs} "
                f"(meilleur modèle : epoch {train_info['best_epoch']}, "
                f"internal val_loss={train_info['best_val_loss']:.5f})"
            )
        mean, std = model._norm_mean, model._norm_std

        for key in held_keys:
            rec = dataset[key]
            x = apply_norm(rec["X"], mean, std)
            mean_t, std_t, score, unc, staff_prob = predict_trial_mc(
                model, x, device, n_passes=mc_passes,
                agg_percentile=agg_percentile,
            )
            y4, y4_reg = trial_y4_reg(rec)
            y9, y9_reg = trial_y9_reg(rec)
            pred_cls = score_to_class(score)
            pred_y9 = score_to_y9(score)
            all_preds.append({
                "key": key,
                "y9": y9,
                "y_reg": y9_reg,
                "y4": y4,
                "y4_reg": y4_reg,
                "score": score,
                "uncertainty": unc,
                "staff_prob": staff_prob,
                "pred_class": pred_cls,
                "pred_y9": pred_y9,
                "participant": held_pid,
                "sublevel": sublevel_from_record(rec, y9=y9),
                "level": rec.get("level", SUBLEVEL_NAMES[y9]),
            })
            T_plot = len(mean_t)
            t_norm = np.linspace(0, 1, T_plot)
            curves.append({
                "t_norm": t_norm,
                "mean": mean_t,
                "std": std_t,
                "y4": y4,
                "y9": y9,
                "key": key,
            })

    return all_preds, {"curves": curves, "fold_train_info": fold_train_info}


def run_corrected_lopo(
    dataset: Dict,
    device: torch.device,
    epochs: int,
    max_folds: Optional[int] = None,
    patience: int = 25,
    batch_size: int = 16,
    mc_passes: int = MC_PASSES,
    n_inner_val: int = N_INTERNAL_VAL_PARTICIPANTS,
    seed: int = NESTED_LOPO_SEED,
    aug_target: Optional[int] = None,
    no_inline_aug: bool = False,
    aug_by_y9: bool = True,
    aug_top_tier_only: bool = True,
    sep_weight: float = 0.5,
    margin: float = 0.3,
    anchor_weight: float = 0.0,
    top_tier_weight: float = 0.0,
    staff_weight: float = 1.0,
    staff_pos_weight: float = 5.0,
    agg_percentile: Optional[float] = None,
    verbose_split_debug: bool = False,
    aug_mode: str = "v2",
    model_name: str = "causal_gru",
    loss_mode: str = "mse",
    focal_alpha: float = 0.1,
    focal_gamma: float = 2.0,
) -> Tuple[pd.DataFrame, Dict, List[dict], Dict]:
    """LOPO nested anti-fuite : early stopping sur val interne ; test = held-out uniquement.

    Returns
    -------
    preds_df : DataFrame TEST-only [participant, trial, true_score, pred_score, group,
               is_augmented, fold_idx]
    metrics : dict agrégé (r_trial, r_participant, r_per_fold, MAE, spearman_rho, ...)
    all_preds : liste legacy compatible print_metrics / graphiques
    aux : courbes + fold_train_info
    """
    dataset = _normalize_dataset(dataset)
    by_pid = defaultdict(list)
    for k in dataset:
        by_pid[k[0]].append(k)

    pids = sorted(by_pid.keys())
    if max_folds is not None:
        pids = pids[:max_folds]

    pred_rows: List[dict] = []
    all_preds: List[dict] = []
    curves: List[dict] = []
    fold_train_info: List[dict] = []
    r_per_fold: List[float] = []

    for fold_i, p_test in enumerate(pids):
        held_keys = set(by_pid[p_test])
        test_trials = {k: dataset[k] for k in held_keys}

        train_pool_real = {k: v for k, v in dataset.items() if k not in held_keys}
        train_pool_pids = sorted({k[0] for k in train_pool_real})

        val_participants = select_stratified_val_participants(
            train_pool_pids, dataset, k=n_inner_val, seed=seed, fold_idx=fold_i,
        )
        val_set = set(val_participants)
        train_fit_pids = [p for p in train_pool_pids if p not in val_set]
        assert_disjoint_participant_splits(p_test, val_participants, train_fit_pids)

        val_trials = {k: v for k, v in train_pool_real.items() if k[0] in val_set}
        train_fit_real = {k: v for k, v in train_pool_real.items() if k[0] not in val_set}

        print(
            f"\n[corrected LOPO fold {fold_i + 1}/{len(pids)}] TEST={p_test} | "
            f"internal VAL={val_participants} | TRAIN fit={len(train_fit_pids)} p"
        )

        effective_aug = "none" if no_inline_aug else aug_mode
        synth = _resolve_augmentation(
            train_fit_real, fold_i=fold_i, aug_mode=effective_aug, seed=seed,
            aug_target=aug_target, aug_by_y9=aug_by_y9, aug_top_tier_only=aug_top_tier_only,
        )

        exclude = val_set | {str(p_test)}
        train_trials = build_train_trials(train_fit_real, synth, exclude)
        assert_no_leakage(train_trials, val_trials, test_trials, p_test, val_participants)

        if verbose_split_debug or fold_i == 0:
            print_fold_split_debug(
                fold_i, train_trials, val_trials, test_trials, p_test, val_participants,
            )

        model, train_info = train_fold(
            train_trials, val_trials, device, epochs=epochs,
            lr=1e-3, weight_decay=1e-4, patience=patience,
            batch_size=batch_size,
            sep_weight=sep_weight,
            margin=margin,
            anchor_weight=anchor_weight,
            top_tier_weight=top_tier_weight,
            staff_weight=staff_weight,
            staff_pos_weight=staff_pos_weight,
            model_name=model_name,
            loss_mode=loss_mode,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        train_info["participant"] = p_test
        train_info["internal_val_participants"] = val_participants
        train_info["fold"] = fold_i + 1
        fold_train_info.append(train_info)

        if train_info["early_stopped"]:
            print(
                f"  Early stopping epoch {train_info['stopped_epoch']}/{epochs} "
                f"(best={train_info['best_epoch']}, val_loss={train_info['best_val_loss']:.5f})"
            )

        mean, std = model._norm_mean, model._norm_std
        fold_true, fold_pred = [], []

        for key in held_keys:
            rec = dataset[key]
            x = apply_norm(rec["X"], mean, std)
            mean_t, std_t, score, unc, staff_prob = predict_trial_mc(
                model, x, device, n_passes=mc_passes,
                agg_percentile=agg_percentile,
            )
            y4, y4_reg = trial_y4_reg(rec)
            y9, y9_reg = trial_y9_reg(rec)
            pred_cls = score_to_class(score)
            pred_y9 = score_to_y9(score)

            fold_true.append(y9_reg)
            fold_pred.append(score)

            pred_rows.append({
                "participant": str(p_test),
                "trial": str(key[1]),
                "true_score": float(y9_reg),
                "pred_score": float(score),
                "group": CLASS4_NAMES[y4],
                "is_augmented": False,
                "fold_idx": fold_i,
            })
            all_preds.append({
                "key": key,
                "y9": y9,
                "y_reg": y9_reg,
                "y4": y4,
                "y4_reg": y4_reg,
                "score": score,
                "uncertainty": unc,
                "staff_prob": staff_prob,
                "pred_class": pred_cls,
                "pred_y9": pred_y9,
                "participant": p_test,
                "fold_idx": fold_i,
                "sublevel": sublevel_from_record(rec, y9=y9),
                "level": rec.get("level", SUBLEVEL_NAMES[y9]),
            })
            T_plot = len(mean_t)
            curves.append({
                "t_norm": np.linspace(0, 1, T_plot),
                "mean": mean_t,
                "std": std_t,
                "y4": y4,
                "y9": y9,
                "key": key,
            })

        r_fold = float("nan")
        if len(fold_true) >= 2:
            ft = np.asarray(fold_true, dtype=float)
            fp = np.asarray(fold_pred, dtype=float)
            if ft.std() > 1e-8 and fp.std() > 1e-8:
                with np.errstate(invalid="ignore"):
                    r_fold, _ = pearsonr(ft, fp)
                r_fold = float(r_fold)
        r_per_fold.append(r_fold)

    preds_df = pd.DataFrame(pred_rows)
    metrics = compute_corrected_lopo_metrics(preds_df, r_per_fold)
    aux = {"curves": curves, "fold_train_info": fold_train_info}
    return preds_df, metrics, all_preds, aux


def compute_corrected_lopo_metrics(
    preds_df: pd.DataFrame,
    r_per_fold: List[float],
) -> Dict:
    """Agrège r_trial, r_participant, r_per_fold, MAE, Spearman (trial + participant)."""
    if preds_df.empty:
        return {}

    y_true = preds_df["true_score"].to_numpy(dtype=float)
    y_pred = preds_df["pred_score"].to_numpy(dtype=float)

    r_trial, p_trial = pearsonr(y_true, y_pred)
    rho_trial, sp_trial = spearmanr(y_true, y_pred)
    mae_trial = float(np.mean(np.abs(y_true - y_pred)))

    by_part = preds_df.groupby("participant", as_index=False).agg(
        true_score=("true_score", "mean"),
        pred_score=("pred_score", "mean"),
    )
    y_p_true = by_part["true_score"].to_numpy(dtype=float)
    y_p_pred = by_part["pred_score"].to_numpy(dtype=float)

    if len(by_part) >= 2:
        r_participant, p_participant = pearsonr(y_p_true, y_p_pred)
        rho_participant, sp_participant = spearmanr(y_p_true, y_p_pred)
    else:
        r_participant = p_participant = rho_participant = sp_participant = float("nan")

    mae_participant = float(np.mean(np.abs(y_p_true - y_p_pred)))
    r_valid = [r for r in r_per_fold if not np.isnan(r)]

    return {
        "r_trial": float(r_trial),
        "r_trial_pvalue": float(p_trial),
        "r_participant": float(r_participant),
        "r_participant_pvalue": float(p_participant),
        "r_per_fold": [float(r) for r in r_per_fold],
        "r_per_fold_mean": float(np.mean(r_valid)) if r_valid else float("nan"),
        "r_per_fold_std": float(np.std(r_valid)) if r_valid else float("nan"),
        "mae": mae_trial,
        "mae_trial": mae_trial,
        "mae_participant": mae_participant,
        "spearman_rho": float(rho_participant),
        "spearman_rho_trial": float(rho_trial),
        "spearman_rho_trial_pvalue": float(sp_trial),
        "spearman_rho_participant": float(rho_participant),
        "spearman_rho_participant_pvalue": float(sp_participant),
        "n_trials": int(len(preds_df)),
        "n_participants": int(len(by_part)),
        "n_folds": int(len(r_per_fold)),
    }


def print_corrected_lopo_metrics(metrics: Dict) -> None:
    if not metrics:
        print("Aucune métrique (predictions vides).")
        return
    print("\n" + "=" * 60)
    print(" Métriques LOPO corrigées (nested validation, TEST only)")
    print("=" * 60)
    print(
        f"  r_trial        = {metrics['r_trial']:+.4f}  "
        f"(p = {metrics['r_trial_pvalue']:.4e})  n_trials = {metrics['n_trials']}"
    )
    print(
        f"  r_participant  = {metrics['r_participant']:+.4f}  "
        f"(p = {metrics['r_participant_pvalue']:.4e})  n = {metrics['n_participants']}"
    )
    print(
        f"  r_per_fold     = {metrics['r_per_fold_mean']:+.4f} "
        f"± {metrics['r_per_fold_std']:.4f}  ({metrics['n_folds']} folds)"
    )
    print(f"  MAE trial      = {metrics['mae_trial']:.4f}")
    print(f"  MAE participant= {metrics['mae_participant']:.4f}")
    print(
        f"  Spearman ρ trial       = {metrics['spearman_rho_trial']:+.4f}  "
        f"(p = {metrics['spearman_rho_trial_pvalue']:.4e})"
    )
    print(
        f"  Spearman ρ participant = {metrics['spearman_rho_participant']:+.4f}  "
        f"(p = {metrics['spearman_rho_participant_pvalue']:.4e})"
    )


def save_corrected_lopo_results(
    out_dir: Path,
    preds_df: pd.DataFrame,
    metrics: Dict,
    all_preds: List[dict],
    aux: Dict,
) -> Tuple[Path, Path]:
    """Sauvegarde lopo_predictions.pkl (DataFrame TEST) + metrics_summary.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / LOPO_PREDICTIONS_FILE
    metrics_path = out_dir / "metrics_summary.json"

    payload = {
        "preds_df": preds_df,
        "preds": all_preds,
        "curves": aux.get("curves", []),
        "fold_train_info": aux.get("fold_train_info", []),
        "metrics": metrics,
    }
    with open(pred_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            _sanitize_for_json(metrics),
            f, indent=2, ensure_ascii=False, allow_nan=False,
        )

    return pred_path, metrics_path


def corrected_lopo_output_dir(base: Optional[Path] = None) -> Path:
    root = base or CORRECTED_LOPO_RESULTS_ROOT
    return root / date.today().isoformat()


def _json_default(obj):
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _sanitize_for_json(obj):
    """Convertit récursivement NaN/Inf en null pour JSON valide."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    return obj


def regenerate_metrics_from_predictions_pkl(pkl_path: Path) -> Tuple[Dict, pd.DataFrame]:
    """Recalcule metrics_summary depuis lopo_predictions.pkl (sans ré-entraîner)."""
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    preds_df = payload.get("preds_df")
    if preds_df is None:
        preds_df = pd.DataFrame(payload.get("preds", []))
    r_per_fold: List[float] = []
    for _, grp in preds_df.groupby("fold_idx", sort=True):
        ft = grp["true_score"].to_numpy(dtype=float)
        fp = grp["pred_score"].to_numpy(dtype=float)
        r_fold = float("nan")
        if len(ft) >= 2 and ft.std() > 1e-8 and fp.std() > 1e-8:
            with np.errstate(invalid="ignore"):
                r_fold, _ = pearsonr(ft, fp)
            r_fold = float(r_fold)
        r_per_fold.append(r_fold)
    metrics = compute_corrected_lopo_metrics(preds_df, r_per_fold)
    return metrics, preds_df

def print_early_stopping_summary(fold_train_info: List[dict], max_epochs: int) -> None:
    if not fold_train_info:
        return
    print("\n" + "=" * 60)
    print(" Early stopping par fold LOPO (validation interne, pas held-out)")
    print("=" * 60)
    print(
        f"  {'Fold':>4}  {'Held out':<12}  {'Internal val':<22}  "
        f"{'Stop':>10}  {'Best ep':>8}  {'Int. val loss':>13}"
    )
    print("  " + "-" * 78)
    for info in fold_train_info:
        stop_label = (
            f"{info['stopped_epoch']}/{max_epochs}"
            if info["early_stopped"]
            else f"{max_epochs}/{max_epochs}*"
        )
        int_val = ",".join(info.get("internal_val_participants", []))
        print(
            f"  {info['fold']:>4}  {info['participant']:<12}  {int_val:<22}  "
            f"{stop_label:>10}  {info['best_epoch']:>8}  {info['best_val_loss']:>13.5f}"
        )
    n_early = sum(1 for i in fold_train_info if i["early_stopped"])
    avg_stop = np.mean([i["stopped_epoch"] for i in fold_train_info])
    print(f"\n  * pas d'early stopping (epochs max atteintes)")
    print(f"  Folds avec early stopping : {n_early}/{len(fold_train_info)}")
    print(f"  Epoch moyenne d'arrêt     : {avg_stop:.1f}/{max_epochs}")


def save_lopo_results(
    out_dir: Path, preds: List[dict], aux: Dict, metrics: Optional[Dict] = None,
) -> Path:
    """Sauvegarde prédictions + courbes temporelles pour régénération sans LOPO."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / LOPO_PREDICTIONS_FILE
    payload = {
        "preds": preds,
        "curves": aux["curves"],
        "fold_train_info": aux.get("fold_train_info", []),
    }
    if metrics is not None:
        payload["metrics"] = metrics
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def lopo_v2_output_dir(base: Optional[Path] = None) -> Path:
    root = base or LOPO_V2_RESULTS_ROOT
    return root / date.today().isoformat()


def _normalize_participant_id(pid) -> str:
    """ID participant 8 chiffres (ex. 01020614), compatible float CSV."""
    s = str(pid).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s.zfill(8) if s.isdigit() else s


def _default_participant_csv() -> Optional[Path]:
    """CSV métadonnées participants (TABLE I) si présent dans le repo."""
    for root in (Path(__file__).resolve().parent.parent, Path.cwd()):
        candidate = root / "data" / "Exvivo_trial_Participants(Sheet1).csv"
        if candidate.exists():
            return candidate
    return None


def sublevel_from_record(rec: dict, y9: Optional[int] = None) -> str:
    """Nom sous-niveau publication (TABLE I) depuis y9 ou champ level."""
    if y9 is None:
        y9 = int(rec.get("y9", 0))
    level = str(rec.get("level", "")).strip()
    if level:
        level_lower = level.lower()
        for paper_name in PAPER_SUBLEVEL_BY_Y9:
            if paper_name.lower().replace(" ", "") in level_lower.replace(" ", ""):
                return paper_name
        if level_lower.startswith("staff"):
            return "Neurosurgeon"
        if level_lower.startswith("medical"):
            return "Medical Student"
    if 0 <= y9 < len(PAPER_SUBLEVEL_BY_Y9):
        return PAPER_SUBLEVEL_BY_Y9[y9]
    return PAPER_SUBLEVEL_BY_Y9[0]


def enrich_preds_with_sublevel(
    preds: List[dict],
    dataset: Optional[Dict] = None,
    participant_csv: Optional[Path] = None,
) -> List[dict]:
    """Ajoute le champ ``sublevel`` (TABLE I) à chaque prédiction."""
    pid_level: Dict[str, str] = {}
    csv_path = participant_csv or _default_participant_csv()
    if csv_path and csv_path.exists():
        df = pd.read_csv(
            csv_path,
            sep=";",
            encoding="utf-8",
            on_bad_lines="skip",
            dtype={"ID": str},
        )
        id_col = next((c for c in df.columns if c.strip().lower() == "id"), None)
        level_col = next((c for c in df.columns if c.strip().lower() == "level"), None)
        if id_col and level_col:
            for _, row in df.iterrows():
                pid = _normalize_participant_id(row[id_col])
                lvl = str(row[level_col]).strip()
                if pid and lvl and lvl.lower() != "nan":
                    pid_level[pid] = lvl

    key_level: Dict[Tuple, str] = {}
    if dataset:
        for k, rec in dataset.items():
            key_level[k] = str(rec.get("level", ""))

    enriched = []
    for p in preds:
        row = dict(p)
        if row.get("sublevel"):
            enriched.append(row)
            continue
        y9 = int(row.get("y9", 0))
        pid = _normalize_participant_id(row.get("participant", ""))
        level = pid_level.get(pid, "")
        if not level:
            key = row.get("key")
            if key and key in key_level:
                level = key_level[key]
        fake_rec = {"y9": y9, "level": level}
        row["sublevel"] = sublevel_from_record(fake_rec, y9=y9)
        enriched.append(row)
    return enriched


def compute_v2_metrics_with_baseline(preds: List[dict]) -> Dict:
    """Métriques V2 + comparaison baseline (run 9 juin)."""
    y9_reg = np.array([p["y_reg"] for p in preds], dtype=float)
    y_pred = np.array([p["score"] for p in preds], dtype=float)
    y4 = np.array([p["y4"] for p in preds], dtype=int)
    pred_cls = np.array([p["pred_class"] for p in preds], dtype=int)

    by_pid: Dict[str, List[float]] = defaultdict(list)
    by_pid_true: Dict[str, float] = {}
    for p in preds:
        pid = str(p["participant"])
        by_pid[pid].append(float(p["score"]))
        by_pid_true[pid] = float(p["y4_reg"])

    p_true = np.array([by_pid_true[pid] for pid in sorted(by_pid.keys())])
    p_pred = np.array([float(np.mean(by_pid[pid])) for pid in sorted(by_pid.keys())])

    r_trial = float(pearsonr(y9_reg, y_pred)[0]) if len(preds) >= 2 else float("nan")
    r_participant = float(pearsonr(p_true, p_pred)[0]) if len(p_true) >= 2 else float("nan")
    rho = float(spearmanr(y9_reg, y_pred)[0]) if len(preds) >= 2 else float("nan")
    mae = float(np.mean(np.abs(y9_reg - y_pred)))
    acc = float((pred_cls == y4).mean())

    expert_mask = y4 == 3
    senior_mask = y4 == 2
    expert_recall = (
        float((pred_cls[expert_mask] == 3).mean()) if expert_mask.any() else float("nan")
    )
    mean_expert = float(y_pred[expert_mask].mean()) if expert_mask.any() else float("nan")
    mean_senior = float(y_pred[senior_mask].mean()) if senior_mask.any() else float("nan")
    senior_expert_diff = (
        mean_expert - mean_senior
        if not (np.isnan(mean_expert) or np.isnan(mean_senior))
        else float("nan")
    )

    v2 = {
        "r_trial": r_trial,
        "r_participant": r_participant,
        "spearman_rho": rho,
        "mae": mae,
        "accuracy": acc,
        "expert_recall": expert_recall,
        "mean_expert_score": mean_expert,
        "mean_senior_score": mean_senior,
        "senior_expert_diff": senior_expert_diff,
        "max_pred_score": float(y_pred.max()) if len(y_pred) else float("nan"),
        "n_trials": len(preds),
        "n_participants": len(by_pid),
    }

    comparison = {}
    for key, baseline_val in BASELINE_METRICS.items():
        v2_val = v2.get(key)
        if v2_val is None or (isinstance(v2_val, float) and np.isnan(v2_val)):
            continue
        if key in ("mae",):
            improved = v2_val < baseline_val
        elif key in ("senior_expert_diff", "max_pred_score", "mean_expert_score",
                     "expert_recall", "r_participant", "r_trial", "spearman_rho", "accuracy"):
            improved = v2_val > baseline_val
        else:
            improved = None
        comparison[key] = {
            "baseline": baseline_val,
            "v2": round(v2_val, 4) if isinstance(v2_val, float) else v2_val,
            "delta": round(v2_val - baseline_val, 4) if isinstance(v2_val, float) else None,
            "improved": improved,
        }

    return {"v2": v2, "baseline": BASELINE_METRICS, "comparison": comparison}


def save_loss_components_per_fold(fold_train_info: List[dict], out_path: Path) -> None:
    payload = []
    for info in fold_train_info:
        payload.append({
            "fold": info.get("fold"),
            "participant": info.get("participant"),
            "loss_components": info.get("loss_components"),
            "loss_epochs": info.get("loss_epochs", []),
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(payload), f, indent=2, ensure_ascii=False)


def prepare_curves_for_granular_plots(curves: List[dict]) -> List[dict]:
    """Convertit les courbes LOPO au format attendu par plot_granular_9sublevels."""
    time_grid = np.linspace(0.0, 1.0, 100)
    out = []
    for c in curves:
        y9 = int(c.get("y9", 0))
        sublevel = PAPER_SUBLEVEL_BY_Y9[y9] if 0 <= y9 < 9 else PAPER_SUBLEVEL_BY_Y9[0]
        mean = np.asarray(c["mean"], dtype=float)
        t_norm = np.asarray(c["t_norm"], dtype=float)
        out.append({
            "sublevel": sublevel,
            "time": t_norm,
            "scores": mean,
            "scores_interp": np.interp(time_grid, t_norm, mean),
            "time_grid": time_grid,
        })
    return out


def load_lopo_results(path: Path) -> Tuple[List[dict], Dict]:
    """Charge prédictions LOPO sauvegardées."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    preds = payload.get("preds", [])
    preds = enrich_preds_with_sublevel(list(preds))
    aux = {
        "curves": payload.get("curves", []),
        "fold_train_info": payload.get("fold_train_info", []),
        "metrics": payload.get("metrics"),
    }
    return preds, aux


def resolve_lopo_predictions(out_dir: Path, explicit: Optional[Path] = None) -> Path:
    """Cherche le fichier de prédictions (chemin explicite ou dossier de sortie)."""
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Fichier de prédictions introuvable : {explicit}")
        return explicit
    path = out_dir / LOPO_PREDICTIONS_FILE
    if path.exists():
        return path
    raise FileNotFoundError(
        f"Aucune prédiction sauvegardée dans {out_dir / LOPO_PREDICTIONS_FILE}.\n"
        f"Lancez d'abord le LOPO complet (sans --plot-only) pour générer ce fichier."
    )


def print_metrics(preds: List[dict]) -> None:
    y9_true = np.array([p["y9"] for p in preds])
    y9_reg = np.array([p["y_reg"] for p in preds])
    y_pred = np.array([p["score"] for p in preds])
    y4 = np.array([p["y4"] for p in preds])
    pred_cls = np.array([p["pred_class"] for p in preds])
    pred_y9 = np.array([p.get("pred_y9", score_to_y9(p["score"])) for p in preds])

    pr9, pp9 = pearsonr(y9_reg, y_pred)
    sr9, sp9 = spearmanr(y9_true, y_pred)
    acc4 = float((pred_cls == y4).mean())
    acc9 = float((pred_y9 == y9_true).mean())

    # Focus Senior (y4=2) vs Expert (y4=3)
    mask_se = (y4 == 2) | (y4 == 3)
    if mask_se.any():
        y4_se = y4[mask_se]
        pred_se = pred_cls[mask_se]
        acc_se = float((pred_se == y4_se).mean())
        expert_mask = y4_se == 3
        expert_recall = float((pred_se[expert_mask] == 3).mean()) if expert_mask.any() else float("nan")
        n_expert_as_senior = int(((y4_se == 3) & (pred_se == 2)).sum())
        n_expert = int(expert_mask.sum())
    else:
        acc_se, expert_recall, n_expert_as_senior, n_expert = float("nan"), float("nan"), 0, 0

    print("\n" + "=" * 60)
    print(" Métriques LOPO (participants tenus out)")
    print("=" * 60)
    print(f"  Pearson  r (y9_reg)  = {pr9:+.4f}  (p = {pp9:.4e})  n = {len(preds)}")
    print(f"  Spearman r (y9 rang) = {sr9:+.4f}  (p = {sp9:.4e})")
    print(f"  Accuracy 9 sous-niv. = {acc9 * 100:.1f}%  ({int((pred_y9 == y9_true).sum())}/{len(preds)})")
    print(f"  Accuracy 4 classes   = {acc4 * 100:.1f}%  ({int((pred_cls == y4).sum())}/{len(preds)})")
    print(f"\n  --- Focus Senior / Expert (y4) ---")
    print(f"  Accuracy binaire S/E = {acc_se * 100:.1f}%  (n={int(mask_se.sum())})")
    print(f"  Recall Expert        = {expert_recall * 100:.1f}%  ({n_expert - n_expert_as_senior}/{n_expert})")
    print(f"  Expert→Senior        = {n_expert_as_senior}/{n_expert}")


def plot_score_vs_time(
    curves: List[dict],
    out_path: Path,
    n_trials: Optional[int] = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    SMOOTH_SIGMA = 30
    t_common = np.linspace(0.0, 1.0, 1000)

    by_class: Dict[int, List[np.ndarray]] = defaultdict(list)
    for c in curves:
        by_class[c["y4"]].append(np.interp(t_common, c["t_norm"], c["mean"]))

    target_lines = [
        (-1.00, "Student"),
        (-0.33, "Junior"),
        (+0.33, "Senior"),
        (+1.00, "Expert"),
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    for y4 in range(4):
        stack = by_class.get(y4)
        if not stack:
            continue
        arr = np.stack(stack)
        median_curve = np.median(arr, axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)

        median_smooth = gaussian_filter1d(median_curve, sigma=SMOOTH_SIGMA)
        p25_smooth = gaussian_filter1d(p25, sigma=SMOOTH_SIGMA)
        p75_smooth = gaussian_filter1d(p75, sigma=SMOOTH_SIGMA)

        color = CLASS4_COLORS[y4]
        ax.fill_between(t_common, p25_smooth, p75_smooth, color=color, alpha=0.15)
        ax.plot(
            t_common, median_smooth,
            color=color, lw=3.0, label=CLASS4_NAMES[y4], zorder=5,
        )

    for y_val, label in target_lines:
        ax.axhline(y_val, color="gray", ls=":", lw=0.8, alpha=0.6)
        ax.text(
            1.01, y_val, f"{label} ({y_val:+.2f})",
            transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=8, color="gray",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Temps normalisé")
    ax.set_ylabel("Score d'expertise [-1, +1]")
    n = n_trials if n_trials is not None else len(curves)
    ax.set_title(
        "Progression du score d'expertise au cours du geste\n"
        f"GRU Causal + LOPO · n={n} trials"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.subplots_adjust(right=0.84)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def _sublevel_color(y9: int):
    """Dégradé novice (rouge) → expert (vert) sur 9 sous-niveaux."""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("RdYlGn")
    return cmap(y9 / 8.0)


def _curves_by_sublevel(curves: List[dict]) -> Tuple[Dict[int, List[np.ndarray]], np.ndarray]:
    """Interpole chaque courbe trial sur une grille temporelle commune, groupée par y9."""
    t_common = np.linspace(0.0, 1.0, 1000)
    by_y9: Dict[int, List[np.ndarray]] = defaultdict(list)
    for c in curves:
        y9 = c.get("y9")
        if y9 is None:
            continue
        by_y9[int(y9)].append(np.interp(t_common, c["t_norm"], c["mean"]))
    return by_y9, t_common


def plot_score_vs_time_sublevels(
    curves: List[dict],
    out_path: Path,
    n_trials: Optional[int] = None,
) -> None:
    """
    Courbes médianes du score prédit (LOPO) par sous-niveau clinique (9 groupes).
    Chaque courbe = médiane des trials de ce sous-niveau ; bande = IQR (p25–p75).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    SMOOTH_SIGMA = 30
    by_y9, t_common = _curves_by_sublevel(curves)
    if not by_y9:
        print(f"  ⚠ Aucune courbe avec y9 — ignoré : {out_path}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for y9 in range(9):
        stack = by_y9.get(y9)
        if not stack:
            continue
        arr = np.stack(stack)
        median_curve = np.median(arr, axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)

        median_smooth = gaussian_filter1d(median_curve, sigma=SMOOTH_SIGMA)
        p25_smooth = gaussian_filter1d(p25, sigma=SMOOTH_SIGMA)
        p75_smooth = gaussian_filter1d(p75, sigma=SMOOTH_SIGMA)

        color = _sublevel_color(y9)
        label = f"{SUBLEVEL_NAMES[y9]} (n={len(stack)})"
        ax.fill_between(t_common, p25_smooth, p75_smooth, color=color, alpha=0.12)
        ax.plot(t_common, median_smooth, color=color, lw=2.2, label=label, zorder=5)

    for y9 in range(9):
        y_val = float(Y9_TO_REG[y9])
        ax.axhline(y_val, color="gray", ls=":", lw=0.6, alpha=0.45)

    ax.set_xlim(0, 1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Temps normalisé (0 = début du geste, 1 = fin)")
    ax.set_ylabel("Score d'expertise prédit [-1, +1]")
    n = n_trials if n_trials is not None else len(curves)
    ax.set_title(
        "Évolution du score prédit par sous-niveau clinique\n"
        f"CausalGRU + LOPO · {n} trials · lignes pointillées = cibles y9 (dataset)"
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.5, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def plot_score_vs_time_sublevels_grid(
    curves: List[dict],
    out_path: Path,
) -> None:
    """Grille 3×3 : un panneau par sous-niveau (médiane + IQR)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    SMOOTH_SIGMA = 30
    by_y9, t_common = _curves_by_sublevel(curves)
    if not by_y9:
        print(f"  ⚠ Aucune courbe avec y9 — ignoré : {out_path}")
        return

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for y9 in range(9):
        ax = axes_flat[y9]
        color = _sublevel_color(y9)
        stack = by_y9.get(y9)
        y_target = float(Y9_TO_REG[y9])

        if stack:
            arr = np.stack(stack)
            med = gaussian_filter1d(np.median(arr, axis=0), sigma=SMOOTH_SIGMA)
            p25 = gaussian_filter1d(np.percentile(arr, 25, axis=0), sigma=SMOOTH_SIGMA)
            p75 = gaussian_filter1d(np.percentile(arr, 75, axis=0), sigma=SMOOTH_SIGMA)
            ax.fill_between(t_common, p25, p75, color=color, alpha=0.25)
            ax.plot(t_common, med, color=color, lw=2.0)

        ax.axhline(y_target, color="gray", ls="--", lw=1.0, alpha=0.7)
        ax.set_title(f"{SUBLEVEL_NAMES[y9]}  (n={len(stack) if stack else 0})", fontsize=10)
        ax.grid(alpha=0.25)

    for ax in axes_flat[6:]:
        ax.set_xlabel("Temps normalisé")
    for ax in axes_flat[0::3]:
        ax.set_ylabel("Score prédit")
    fig.suptitle(
        "Score d'expertise prédit vs temps — un panneau par sous-niveau (LOPO)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def plot_score_vs_time_per_sublevel(
    curves: List[dict],
    out_dir: Path,
) -> None:
    """
    Un graphe dédié par sous-niveau (9 PNG) : courbes réelles trial-par-trial
    sur le temps normalisé + médiane / IQR + cible y9.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    by_y9: Dict[int, List[dict]] = defaultdict(list)
    for c in curves:
        y9 = c.get("y9")
        if y9 is None:
            continue
        by_y9[int(y9)].append(c)

    if not by_y9:
        print(f"  ⚠ Aucune courbe avec y9 — ignoré : {out_dir}")
        return

    t_common = np.linspace(0.0, 1.0, 500)

    for y9 in range(9):
        group = by_y9.get(y9, [])
        short = SUBLEVEL_SHORT[y9]
        name = SUBLEVEL_NAMES[y9]
        y_target = float(Y9_TO_REG[y9])
        color = _sublevel_color(y9)

        fig, ax = plt.subplots(figsize=(12, 6))

        for c in group:
            key = c.get("key", ("?", "?"))
            pid = key[0] if isinstance(key, (tuple, list)) else "?"
            trial = key[1] if isinstance(key, (tuple, list)) and len(key) > 1 else "?"
            ax.plot(
                c["t_norm"], c["mean"],
                color=color, alpha=0.38, lw=0.85, zorder=2,
                label=f"{pid}/{trial}" if len(group) <= 12 else None,
            )

        if group:
            stack = [np.interp(t_common, c["t_norm"], c["mean"]) for c in group]
            arr = np.stack(stack)
            med = np.median(arr, axis=0)
            p25 = np.percentile(arr, 25, axis=0)
            p75 = np.percentile(arr, 75, axis=0)
            ax.fill_between(
                t_common, p25, p75, color="black", alpha=0.12,
                label="IQR (p25–p75)", zorder=3,
            )
            ax.plot(
                t_common, med, color="black", lw=3.0,
                label=f"Médiane (n={len(group)})", zorder=5,
            )

        ax.axhline(
            y_target, color="#333333", ls="--", lw=1.6,
            label=f"Cible {short} ({y_target:+.2f})",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Temps normalisé (0 = début du geste, 1 = fin)")
        ax.set_ylabel("Score d'expertise prédit [-1, +1]")
        ax.set_title(
            f"{name} ({short}) — courbes réelles par trial (LOPO)\n"
            f"{len(group)} trial(s) · ligne pointillée = cible y9 du dataset"
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=7 if len(group) <= 12 else 8, framealpha=0.92)

        out_path = out_dir / f"score_vs_time_{short}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out_path}")


def plot_confusion(preds: List[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y4 = np.array([p["y4"] for p in preds])
    pred_cls = np.array([p["pred_class"] for p in preds])
    acc = float((pred_cls == y4).mean())
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(y4, pred_cls):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    try:
        import seaborn as sns
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=CLASS4_NAMES, yticklabels=CLASS4_NAMES,
        )
    except ImportError:
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(4), CLASS4_NAMES)
        ax.set_yticks(range(4), CLASS4_NAMES)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title(f"Matrice 4×4 — accuracy = {acc * 100:.1f}%")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  → {out_path}")


def plot_confusion_y9(preds: List[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short = ["MS", "PGY1", "PGY2", "PGY3", "PGY4", "PGY5", "PGY6", "Fellow", "Staff"]
    y9 = np.array([p["y9"] for p in preds])
    pred_y9 = np.array([p.get("pred_y9", score_to_y9(p["score"])) for p in preds])
    acc = float((pred_y9 == y9).mean())
    cm = np.zeros((9, 9), dtype=int)
    for t, p in zip(y9, pred_y9):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    try:
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=short, yticklabels=short)
    except ImportError:
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(9), short)
        ax.set_yticks(range(9), short)
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel("Sous-niveau prédit")
    ax.set_ylabel("Sous-niveau réel")
    ax.set_title(f"Matrice 9×9 — accuracy = {acc * 100:.1f}%")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  → {out_path}")


def plot_senior_expert_focus(preds: List[dict], out_path: Path) -> None:
    """Scatter scores : PGY6 / Fellow / Staff avec seuils y4 et y9."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {6: "#9467bd", 7: "#1f77b4", 8: "#2ca02c"}
    labels = {6: "PGY6 (Senior)", 7: "Fellow (Senior)", 8: "Staff (Expert)"}
    for y9 in (6, 7, 8):
        scores = [p["score"] for p in preds if p["y9"] == y9]
        if not scores:
            continue
        x = np.full(len(scores), y9) + np.random.default_rng(42).uniform(-0.12, 0.12, len(scores))
        ax.scatter(x, scores, c=colors[y9], label=f"{labels[y9]} (n={len(scores)})", s=55, alpha=0.8)
        ax.axhline(float(Y9_TO_REG[y9]), color=colors[y9], ls="--", lw=1, alpha=0.5)

    thresh_expert = (Y4_TO_REG[2] + Y4_TO_REG[3]) / 2
    ax.axhline(float(thresh_expert), color="red", ls=":", lw=1.5,
               label=f"Seuil Expert y4 ({thresh_expert:+.2f})")
    ax.set_xticks([6, 7, 8], ["PGY6", "Fellow", "Staff"])
    ax.set_ylabel("Score prédit")
    ax.set_title("Frontière Senior / Expert — scores LOPO par sous-niveau")
    ax.set_ylim(-0.3, 1.1)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  → {out_path}")


def plot_scatter_4class(preds: List[dict], out_path: Path) -> None:
    """Scatter score prédit vs réel agrégé 4 classes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.array([p["y4_reg"] for p in preds])
    y_pred = np.array([p["score"] for p in preds])
    y4 = np.array([p["y4"] for p in preds])
    pr, _ = pearsonr(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 7))
    for y4_i in range(4):
        m = y4 == y4_i
        if not m.any():
            continue
        ax.scatter(
            y_true[m], y_pred[m],
            c=CLASS4_COLORS[y4_i], label=CLASS4_NAMES[y4_i], alpha=0.75, s=50,
        )
    lims = [-1.1, 1.1]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Score réel (4 classes)")
    ax.set_ylabel("Score prédit")
    ax.set_title(f"Score prédit vs réel (4 classes) — Pearson r = {pr:+.3f}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  → {out_path}")


def _run_granular_plots(preds: List[dict], curves: List[dict], out_dir: Path) -> None:
    try:
        from plot_granular_9sublevels import run_granular_plots
    except ImportError:
        from src.plot_granular_9sublevels import run_granular_plots
    curves_data = prepare_curves_for_granular_plots(curves) if curves else None
    run_granular_plots(preds, curves_data, out_dir)


def plot_scatter(preds: List[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.array([p["y_reg"] for p in preds])
    y_pred = np.array([p["score"] for p in preds])
    y9 = np.array([p["y9"] for p in preds])
    pr, _ = pearsonr(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap("RdYlGn")
    for y9_i in range(9):
        m = y9 == y9_i
        if not m.any():
            continue
        ax.scatter(
            y_true[m], y_pred[m],
            c=[cmap(y9_i / 8.0)], label=SUBLEVEL_NAMES[y9_i], alpha=0.75, s=40,
        )
    lims = [-1.1, 1.1]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Score réel (y9_reg)")
    ax.set_ylabel("Score prédit")
    ax.set_title(f"Score prédit vs réel (9 sous-niv.) — Pearson r = {pr:+.3f}")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  → {out_path}")


def _run_senior_expert_diagnostic(preds: List[dict], out_dir: Path) -> None:
    try:
        from diagnose_senior_expert import plot_diagnostic, print_score_distribution, print_y4_confusion
    except ImportError:
        from src.diagnose_senior_expert import plot_diagnostic, print_score_distribution, print_y4_confusion
    print("\n[Diagnostic Senior/Expert]")
    print_score_distribution(preds)
    print_y4_confusion(preds)
    plot_diagnostic(preds, out_dir)


def _run_sublevel_analysis(preds: List[dict], out_dir: Path) -> None:
    """Violons + Spearman par sous-niveau (9 groupes)."""
    try:
        from sublevel_analysis import run_sublevel_analysis
    except ImportError:
        from src.sublevel_analysis import run_sublevel_analysis
    print("\n[Analyse sous-niveaux]")
    run_sublevel_analysis(preds, out_dir / "sublevel", prefix="icems_lopo")


def main():
    ap = argparse.ArgumentParser(description="Step B — classification LOPO + score continu.")
    ap.add_argument("--data", type=Path, default=Path("data/continuous_per_trial.pkl"))
    ap.add_argument("--out", type=Path, default=Path("results/sublevel_run"))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--max_folds", type=int, default=None,
                    help="Limite le nombre de folds LOPO (test rapide).")
    ap.add_argument("--folds", type=int, default=None, dest="max_folds",
                    help="Alias de --max_folds (ex. --folds 2 pour smoke test).")
    ap.add_argument(
        "--smoke-test",
        action="store_true",
        help="Smoke test V2 : 2 folds × 5 epochs, sortie results/lopo_v2/<date>/.",
    )
    ap.add_argument(
        "--lopo-v2",
        action="store_true",
        help="Sortie dans results/lopo_v2/<date>/ avec métriques baseline + figures granulaires.",
    )
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--sep-weight", type=float, default=0.5,
                    help="Poids de la pénalité de séparation inter-classes (y9).")
    ap.add_argument("--margin", type=float, default=0.15,
                    help="Marge minimale entre rangs y9 dans separation_loss.")
    ap.add_argument("--anchor-weight", type=float, default=1.0,
                    help="Poids ancrage MS / Fellow / Staff.")
    ap.add_argument("--top-tier-weight", type=float, default=1.0,
                    help="Poids marge renforcée PGY6 / Fellow / Staff.")
    ap.add_argument("--staff-weight", type=float, default=1.0,
                    help="Poids tête binaire Staff vs non-Staff (modif 3).")
    ap.add_argument("--staff-pos-weight", type=float, default=5.0,
                    help="pos_weight BCE pour la classe Staff (y9=8).")
    ap.add_argument(
        "--agg-percentile",
        type=float,
        default=None,
        help="Percentile frame pour score trial (ex. 90 pour réduire plafond Expert).",
    )
    ap.add_argument(
        "--aug-all-y9",
        action="store_true",
        help="DBA sur les 9 sous-niveaux (lent). Défaut : PGY6/Fellow/Staff uniquement.",
    )
    ap.add_argument(
        "--aug-by-y4",
        action="store_true",
        help="DBA par classe 4 niveaux (legacy). Défaut : DBA par sous-niveau y9.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mc-passes", type=int, default=MC_PASSES,
                    help="Passes MC Dropout (réduire pour test rapide).")
    ap.add_argument(
        "--aug-target",
        type=int,
        default=None,
        help="Trials par classe après DBA+jitter inline (ex. 200 → ~800 trials/fold). "
             "Défaut : équilibrer à la classe majoritaire du fold (Run 5).",
    )
    ap.add_argument(
        "--no-inline-aug",
        action="store_true",
        help="Désactive l'augmentation DBA+jitter inline (dataset déjà augmenté en amont).",
    )
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="Régénère uniquement score_vs_time.png depuis les prédictions sauvegardées.",
    )
    ap.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help=f"Chemin vers {LOPO_PREDICTIONS_FILE} (défaut : --out/{LOPO_PREDICTIONS_FILE}).",
    )
    args = ap.parse_args()

    smoke = args.smoke_test
    if smoke:
        if args.epochs == 150:
            args.epochs = 5
        if args.max_folds is None:
            args.max_folds = 2
        if args.mc_passes == MC_PASSES:
            args.mc_passes = 3
        args.lopo_v2 = True

    if args.lopo_v2 and args.out == Path("results/sublevel_run"):
        args.out = lopo_v2_output_dir()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        pred_path = resolve_lopo_predictions(args.out, args.predictions)
        preds, aux = load_lopo_results(pred_path)
        print("=" * 60)
        print(" Step B — plot-only (score vs temps)")
        print("=" * 60)
        print(f"\n[Chargement] {pred_path.resolve()}")
        print(f"  {len(preds)} prédictions, {len(aux['curves'])} courbes")
        print("\n[Graphiques]")
        plot_score_vs_time(
            aux["curves"],
            args.out / "score_vs_time_4class.png",
            n_trials=len(preds),
        )
        plot_score_vs_time_sublevels(
            aux["curves"],
            args.out / "score_vs_time_9sublevels.png",
            n_trials=len(preds),
        )
        plot_score_vs_time_sublevels_grid(
            aux["curves"],
            args.out / "score_vs_time_9sublevels_grid.png",
        )
        plot_score_vs_time_per_sublevel(
            aux["curves"],
            args.out / "sublevels_time",
        )
        _run_sublevel_analysis(preds, args.out)
        _run_granular_plots(preds, aux["curves"], args.out)
        print(f"\n✅ Graphiques régénérés dans {args.out.resolve()}")
        return

    if not args.data.exists():
        raise FileNotFoundError(f"{args.data} introuvable. Lancez step_A d'abord.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.data, "rb") as f:
        raw = pickle.load(f)

    dataset = _normalize_dataset(raw)

    # ── Validation du mapping sublevel → y_reg ───────────────────────────────
    # Lève KeyError si un sublevel du dataset n'est pas dans SUBLEVEL_TO_SCORE.
    assert_all_sublevels_known(dataset.values())
    # Affiche la distribution des cibles y après application du nouveau mapping.
    log_y_distribution(
        [float(Y9_TO_REG[int(v["y9"])]) for v in dataset.values()],
        label="y_reg (SUBLEVEL_TO_SCORE)",
    )

    print("=" * 60)
    print(" Step B — Causal GRU scorer + LOPO (MSELoss + Aug Expert×3)" if args.lopo_v2
          else " Step B — Causal GRU scorer + LOPO")
    print("=" * 60)
    if args.lopo_v2:
        print("[Loss] MSELoss standard")
        print("[Aug V2]  Expert×3 / Senior×1.5 / Student+Junior×1.0")
    aug_gran = "y4 (legacy)" if args.aug_by_y4 else (
        "y9 all" if args.aug_all_y9 else "y9 top-tier (PGY6/Fellow/Staff)"
    )
    aug_mode = "désactivée" if args.no_inline_aug else (
        f"DBA+jitter {aug_gran} → {args.aug_target}/groupe" if args.aug_target is not None
        else (
            "DBA+jitter V2 (ratios par classe y4)" if args.lopo_v2
            else f"DBA+jitter {aug_gran} → groupe majoritaire du fold"
        )
    )
    print(f"\n[Dataset] {len(dataset)} trials réels")
    print(f"[Augmentation] {aug_mode}")
    dist_y9 = Counter(int(v["y9"]) for v in dataset.values())
    dist = Counter(int(v.get("y4", Y9_TO_Y4[int(v["y9"])])) for v in dataset.values())
    print("  Distribution y9 :", {SUBLEVEL_NAMES[i][:6]: dist_y9[i] for i in range(9) if dist_y9[i]})
    for c in range(4):
        print(f"  {CLASS4_NAMES[c]:>7} (y4): {dist[c]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}  |  epochs={args.epochs}  |  max_folds={args.max_folds}")
    print(
        f"[Loss] ExpertAwareLoss  |  legacy sep/anchor/staff CLI ignorés en V2"
        if args.lopo_v2 else
        f"[Loss] sep={args.sep_weight}  margin={args.margin}  "
        f"anchor={args.anchor_weight}  top_tier={args.top_tier_weight}  "
        f"staff={args.staff_weight}  agg_pct={args.agg_percentile}"
    )

    preds, aux = run_lopo(
        dataset, device, epochs=args.epochs,
        max_folds=args.max_folds, patience=args.patience,
        batch_size=args.batch_size, mc_passes=args.mc_passes,
        aug_target=args.aug_target, no_inline_aug=args.no_inline_aug,
        aug_by_y9=not args.aug_by_y4,
        aug_top_tier_only=not args.aug_all_y9,
        sep_weight=args.sep_weight,
        margin=args.margin,
        anchor_weight=args.anchor_weight,
        top_tier_weight=args.top_tier_weight,
        staff_weight=args.staff_weight,
        staff_pos_weight=args.staff_pos_weight,
        agg_percentile=args.agg_percentile,
    )

    if not preds:
        print("Aucune prédiction — vérifiez le dataset.")
        return

    preds = enrich_preds_with_sublevel(preds, dataset=dataset)
    print_early_stopping_summary(aux.get("fold_train_info", []), args.epochs)
    print_metrics(preds)

    metrics_payload = compute_v2_metrics_with_baseline(preds) if args.lopo_v2 else None
    if metrics_payload:
        print("\n" + "=" * 60)
        print(" Comparaison baseline (9 juin) vs V2")
        print("=" * 60)
        for key, row in metrics_payload["comparison"].items():
            arrow = "↑" if row.get("improved") else ("↓" if row.get("improved") is False else " ")
            print(f"  {key:<22} baseline={row['baseline']:.3f}  v2={row['v2']:.3f}  "
                  f"Δ={row['delta']:+.3f} {arrow}")
        max_score = metrics_payload["v2"]["max_pred_score"]
        print(f"\n  Score max prédit (V2) : {max_score:+.3f}")

    pred_path = save_lopo_results(args.out, preds, aux, metrics=metrics_payload)
    print(f"\n[Sauvegarde] {pred_path.resolve()}")

    if aux.get("fold_train_info"):
        loss_path = args.out / "loss_components_per_fold.json"
        save_loss_components_per_fold(aux["fold_train_info"], loss_path)
        print(f"[Sauvegarde] {loss_path.resolve()}")

    if metrics_payload:
        metrics_path = args.out / "metrics_summary.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(_sanitize_for_json(metrics_payload), f, indent=2, ensure_ascii=False)
        print(f"[Sauvegarde] {metrics_path.resolve()}")

    if smoke and aux.get("fold_train_info"):
        fi0 = aux["fold_train_info"][0]
        checks = [
            ("internal_val_participants", "internal_val_participants" in fi0),
            ("loss_components", "loss_components" in fi0),
            ("loss_epochs", "loss_epochs" in fi0),
        ]
        print("\n[Smoke test checks]")
        for name, ok in checks:
            print(f"  {'✓' if ok else '✗'} {name}")
        if metrics_payload:
            ok_max = metrics_payload["v2"]["max_pred_score"] > 0.3
            print(f"  {'✓' if ok_max else '✗'} score max > +0.3 ({metrics_payload['v2']['max_pred_score']:+.3f})")

    print("\n[Graphiques]")
    plot_score_vs_time(aux["curves"], args.out / "score_vs_time_4class.png", n_trials=len(preds))
    plot_score_vs_time_sublevels(
        aux["curves"], args.out / "score_vs_time_9sublevels.png", n_trials=len(preds),
    )
    plot_score_vs_time_sublevels_grid(
        aux["curves"], args.out / "score_vs_time_9sublevels_grid.png",
    )
    plot_score_vs_time_per_sublevel(
        aux["curves"],
        args.out / "sublevels_time",
    )
    plot_confusion(preds, args.out / "confusion_matrix_4class.png")
    plot_confusion_y9(preds, args.out / "confusion_matrix_9sublevels.png")
    plot_senior_expert_focus(preds, args.out / "senior_expert_scores.png")
    plot_scatter_4class(preds, args.out / "scatter_4class.png")
    plot_scatter(preds, args.out / "scatter_y9.png")
    if args.lopo_v2:
        _run_granular_plots(preds, aux["curves"], args.out)
    _run_sublevel_analysis(preds, args.out)
    _run_senior_expert_diagnostic(preds, args.out / "diagnostic")
    print(f"\n✅ Résultats dans {args.out.resolve()}")


if __name__ == "__main__":
    main()
