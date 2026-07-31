"""
step_B_classification.py
========================
Classification 4 niveaux avec score continu ordonné et évaluation LOPO (47 participants).

Architecture : CausalGRUScorer + loss MSE agrégée + pénalité ordinale + MC Dropout.

Usage (depuis ICEMS-main) :
    python src/step_B_classification.py --data data/continuous_per_trial.pkl --out results/level1 --epochs 40
    python src/step_B_classification.py --data data/continuous_per_trial.pkl --aug-target 200 --out results/run_scaled
    python src/step_B_classification.py --data data/continuous_per_trial.pkl --epochs 5 --max_folds 3
    python src/step_B_classification.py --plot-only --out results/level1_continuous
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from tslearn.barycenters import dtw_barycenter_averaging

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# ── Mapping 4 classes (aligné step_A / barème utilisateur) ───────────────────
Y9_TO_Y4 = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 3}
Y4_TO_REG = np.array([-1.0, -0.33, 0.33, 1.0], dtype=np.float32)
CLASS4_NAMES = ["Student", "Junior", "Senior", "Expert"]
CLASS4_COLORS = {0: "#d62728", 1: "#ff7f0e", 2: "#1f77b4", 3: "#2ca02c"}

N_FEATURES = 10
VALID_COL = 9
MC_PASSES = 30
TRAIN_CROP_LEN = 800
INFER_MAX_LEN = 2500

TRAIN_DROPOUT = 0.15
MC_DROPOUT = 0.3
LOPO_PREDICTIONS_FILE = "lopo_predictions.pkl"

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


def frame_valid_mask(X: np.ndarray, thresh: float = 0.0) -> np.ndarray:
    """1 si frame présente (tracking valide), 0 sinon."""
    return (X[:, VALID_COL] > thresh).astype(np.float32)


def score_to_class(score: float) -> int:
    return int(np.argmin(np.abs(Y4_TO_REG - score)))


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

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, valid_mask.unsqueeze(-1)], dim=-1)
        z = torch.nn.functional.gelu(self.proj(inp))
        out, _ = self.gru(z)
        return self.head(out).squeeze(-1)


def set_dropout_rate(model: CausalGRUScorer, p: float) -> None:
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
    y4_classes: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """
    Force la séparation entre classes :
    Expert doit être > Senior > Junior > Student
    avec une marge minimale paramétrable entre chaque niveau.
    """
    loss = torch.tensor(0.0, device=agg_scores.device)
    count = 0

    for i in range(len(agg_scores)):
        for j in range(len(agg_scores)):
            ci = int(y4_classes[i])
            cj = int(y4_classes[j])
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


def anchor_loss(agg_scores: torch.Tensor, y4_classes: torch.Tensor) -> torch.Tensor:
    """
    Student (classe 0) → pénalise si prédit > -0.8
    Expert  (classe 3) → pénalise si prédit < +0.8
    """
    n = len(agg_scores)
    if n == 0:
        return torch.tensor(0.0, device=agg_scores.device)

    loss = torch.tensor(0.0, device=agg_scores.device)
    for i, score in enumerate(agg_scores):
        if int(y4_classes[i]) == 0:
            loss = loss + nn.functional.relu(score + 0.8)
        elif int(y4_classes[i]) == 3:
            loss = loss + nn.functional.relu(0.8 - score)
    return loss / n


def ordinal_loss(
    agg: torch.Tensor,
    y_score: torch.Tensor,
    y4_classes: torch.Tensor,
    sep_weight: float = 0.5,
    margin: float = 0.3,
    anchor_weight: float = 0.0,
) -> torch.Tensor:
    """MSE + pénalité de séparation inter-classes + ancrage extrêmes."""
    mse = nn.functional.mse_loss(agg, y_score)
    sep = separation_loss(agg, y4_classes, margin=margin)
    anchor = anchor_loss(agg, y4_classes)
    return mse + sep_weight * sep + anchor_weight * anchor


class TrialDataset(Dataset):
    def __init__(self, items: List[Tuple[np.ndarray, float, int]], crop_len: int = TRAIN_CROP_LEN):
        self.items = items
        self.crop_len = crop_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        x, y_reg, y4 = self.items[idx]
        if x.shape[0] > self.crop_len:
            start = np.random.randint(0, x.shape[0] - self.crop_len)
            x = x[start : start + self.crop_len]
        vm = frame_valid_mask(x)
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(vm), float(y_reg), y4


def collate_trials(batch):
    lengths = [b[0].shape[0] for b in batch]
    T_max = max(lengths)
    B = len(batch)
    xs = torch.zeros(B, T_max, N_FEATURES)
    masks = torch.zeros(B, T_max)
    y = torch.zeros(B)
    y4 = torch.zeros(B, dtype=torch.long)
    for i, (x, vm, yr, y4_i) in enumerate(batch):
        L = x.shape[0]
        xs[i, :L] = x
        masks[i, :L] = vm
        y[i] = yr
        y4[i] = y4_i
    return xs, masks, y, y4


def compute_norm_stats(trials: Dict) -> Tuple[np.ndarray, np.ndarray]:
    kin = np.concatenate(
        [rec["X"][:, :VALID_COL].reshape(-1, VALID_COL) for rec in trials.values()],
        axis=0,
    )
    mean = kin.mean(axis=0)
    std = kin.std(axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = X.copy()
    out[:, :VALID_COL] = (out[:, :VALID_COL] - mean) / std
    return out


def build_stratified_items(items: List, shuffle_classes: bool = True) -> List:
    """Intercale les trials par classe pour des batches équilibrés."""
    by_class = {c: [] for c in range(4)}
    for t in items:
        by_class[t[2]].append(t)

    if shuffle_classes:
        for c in range(4):
            np.random.shuffle(by_class[c])

    stratified = []
    max_len = max(len(v) for v in by_class.values()) if by_class else 0
    for i in range(max_len):
        for c in range(4):
            if i < len(by_class[c]):
                stratified.append(by_class[c][i])
    return stratified


def trials_to_items(trials: Dict, mean: np.ndarray, std: np.ndarray) -> List:
    items = []
    for rec in trials.values():
        if rec["X"].shape[0] < 2:
            continue
        x = apply_norm(rec["X"], mean, std)
        y4, y_reg = trial_y4_reg(rec)
        items.append((x, y_reg, y4))
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
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    x = _maybe_truncate(x)
    """MC Dropout : model.train() pour garder dropout actif."""
    set_dropout_rate(model, MC_DROPOUT)
    model.train()
    xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
    vm = torch.from_numpy(frame_valid_mask(x)).unsqueeze(0).to(device)

    scores_list = []
    for _ in range(n_passes):
        s = model(xt, vm).squeeze(0).cpu().numpy()
        scores_list.append(s)
    scores_arr = np.stack(scores_list, axis=0)
    mean_t = scores_arr.mean(axis=0)
    std_t = scores_arr.std(axis=0)

    valid = frame_valid_mask(x) > 0
    if valid.any():
        trial_score = float(np.median(mean_t[valid]))
        trial_std = float(np.median(std_t[valid]))
    else:
        trial_score = float(np.median(mean_t))
        trial_std = float(np.median(std_t))
    return mean_t, std_t, trial_score, trial_std


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
) -> Tuple[CausalGRUScorer, dict]:
    mean, std = compute_norm_stats(train_trials)
    train_items = trials_to_items(train_trials, mean, std)
    val_items = trials_to_items(val_trials, mean, std)

    val_loader = DataLoader(
        TrialDataset(val_items),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_trials,
    )

    model = CausalGRUScorer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, wait = float("inf"), None, 0
    best_epoch, stopped_epoch = 0, epochs

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
        for xs, masks, y, y4_classes in loader:
            xs = xs.to(device)
            masks = masks.to(device)
            y = y.to(device)
            y4_classes = y4_classes.to(device)
            scores = model(xs, masks)
            agg = aggregate_score(scores, masks)
            loss = ordinal_loss(
                agg, y, y4_classes,
                sep_weight=sep_weight,
                margin=margin,
                anchor_weight=anchor_weight,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xs, masks, y, y4_val in val_loader:
                xs, masks, y = xs.to(device), masks.to(device), y.to(device)
                y4_val = y4_val.to(device)
                scores = model(xs, masks)
                agg = aggregate_score(scores, masks)
                val_losses.append(
                    ordinal_loss(
                        agg, y, y4_val,
                        sep_weight=sep_weight,
                        margin=margin,
                        anchor_weight=anchor_weight,
                    ).item()
                )
        vloss = float(np.mean(val_losses)) if val_losses else float("inf")

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
    }
    return model, train_info


def participants_of(dataset: Dict) -> List[str]:
    return sorted({k[0] for k in dataset.keys() if not str(k[0]).startswith("synth_")})


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
        "level": entry.get("level", CLASS4_NAMES[y4]),
        "T": x.shape[0],
        "fs": float(entry.get("fs", 10.0)),
        "is_augmented": bool(entry.get("is_augmented", False)),
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


def _augment_train_fold_inline(
    train_real: Dict,
    fold_i: int,
    n_parents: int = DBA_N_PARENTS,
    alpha: float = DBA_ALPHA,
    seed: int = 42,
    aug_target: Optional[int] = None,
) -> Dict:
    """DBA + jitter sur le train du fold uniquement.

    Si ``aug_target`` est None, équilibre vers la classe majoritaire du fold (Run 5).
    Sinon, génère des synthétiques jusqu'à ``aug_target`` trials par classe.
    """
    rng = np.random.default_rng(seed + fold_i * 1000)
    entries = [_rec_to_entry(k, v) for k, v in train_real.items()]
    by_class: Dict[str, List[dict]] = defaultdict(list)
    for entry in entries:
        by_class[str(entry["expertise"])].append(entry)

    counts = {c: len(by_class.get(c, [])) for c in CLASS4_NAMES}
    if aug_target is not None:
        target = aug_target
    else:
        target = max(counts.values()) if counts else 0
    synth_dict: Dict = {}
    synth_i = 0

    for class_name in CLASS4_NAMES:
        pool = by_class.get(class_name, [])
        n_needed = target - len(pool)
        if n_needed <= 0:
            continue
        if len(pool) < n_parents:
            print(
                f"  [aug] {class_name}: pool={len(pool)} < n_parents={n_parents}, "
                "augmentation ignorée pour cette classe"
            )
            continue

        for _ in range(n_needed):
            parents = list(rng.choice(pool, size=n_parents, replace=False))
            parent_arrays = [_data_to_array(p["data"]) for p in parents]
            parent_feats = [_extract_features(p["data"]) for p in parents]
            bary_feats = _run_dba_on_group(parent_feats)
            full = _rebuild_data_matrix(parent_arrays[0], bary_feats)
            jittered = _apply_jitter(full, alpha, rng)

            template = parents[0]
            pid = f"synth_{fold_i:03d}_{synth_i:03d}"
            synth_i += 1
            trial = f"dba_jitter_{class_name.lower()}_{synth_i}"
            entry = {
                "name": f"{pid}_{trial}",
                "participant": pid,
                "trial": trial,
                "expertise": class_name,
                "expertise_idx": int(template["expertise_idx"]),
                "level_idx": int(template.get("level_idx", 0)),
                "level": template.get("level", CLASS4_NAMES[int(template["expertise_idx"])]),
                "is_augmented": True,
                "aug_type": "dba+jitter",
                "data": jittered.tolist(),
            }
            key = (pid, trial)
            synth_dict[key] = _entry_to_train_rec(entry)

    if synth_dict:
        mode = f"fixe={target}" if aug_target is not None else f"majoritaire={target}"
        print(f"  [aug] +{len(synth_dict)} synthétiques (cible/classe, {mode})")
    return synth_dict


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
    sep_weight: float = 0.5,
    margin: float = 0.3,
    anchor_weight: float = 0.0,
) -> Tuple[List[dict], Dict]:
    """LOPO sur participants réels ; DBA+jitter générés inline sur le train de chaque fold."""
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
        print(f"\n[LOPO fold {fold_i + 1}/{len(pids)}] participant tenu out : {held_pid}")
        val_keys = set(by_pid[held_pid])
        train_real = {k: dataset[k] for k in dataset if k not in val_keys}
        if no_inline_aug:
            synth = {}
        else:
            synth = _augment_train_fold_inline(
                train_real, fold_i=fold_i, aug_target=aug_target,
            )
        train = {**train_real, **synth}
        val = {k: dataset[k] for k in val_keys}

        model, train_info = train_fold(
            train, val, device, epochs=epochs,
            lr=1e-3, weight_decay=1e-4, patience=patience,
            batch_size=batch_size,
            sep_weight=sep_weight,
            margin=margin,
            anchor_weight=anchor_weight,
        )
        train_info["participant"] = held_pid
        train_info["fold"] = fold_i + 1
        fold_train_info.append(train_info)

        if train_info["early_stopped"]:
            print(
                f"  Early stopping à l'epoch {train_info['stopped_epoch']}/{epochs} "
                f"(meilleur modèle : epoch {train_info['best_epoch']}, "
                f"val_loss={train_info['best_val_loss']:.5f})"
            )
        else:
            print(
                f"  Entraînement complet jusqu'à l'epoch {epochs}/{epochs} "
                f"(meilleur modèle : epoch {train_info['best_epoch']}, "
                f"val_loss={train_info['best_val_loss']:.5f})"
            )
        mean, std = model._norm_mean, model._norm_std

        for key in val_keys:
            rec = dataset[key]
            x = apply_norm(rec["X"], mean, std)
            mean_t, std_t, score, unc = predict_trial_mc(model, x, device, n_passes=mc_passes)
            y4, y_reg = trial_y4_reg(rec)
            pred_cls = score_to_class(score)
            all_preds.append({
                "key": key,
                "y_reg": y_reg,
                "y4": y4,
                "score": score,
                "uncertainty": unc,
                "pred_class": pred_cls,
                "participant": held_pid,
            })
            T_plot = len(mean_t)
            t_norm = np.linspace(0, 1, T_plot)
            curves.append({
                "t_norm": t_norm,
                "mean": mean_t,
                "std": std_t,
                "y4": y4,
                "key": key,
            })

    return all_preds, {"curves": curves, "fold_train_info": fold_train_info}


def print_early_stopping_summary(fold_train_info: List[dict], max_epochs: int) -> None:
    if not fold_train_info:
        return
    print("\n" + "=" * 60)
    print(" Early stopping par fold LOPO")
    print("=" * 60)
    print(f"  {'Fold':>4}  {'Participant':<12}  {'Stop epoch':>10}  {'Best epoch':>10}  {'Val loss':>10}")
    print("  " + "-" * 54)
    for info in fold_train_info:
        stop_label = (
            f"{info['stopped_epoch']}/{max_epochs}"
            if info["early_stopped"]
            else f"{max_epochs}/{max_epochs}*"
        )
        print(
            f"  {info['fold']:>4}  {info['participant']:<12}  {stop_label:>10}  "
            f"{info['best_epoch']:>10}  {info['best_val_loss']:>10.5f}"
        )
    n_early = sum(1 for i in fold_train_info if i["early_stopped"])
    avg_stop = np.mean([i["stopped_epoch"] for i in fold_train_info])
    print(f"\n  * pas d'early stopping (epochs max atteintes)")
    print(f"  Folds avec early stopping : {n_early}/{len(fold_train_info)}")
    print(f"  Epoch moyenne d'arrêt     : {avg_stop:.1f}/{max_epochs}")


def save_lopo_results(out_dir: Path, preds: List[dict], aux: Dict) -> Path:
    """Sauvegarde prédictions + courbes temporelles pour régénération sans LOPO."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / LOPO_PREDICTIONS_FILE
    payload = {
        "preds": preds,
        "curves": aux["curves"],
        "fold_train_info": aux.get("fold_train_info", []),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_lopo_results(path: Path) -> Tuple[List[dict], Dict]:
    """Charge prédictions LOPO sauvegardées."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    aux = {
        "curves": payload["curves"],
        "fold_train_info": payload.get("fold_train_info", []),
    }
    return payload["preds"], aux


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
    y_true = np.array([p["y_reg"] for p in preds])
    y_pred = np.array([p["score"] for p in preds])
    y4 = np.array([p["y4"] for p in preds])
    pred_cls = np.array([p["pred_class"] for p in preds])

    pr, pp = pearsonr(y_true, y_pred)
    sr, sp = spearmanr(y_true, y_pred)
    acc = float((pred_cls == y4).mean())

    print("\n" + "=" * 60)
    print(" Métriques LOPO (participants tenus out)")
    print("=" * 60)
    print(f"  Pearson  r = {pr:+.4f}  (p = {pp:.4e})  n = {len(preds)}")
    print(f"  Spearman r = {sr:+.4f}  (p = {sp:.4e})")
    print(f"  Accuracy 4 classes = {acc * 100:.1f}%  ({int((pred_cls == y4).sum())}/{len(preds)})")


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
    ax.set_title(f"Matrice de confusion 4×4 — accuracy = {acc * 100:.1f}%")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  → {out_path}")


def plot_scatter(preds: List[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.array([p["y_reg"] for p in preds])
    y_pred = np.array([p["score"] for p in preds])
    y4 = np.array([p["y4"] for p in preds])
    pr, _ = pearsonr(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 7))
    for c in range(4):
        m = y4 == c
        ax.scatter(y_true[m], y_pred[m], c=CLASS4_COLORS[c], label=CLASS4_NAMES[c], alpha=0.7, s=40)
    lims = [-1.1, 1.1]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Score réel (ordinal)")
    ax.set_ylabel("Score prédit")
    ax.set_title(f"Score prédit vs réel — Pearson r = {pr:+.3f}")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Step B — classification LOPO + score continu.")
    ap.add_argument("--data", type=Path, default=Path("data/augmented_trials.pkl"))
    ap.add_argument("--out", type=Path, default=Path("results/run7_loss_fix"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--max_folds", type=int, default=None,
                    help="Limite le nombre de folds LOPO (test rapide).")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--sep-weight", type=float, default=0.5,
                    help="Poids de la pénalité de séparation inter-classes.")
    ap.add_argument("--margin", type=float, default=0.3,
                    help="Marge minimale entre classes dans separation_loss.")
    ap.add_argument("--anchor-weight", type=float, default=0.0,
                    help="Poids de l'ancrage Student→-1 et Expert→+1.")
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

    args.out.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        pred_path = resolve_lopo_predictions(args.out, args.predictions)
        preds, aux = load_lopo_results(pred_path)
        print("=" * 60)
        print(" Step B — plot-only (score vs temps)")
        print("=" * 60)
        print(f"\n[Chargement] {pred_path.resolve()}")
        print(f"  {len(preds)} prédictions, {len(aux['curves'])} courbes")
        print("\n[Graphique]")
        plot_score_vs_time(
            aux["curves"],
            args.out / "score_vs_time.png",
            n_trials=len(preds),
        )
        print(f"\n✅ Graphique régénéré dans {args.out.resolve()}")
        return

    if not args.data.exists():
        raise FileNotFoundError(f"{args.data} introuvable. Lancez step_A d'abord.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.data, "rb") as f:
        raw = pickle.load(f)

    dataset = _normalize_dataset(raw)
    print("=" * 60)
    print(" Step B — Causal GRU scorer + LOPO")
    print("=" * 60)
    aug_mode = "désactivée" if args.no_inline_aug else (
        f"DBA+jitter → {args.aug_target}/classe" if args.aug_target is not None
        else "DBA+jitter → classe majoritaire du fold (Run 5)"
    )
    print(f"\n[Dataset] {len(dataset)} trials réels")
    print(f"[Augmentation] {aug_mode}")
    dist = Counter(int(v.get("y4", Y9_TO_Y4[int(v["y9"])])) for v in dataset.values())
    for c in range(4):
        print(f"  {CLASS4_NAMES[c]:>7}: {dist[c]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}  |  epochs={args.epochs}  |  max_folds={args.max_folds}")
    print(
        f"[Loss] sep_weight={args.sep_weight}  margin={args.margin}  "
        f"anchor_weight={args.anchor_weight}"
    )

    preds, aux = run_lopo(
        dataset, device, epochs=args.epochs,
        max_folds=args.max_folds, patience=args.patience,
        batch_size=args.batch_size, mc_passes=args.mc_passes,
        aug_target=args.aug_target, no_inline_aug=args.no_inline_aug,
        sep_weight=args.sep_weight,
        margin=args.margin,
        anchor_weight=args.anchor_weight,
    )

    if not preds:
        print("Aucune prédiction — vérifiez le dataset.")
        return

    print_early_stopping_summary(aux.get("fold_train_info", []), args.epochs)
    print_metrics(preds)

    pred_path = save_lopo_results(args.out, preds, aux)
    print(f"\n[Sauvegarde] {pred_path.resolve()}")

    print("\n[Graphiques]")
    plot_score_vs_time(aux["curves"], args.out / "score_vs_time.png", n_trials=len(preds))
    plot_confusion(preds, args.out / "confusion_matrix.png")
    plot_scatter(preds, args.out / "scatter.png")
    print(f"\n✅ Résultats dans {args.out.resolve()}")


if __name__ == "__main__":
    main()
