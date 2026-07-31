"""
icems_v3_train.py
=================
Boucle d'entraînement LOPO nested pour ICEMS V3.

Usage :
    python src/icems_v3_train.py --smoke-test
    python src/icems_v3_train.py --full
    python src/icems_v3_train.py --full --generate-figures
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from icems_v3_augment import augment_fold_v3
from icems_v3_loss import HierarchicalSurgicalLoss
from icems_v3_model import ICEMS_V3

N_FEATURES = 6
VALID_COL = 5
N_FEATURES_RAW = 10

TRAIN_CROP_LEN = 800
INFER_MAX_LEN = 2500
TRAIN_DROPOUT = 0.3
MC_DROPOUT = 0.3
MC_PASSES = 30
N_INTERNAL_VAL = 4
NESTED_SEED = 42

Y9_TO_Y4 = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 3}
Y4_TO_REG = np.array([-1.0, -0.33, 0.33, 1.0], dtype=np.float32)
V3_Y9_TO_REG = np.array(
    [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32,
)
V3_SUBLEVEL_NAMES = [
    "Medical Student", "PGY1", "PGY2", "PGY3", "PGY4",
    "PGY5", "PGY6", "Fellow", "Neurosurgeon",
]
CLASS4_NAMES = ["Student", "Junior", "Senior", "Expert"]

BASELINE_METRICS = {
    "r_participant": 0.870,
    "expert_recall": 0.0,
    "senior_expert_distance": 1.602,
}


def extract_6_features(X: np.ndarray) -> np.ndarray:
    """
    Réduit le tenseur 10 canaux (3 instruments × 3 métriques + valid_ratio)
    vers les 6 features cinématiques V3.
    """
    if X.shape[1] == N_FEATURES:
        return X.astype(np.float32)
    vel = np.linalg.norm(X[:, [0, 3, 6]], axis=1).astype(np.float32)
    acc = np.linalg.norm(X[:, [1, 4, 7]], axis=1).astype(np.float32)
    jerk = np.linalg.norm(X[:, [2, 5, 8]], axis=1).astype(np.float32)
    spread = np.std(X[:, [0, 3, 6]], axis=1).astype(np.float32)
    axis_angle = jerk / (vel + 1e-6)
    valid_ratio = X[:, 9].astype(np.float32)
    return np.stack([vel, acc, jerk, spread, axis_angle, valid_ratio], axis=1)


def frame_valid_mask(X: np.ndarray, thresh: float = 0.0) -> np.ndarray:
    return (X[:, VALID_COL] > thresh).astype(np.float32)


def trial_y4(rec: dict) -> int:
    if "y4" in rec:
        return int(rec["y4"])
    return Y9_TO_Y4[int(rec["y9"])]


def trial_y9_reg(rec: dict) -> Tuple[int, float]:
    y9 = int(rec["y9"])
    y_reg = float(rec.get("y_reg", V3_Y9_TO_REG[y9]))
    return y9, y_reg


def sublevel_name(rec: dict, y9: Optional[int] = None) -> str:
    if y9 is None:
        y9 = int(rec.get("y9", 0))
    if 0 <= y9 < len(V3_SUBLEVEL_NAMES):
        return V3_SUBLEVEL_NAMES[y9]
    return V3_SUBLEVEL_NAMES[0]


def score_to_class(score: float) -> int:
    return int(np.argmin(np.abs(Y4_TO_REG - score)))


def set_dropout_rate(model: nn.Module, p: float) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = p


def _normalize_dataset(dataset) -> Dict:
    if isinstance(dataset, list):
        out: Dict = {}
        for entry in dataset:
            if entry.get("is_augmented", False):
                continue
            pid = str(entry["participant"])
            trial = str(entry.get("trial", entry.get("name", "0")))
            out[(pid, trial)] = _entry_to_rec(entry)
        return out
    return {
        k: v for k, v in dataset.items()
        if not str(k[0]).startswith("synth_") and not v.get("is_augmented", False)
    }


def _entry_to_rec(entry: dict) -> dict:
    X = extract_6_features(np.asarray(entry["X"], dtype=np.float32))
    y9 = int(entry.get("y9", entry.get("expertise_idx", 0)))
    y4 = int(entry.get("y4", Y9_TO_Y4[y9]))
    return {
        "X": X,
        "y9": y9,
        "y4": y4,
        "y_reg": float(entry.get("y_reg", V3_Y9_TO_REG[y9])),
        "level": entry.get("level", V3_SUBLEVEL_NAMES[y9]),
    }


def enrich_rec(rec: dict) -> dict:
    out = dict(rec)
    out["X"] = extract_6_features(np.asarray(rec["X"], dtype=np.float32))
    y9 = int(rec["y9"])
    out["y9"] = y9
    out["y4"] = trial_y4(rec)
    out["y_reg"] = float(rec.get("y_reg", V3_Y9_TO_REG[y9]))
    return out


class TrialDataset(Dataset):
    def __init__(self, items: List[Tuple], crop_len: int = TRAIN_CROP_LEN):
        self.items = items
        self.crop_len = crop_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        x, y_reg, y4, y9 = self.items[idx]
        if x.shape[0] > self.crop_len:
            start = np.random.randint(0, x.shape[0] - self.crop_len)
            x = x[start: start + self.crop_len]
        vm = frame_valid_mask(x)
        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(vm),
            float(y_reg),
            y4,
            y9,
        )


def collate_trials(batch):
    lengths = [b[0].shape[0] for b in batch]
    T_max = max(lengths)
    B = len(batch)
    xs = torch.zeros(B, T_max, N_FEATURES)
    masks = torch.zeros(B, T_max, dtype=torch.bool)
    y = torch.zeros(B)
    y4 = torch.zeros(B, dtype=torch.long)
    y9 = torch.zeros(B, dtype=torch.long)
    for i, (x, vm, yr, y4_i, y9_i) in enumerate(batch):
        L = x.shape[0]
        xs[i, :L] = x
        masks[i, :L] = vm.bool()
        y[i] = yr
        y4[i] = y4_i
        y9[i] = y9_i
    return xs, masks, y, y4, y9


def compute_norm_stats(trials: Dict) -> Tuple[np.ndarray, np.ndarray]:
    kin = np.concatenate(
        [rec["X"].reshape(-1, N_FEATURES) for rec in trials.values()],
        axis=0,
    )
    mean = kin.mean(axis=0).astype(np.float32)
    std = kin.std(axis=0).astype(np.float32) + 1e-6
    mean[VALID_COL] = 0.0
    std[VALID_COL] = 1.0
    return mean, std


def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = (X - mean) / std
    out[:, VALID_COL] = X[:, VALID_COL]
    return out.astype(np.float32)


def trials_to_items(trials: Dict, mean: np.ndarray, std: np.ndarray) -> List:
    items = []
    for rec in trials.values():
        if rec["X"].shape[0] < 2:
            continue
        x = apply_norm(rec["X"], mean, std)
        y9, y_reg = trial_y9_reg(rec)
        y4 = trial_y4(rec)
        items.append((x, y_reg, y4, y9))
    return items


def build_stratified_items(items: List) -> List:
    by_y9 = {c: [] for c in range(9)}
    for t in items:
        by_y9[t[3]].append(t)
    for c in range(9):
        np.random.shuffle(by_y9[c])
    stratified = []
    max_len = max(len(v) for v in by_y9.values()) if by_y9 else 0
    for i in range(max_len):
        for c in range(9):
            if i < len(by_y9[c]):
                stratified.append(by_y9[c][i])
    return stratified


def _participant_y4_map(dataset: Dict, pids: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for pid in pids:
        keys = [k for k in dataset if str(k[0]) == str(pid)]
        if keys:
            out[str(pid)] = trial_y4(dataset[keys[0]])
    return out


def select_stratified_val_participants(
    train_pool_pids: List[str],
    dataset: Dict,
    k: int = N_INTERNAL_VAL,
    seed: int = NESTED_SEED,
    fold_idx: int = 0,
) -> List[str]:
    pool = sorted({str(p) for p in train_pool_pids})
    if len(pool) < k + 1:
        raise ValueError(
            f"Fold {fold_idx}: nested LOPO requiert au moins {k + 1} participants "
            f"dans le pool train, seulement {len(pool)} disponibles."
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
        if candidates:
            chosen.append(str(rng.choice(candidates)))

    remaining = [p for p in pool if p not in chosen]
    n_needed = k - len(chosen)
    if n_needed > 0 and remaining:
        extra = rng.choice(remaining, size=min(n_needed, len(remaining)), replace=False)
        chosen.extend(str(p) for p in np.atleast_1d(extra))
    return sorted(chosen)


def _real_participants_of_trials(trials: Dict):
    return {
        str(k[0]) if isinstance(k, tuple) else str(k).split("_")[0]
        for k, v in trials.items()
        if not v.get("is_augmented", False) and not str(k).startswith("synth_")
    }


def _aug_source_participants(rec: dict):
    return {str(p) for p in rec.get("aug_source_participants", [])}


def filter_synth_by_excluded_sources(synth_dict: Dict, exclude_pids) -> Dict:
    exclude = {str(p) for p in exclude_pids}
    return {
        k: v for k, v in synth_dict.items()
        if not (_aug_source_participants(v) & exclude)
    }


def build_train_trials(train_fit_real: Dict, synth_dict: Dict, exclude_pids) -> Dict:
    synth_filtered = filter_synth_by_excluded_sources(synth_dict, exclude_pids)
    merged = {**train_fit_real, **synth_filtered}
    return {k: enrich_rec(v) if not v.get("is_augmented") else v for k, v in merged.items()}


def assert_no_leakage(train_trials, val_trials, test_trials, p_test, val_participants):
    val_set = {str(p) for p in val_participants}
    test_set = {str(p_test)}
    exclude = val_set | test_set

    train_p = _real_participants_of_trials(train_trials)
    val_p = _real_participants_of_trials(val_trials)
    test_p = _real_participants_of_trials(test_trials)

    assert test_p == test_set, f"FUITE: test {test_p} != {{{p_test}}}"
    assert val_p == val_set, f"FUITE: val {val_p} != {val_set}"
    assert train_p.isdisjoint(val_p)
    assert train_p.isdisjoint(test_p)
    assert val_p.isdisjoint(test_p)

    for rec in val_trials.values():
        assert not rec.get("is_augmented", False)
    for rec in test_trials.values():
        assert not rec.get("is_augmented", False)

    for k, rec in train_trials.items():
        if not rec.get("is_augmented", False):
            continue
        sources = _aug_source_participants(rec)
        assert sources, f"synthétique {k} sans aug_source_participants"
        assert not (sources & exclude), f"FUITE augmentation {k}"


def train_fold_v3(
    train_trials: Dict,
    val_trials: Dict,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[ICEMS_V3, dict]:
    mean, std = compute_norm_stats(train_trials)
    train_items = trials_to_items(train_trials, mean, std)
    val_items = trials_to_items(val_trials, mean, std)

    val_loader = DataLoader(
        TrialDataset(val_items),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_trials,
    )

    model = ICEMS_V3(n_features=N_FEATURES, dropout=TRAIN_DROPOUT).to(device)
    criterion = HierarchicalSurgicalLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val, best_state, wait = float("inf"), None, 0
    best_epoch, stopped_epoch = 0, epochs
    epoch_loss_log: List[dict] = []

    for ep in range(1, epochs + 1):
        model.train()
        set_dropout_rate(model, TRAIN_DROPOUT)
        loader = DataLoader(
            TrialDataset(build_stratified_items(train_items)),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_trials,
        )
        train_comps = {"mse": [], "focal": [], "fine": [], "ratio_mse": []}

        for xs, masks, y_reg, y4, y9 in loader:
            xs, masks = xs.to(device), masks.to(device)
            y_reg = y_reg.to(device)
            y4, y9 = y4.to(device), y9.to(device)

            _, score_agg, logits_4, logits_9, _ = model(xs, masks)
            loss, comps = criterion(score_agg, logits_4, logits_9, y_reg, y4, y9)

            opt.zero_grad()
            loss.backward()
            opt.step()
            for k in train_comps:
                train_comps[k].append(comps[k])

        model.eval()
        val_losses, val_comps = [], {"mse": [], "focal": [], "fine": [], "ratio_mse": []}
        with torch.no_grad():
            for xs, masks, y_reg, y4, y9 in val_loader:
                xs, masks = xs.to(device), masks.to(device)
                y_reg = y_reg.to(device)
                y4, y9 = y4.to(device), y9.to(device)
                _, score_agg, logits_4, logits_9, _ = model(xs, masks)
                vloss, vcomps = criterion(score_agg, logits_4, logits_9, y_reg, y4, y9)
                val_losses.append(vloss.item())
                for k in val_comps:
                    val_comps[k].append(vcomps[k])

        vloss = float(np.mean(val_losses)) if val_losses else float("inf")
        ep_log = {
            "epoch": ep,
            "train_mse": float(np.mean(train_comps["mse"])) if train_comps["mse"] else None,
            "train_focal": float(np.mean(train_comps["focal"])) if train_comps["focal"] else None,
            "train_fine": float(np.mean(train_comps["fine"])) if train_comps["fine"] else None,
            "train_ratio_mse": float(np.mean(train_comps["ratio_mse"])) if train_comps["ratio_mse"] else None,
            "val_mse": float(np.mean(val_comps["mse"])) if val_comps["mse"] else None,
            "val_focal": float(np.mean(val_comps["focal"])) if val_comps["focal"] else None,
            "val_fine": float(np.mean(val_comps["fine"])) if val_comps["fine"] else None,
            "val_ratio_mse": float(np.mean(val_comps["ratio_mse"])) if val_comps["ratio_mse"] else None,
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
        "model": "icems_v3",
    }
    if epoch_loss_log and best_epoch > 0:
        best_log = epoch_loss_log[best_epoch - 1]
        train_info["loss_components"] = {
            "mse": best_log.get("val_mse"),
            "focal": best_log.get("val_focal"),
            "fine": best_log.get("val_fine"),
            "ratio_mse": best_log.get("val_ratio_mse"),
        }
    return model, train_info


def _maybe_truncate(x: np.ndarray, max_len: int = INFER_MAX_LEN) -> np.ndarray:
    if x.shape[0] <= max_len:
        return x
    idx = np.linspace(0, x.shape[0] - 1, max_len, dtype=int)
    return x[idx]


@torch.no_grad()
def predict_trial_v3(
    model: ICEMS_V3,
    x: np.ndarray,
    device: torch.device,
    n_passes: int = MC_PASSES,
) -> Tuple[np.ndarray, float, int, np.ndarray]:
    x = _maybe_truncate(x)
    set_dropout_rate(model, MC_DROPOUT)
    model.train()

    xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
    vm = torch.from_numpy(frame_valid_mask(x)).unsqueeze(0).bool().to(device)

    scores_list, agg_list, logits4_list, hidden_list = [], [], [], []
    for _ in range(n_passes):
        scores, score_agg, logits_4, _, hidden = model(xt, vm)
        scores_list.append(scores.squeeze(0).cpu().numpy())
        agg_list.append(score_agg.item())
        logits4_list.append(logits_4.argmax(dim=-1).item())
        hidden_list.append(hidden.squeeze(0).cpu().numpy())

    scores_arr = np.stack(scores_list, axis=0)
    mean_t = scores_arr.mean(axis=0)
    trial_score = float(np.mean(agg_list))
    pred_class = int(np.round(np.mean(logits4_list)))
    hidden_mean = np.mean(hidden_list, axis=0)
    return mean_t, trial_score, pred_class, hidden_mean


def compute_metrics(preds_df: pd.DataFrame, r_per_fold: List[float]) -> Dict:
    if preds_df.empty:
        return {}

    y_true = preds_df["true_score"].to_numpy(dtype=float)
    y_pred = preds_df["pred_score"].to_numpy(dtype=float)
    r_trial, _ = pearsonr(y_true, y_pred)
    rho_trial, _ = spearmanr(y_true, y_pred)
    mae_trial = float(np.mean(np.abs(y_true - y_pred)))

    by_part = preds_df.groupby("participant", as_index=False).agg(
        true_score=("true_score", "mean"),
        pred_score=("pred_score", "mean"),
    )
    y_p_true = by_part["true_score"].to_numpy(dtype=float)
    y_p_pred = by_part["pred_score"].to_numpy(dtype=float)

    if len(by_part) >= 2:
        r_participant, _ = pearsonr(y_p_true, y_p_pred)
        rho_participant, _ = spearmanr(y_p_true, y_p_pred)
    else:
        r_participant = rho_participant = float("nan")

    expert_mask = preds_df["group"] == "Expert"
    expert_recall = float(
        (preds_df.loc[expert_mask, "pred_class"] == 3).mean()
    ) if expert_mask.any() else float("nan")

    r_valid = [r for r in r_per_fold if not np.isnan(r)]
    return {
        "r_trial": float(r_trial),
        "r_participant": float(r_participant),
        "r_per_fold": [float(r) for r in r_per_fold],
        "r_per_fold_mean": float(np.mean(r_valid)) if r_valid else float("nan"),
        "mae": mae_trial,
        "spearman_rho": float(rho_participant),
        "spearman_rho_trial": float(rho_trial),
        "expert_recall": expert_recall,
        "n_trials": int(len(preds_df)),
        "n_participants": int(len(by_part)),
        "n_folds": int(len(r_per_fold)),
    }


def run_lopo_v3(
    dataset: Dict,
    device: torch.device,
    epochs: int = 40,
    max_folds: Optional[int] = None,
    patience: int = 25,
    batch_size: int = 16,
    mc_passes: int = MC_PASSES,
    seed: int = NESTED_SEED,
    no_aug: bool = False,
    fast_aug: bool = False,
    verbose_split_debug: bool = False,
) -> Tuple[pd.DataFrame, Dict, List[dict], Dict]:
    dataset = {k: enrich_rec(v) for k, v in _normalize_dataset(dataset).items()}

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
    participant_embeddings: Dict[str, dict] = {}

    for fold_i, p_test in enumerate(pids):
        held_keys = set(by_pid[p_test])
        test_trials = {k: dataset[k] for k in held_keys}

        train_pool_real = {k: v for k, v in dataset.items() if k not in held_keys}
        train_pool_pids = sorted({k[0] for k in train_pool_real})

        val_participants = select_stratified_val_participants(
            train_pool_pids, dataset, k=N_INTERNAL_VAL, seed=seed, fold_idx=fold_i,
        )
        val_set = set(val_participants)
        train_fit_real = {k: v for k, v in train_pool_real.items() if k[0] not in val_set}
        val_trials = {k: v for k, v in train_pool_real.items() if k[0] in val_set}

        print(
            f"\n[V3 LOPO fold {fold_i + 1}/{len(pids)}] TEST={p_test} | "
            f"internal VAL={val_participants} | TRAIN fit={len(train_fit_real)} trials"
        )

        synth = {} if no_aug else augment_fold_v3(
            train_fit_real, fold_i, seed=seed, fast_mode=fast_aug,
        )
        if synth:
            print(f"  [aug V3] +{len(synth)} synthétiques (DBA+jitter+timewarp)")

        exclude = val_set | {str(p_test)}
        train_trials = build_train_trials(train_fit_real, synth, exclude)
        assert_no_leakage(train_trials, val_trials, test_trials, p_test, val_participants)

        if verbose_split_debug:
            n_aug = sum(1 for v in train_trials.values() if v.get("is_augmented"))
            print(f"  [split] train={len(train_trials)} (aug={n_aug}) val={len(val_trials)} test={len(test_trials)}")

        model, train_info = train_fold_v3(
            train_trials, val_trials, device,
            epochs=epochs, patience=patience, batch_size=batch_size,
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
        ratio = (train_info.get("loss_components") or {}).get("ratio_mse")
        if ratio is not None:
            print(f"  ratio_mse (best val) = {ratio:.3f} {'✅' if ratio > 0.80 else '⚠️'}")

        mean, std = model._norm_mean, model._norm_std
        fold_true, fold_pred = [], []
        fold_hiddens = []

        for key in held_keys:
            rec = dataset[key]
            x = apply_norm(rec["X"], mean, std)
            frame_scores, score, pred_cls_logits, hidden = predict_trial_v3(
                model, x, device, n_passes=mc_passes,
            )
            y9, y9_reg = trial_y9_reg(rec)
            y4 = trial_y4(rec)
            pred_cls = pred_cls_logits if not np.isnan(pred_cls_logits) else score_to_class(score)

            fold_true.append(y9_reg)
            fold_pred.append(score)
            fold_hiddens.append(hidden)

            pred_rows.append({
                "participant": str(p_test),
                "trial": str(key[1]),
                "true_score": float(y9_reg),
                "pred_score": float(score),
                "group": CLASS4_NAMES[y4],
                "pred_class": int(pred_cls),
                "is_augmented": False,
                "fold_idx": fold_i,
            })
            all_preds.append({
                "key": key,
                "y9": y9,
                "y_reg": y9_reg,
                "y4": y4,
                "score": score,
                "pred_class": int(pred_cls),
                "participant": p_test,
                "sublevel": sublevel_name(rec, y9=y9),
                "is_augmented": False,
                "fold_idx": fold_i,
            })
            curves.append({
                "time_norm": np.linspace(0, 1, len(frame_scores)),
                "scores": frame_scores,
                "y4": y4,
                "y9": y9,
                "sublevel": sublevel_name(rec, y9=y9),
                "key": key,
                "is_augmented": False,
            })

        if fold_hiddens:
            participant_embeddings[str(p_test)] = {
                "embedding": np.mean(fold_hiddens, axis=0),
                "y4": trial_y4(dataset[next(iter(held_keys))]),
                "sublevel": sublevel_name(dataset[next(iter(held_keys))]),
            }

        r_fold = float("nan")
        if len(fold_true) >= 2:
            ft, fp = np.asarray(fold_true, float), np.asarray(fold_pred, float)
            if ft.std() > 1e-8 and fp.std() > 1e-8:
                r_fold, _ = pearsonr(ft, fp)
                r_fold = float(r_fold)
        r_per_fold.append(r_fold)

    preds_df = pd.DataFrame(pred_rows)
    metrics = compute_metrics(preds_df, r_per_fold)

    emb_pids = sorted(participant_embeddings.keys())
    embeddings = np.stack([participant_embeddings[p]["embedding"] for p in emb_pids])
    labels_4 = [participant_embeddings[p]["y4"] for p in emb_pids]
    sublevels = [participant_embeddings[p]["sublevel"] for p in emb_pids]

    if len(embeddings) >= 2:
        senior = embeddings[np.array(labels_4) == 2]
        expert = embeddings[np.array(labels_4) == 3]
        if len(senior) and len(expert):
            dist = float(np.linalg.norm(senior.mean(0) - expert.mean(0)))
            metrics["senior_expert_distance"] = dist

    aux = {
        "curves": curves,
        "fold_train_info": fold_train_info,
        "embeddings": embeddings,
        "labels_4": labels_4,
        "sublevels": sublevels,
        "participant_ids": emb_pids,
    }
    return preds_df, metrics, all_preds, aux


def save_v3_results(out_dir: Path, preds_df, metrics, all_preds, aux) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "lopo_predictions.pkl"
    metrics_path = out_dir / "metrics_summary.json"

    payload = {
        "preds": all_preds,
        "curves": aux["curves"],
        "embeddings": aux["embeddings"],
        "labels_4": aux["labels_4"],
        "sublevels": aux["sublevels"],
        "participant_ids": aux.get("participant_ids", []),
        "metrics": metrics,
        "preds_df": preds_df,
    }
    with open(pred_path, "wb") as f:
        pickle.dump(payload, f)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    loss_path = out_dir / "loss_components_per_fold.json"
    with open(loss_path, "w", encoding="utf-8") as f:
        json.dump(aux.get("fold_train_info", []), f, indent=2, ensure_ascii=False, default=str)

    return pred_path, metrics_path


def print_metrics(metrics: Dict) -> None:
    if not metrics:
        print("Aucune métrique.")
        return
    print("\n" + "=" * 60)
    print(" Métriques ICEMS V3 LOPO (TEST only)")
    print("=" * 60)
    print(f"  r_trial         : {metrics.get('r_trial', float('nan')):.3f}")
    print(f"  r_participant   : {metrics.get('r_participant', float('nan')):.3f}")
    print(f"  spearman_rho    : {metrics.get('spearman_rho', float('nan')):.3f}")
    print(f"  MAE             : {metrics.get('mae', float('nan')):.3f}")
    print(f"  expert_recall   : {metrics.get('expert_recall', float('nan')):.1%}")
    if "senior_expert_distance" in metrics:
        print(f"  dist S/E        : {metrics['senior_expert_distance']:.3f}")
    print(f"  n_trials        : {metrics.get('n_trials', 0)}")
    print(f"  n_folds         : {metrics.get('n_folds', 0)}")
    print("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser(description="Entraînement LOPO ICEMS V3")
    ap.add_argument("--data", type=Path, default=Path("data/continuous_per_trial.pkl"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--mc-passes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=NESTED_SEED)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--generate-figures", action="store_true")
    args = ap.parse_args()

    smoke = args.smoke_test or not args.full
    epochs = args.epochs if args.epochs is not None else (3 if smoke else 40)
    max_folds = args.max_folds if args.max_folds is not None else (2 if smoke else None)
    mc_passes = args.mc_passes if args.mc_passes is not None else (3 if smoke else MC_PASSES)

    out_dir = args.out or Path("results/lopo_v3_run1") / date.today().isoformat()

    if not args.data.exists():
        raise FileNotFoundError(
            f"{args.data} introuvable. Lancez : python src/build_continuous_dataset.py"
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.data, "rb") as f:
        raw = pickle.load(f)

    dataset = _normalize_dataset(raw)
    print("=" * 60)
    print(" ICEMS V3 — LOPO nested + HierarchicalSurgicalLoss")
    print("=" * 60)
    print(f"  Trials réels  : {len(dataset)}")
    print(f"  Participants  : {len({k[0] for k in dataset})}")
    print(f"  Mode          : {'SMOKE TEST' if smoke else 'RUN COMPLET'}")
    print(f"  Epochs        : {epochs}")
    print(f"  Sortie        : {out_dir.resolve()}")
    print(f"  Device        : {device}")
    print("=" * 60)

    preds_df, metrics, all_preds, aux = run_lopo_v3(
        dataset,
        device,
        epochs=epochs,
        max_folds=max_folds,
        patience=args.patience,
        batch_size=args.batch_size,
        mc_passes=mc_passes,
        seed=args.seed,
        no_aug=args.no_aug,
        fast_aug=smoke,
        verbose_split_debug=smoke,
    )

    print_metrics(metrics)

    r_part = metrics.get("r_participant", float("nan"))
    if not np.isnan(r_part):
        baseline = BASELINE_METRICS["r_participant"]
        if r_part >= baseline:
            print(f"\n✅ r_participant={r_part:.3f} ≥ baseline ({baseline:.3f})")
        else:
            print(f"\n❌ r_participant={r_part:.3f} < baseline ({baseline:.3f})")

    pred_path, metrics_path = save_v3_results(out_dir, preds_df, metrics, all_preds, aux)
    print(f"\n[Sauvegarde] {pred_path.resolve()}")
    print(f"             {metrics_path.resolve()}")

    if args.generate_figures:
        from icems_v3_visualize import generate_all_figures
        fig_dir = out_dir / "figures_granulaires"
        generate_all_figures(
            preds=all_preds,
            curves_data=aux["curves"],
            embeddings=aux["embeddings"],
            labels_4=aux["labels_4"],
            sublevels=aux["sublevels"],
            out_dir=fig_dir,
        )

    if smoke:
        print("\n✅ Smoke test V3 terminé.")
        print("   Prochaine étape : python src/icems_v3_train.py --full --generate-figures")


if __name__ == "__main__":
    main()
