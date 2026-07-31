"""
train_clf_multitask.py
======================
Entrainement multi-tache (y4 + y9) avec validation LOPO.

Modele 1 (y4) : pas d'augmentation sur les echantillons reels.
Modele 2 (y9) : augmentation DBA/Jitter/TimeWarp dans le fold train uniquement.

Usage :
    python src/train_clf_multitask.py --smoke-test
    python src/train_clf_multitask.py --full
    python src/train_clf_multitask.py --full --epochs 40 --alpha 0.4
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from build_clf_datasets import EXPERTISE_TO_Y4, _build_participant_expertise_map
from clf_model2_augment import Y9_NAMES, augment_model2_fold
from models.clf_multitask_evicems import MultiTaskEVICEMS

ROOT = _SRC_DIR.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "clf_multitask"

PATH_PKL = DATA_DIR / "continuous_per_trial.pkl"
PATH_JSON = DATA_DIR / "filtered_data.json"

N_FEATURES = 10
VALID_COL = 9
SEQ_LEN = 800
N_CLASSES_Y4 = 4
N_CLASSES_Y9 = 9
Y4_NAMES = ["Student", "Junior", "Senior", "Expert"]
Y9_TO_Y4 = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 3}
NESTED_SEED = 42
N_INTERNAL_VAL = 4


def _log(msg: str) -> None:
    """Affichage immediat (visible avec tail -f sur Narval)."""
    print(msg, flush=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_real_dataset() -> Dict:
    with open(PATH_PKL, "rb") as f:
        pkl = pickle.load(f)
    pid_to_expertise = _build_participant_expertise_map(PATH_JSON)

    dataset: Dict = {}
    for (pid, tid), trial in pkl.items():
        expertise = pid_to_expertise.get(str(pid))
        if expertise not in EXPERTISE_TO_Y4:
            continue
        y4 = EXPERTISE_TO_Y4[expertise]
        y9 = int(trial["y9"])
        dataset[(pid, tid)] = {
            "X": np.asarray(trial["X"], dtype=np.float32),
            "y4": y4,
            "y9": y9,
            "y_reg": float(trial["y_reg"]),
            "level": trial["level"],
            "expertise": expertise,
            "is_augmented": False,
        }
    return dataset


def resample_sequence(x: np.ndarray, target_len: int) -> np.ndarray:
    t, f = x.shape
    if t == target_len:
        return x
    if t <= 1:
        out = np.zeros((target_len, f), dtype=np.float32)
        out[: min(t, target_len)] = x[: min(t, target_len)]
        return out
    idx = np.linspace(0, t - 1, target_len).astype(int)
    return x[idx].astype(np.float32)


def pad_or_crop(x: np.ndarray, seq_len: int, random_crop: bool) -> np.ndarray:
    t = x.shape[0]
    if t >= seq_len:
        if random_crop and t > seq_len:
            start = np.random.randint(0, t - seq_len)
            return x[start : start + seq_len]
        return resample_sequence(x, seq_len)
    out = np.zeros((seq_len, x.shape[1]), dtype=np.float32)
    out[:t] = x
    return out


def compute_norm_stats(trials: Dict) -> Tuple[np.ndarray, np.ndarray]:
    kin = np.concatenate([rec["X"].reshape(-1, N_FEATURES) for rec in trials.values()], axis=0)
    mean = kin.mean(axis=0).astype(np.float32)
    std = kin.std(axis=0).astype(np.float32) + 1e-6
    mean[VALID_COL] = 0.0
    std[VALID_COL] = 1.0
    return mean, std


def apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = (x - mean) / std
    out[:, VALID_COL] = x[:, VALID_COL]
    return out.astype(np.float32)


class MultiTaskDataset(Dataset):
    def __init__(self, items: List[dict], random_crop: bool = True):
        self.items = items
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        x = pad_or_crop(rec["x"], SEQ_LEN, random_crop=self.random_crop)
        return (
            torch.from_numpy(x),
            torch.tensor(rec["y4"], dtype=torch.long),
            torch.tensor(rec["y9"], dtype=torch.long),
            torch.tensor(rec["is_augmented"], dtype=torch.bool),
        )


def collate_batch(batch):
    xs, y4, y9, aug = zip(*batch)
    return (
        torch.stack(xs, dim=0),
        torch.stack(y4, dim=0),
        torch.stack(y9, dim=0),
        torch.stack(aug, dim=0),
    )


def trials_to_items(trials: Dict, mean: np.ndarray, std: np.ndarray) -> List[dict]:
    items = []
    for rec in trials.values():
        if rec["X"].shape[0] < 2:
            continue
        items.append({
            "x": apply_norm(rec["X"], mean, std),
            "y4": int(rec["y4"]),
            "y9": int(rec["y9"]),
            "is_augmented": bool(rec.get("is_augmented", False)),
        })
    return items


def build_stratified_items(items: List[dict]) -> List[dict]:
    by_y9 = {c: [] for c in range(N_CLASSES_Y9)}
    for item in items:
        by_y9[item["y9"]].append(item)
    for c in range(N_CLASSES_Y9):
        np.random.shuffle(by_y9[c])
    out = []
    max_len = max((len(v) for v in by_y9.values()), default=0)
    for i in range(max_len):
        for c in range(N_CLASSES_Y9):
            if i < len(by_y9[c]):
                out.append(by_y9[c][i])
    return out


def _participant_y4_map(dataset: Dict, pids: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for pid in pids:
        keys = [k for k in dataset if str(k[0]) == str(pid)]
        if keys:
            out[str(pid)] = int(dataset[keys[0]]["y4"])
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
        raise ValueError(f"Fold {fold_idx}: au moins {k + 1} participants requis, {len(pool)} disponibles.")
    pid_y4 = _participant_y4_map(dataset, pool)
    by_class: Dict[int, List[str]] = defaultdict(list)
    for p in pool:
        by_class[pid_y4[p]].append(p)

    rng = np.random.default_rng(seed + fold_idx)
    chosen: List[str] = []
    for y4 in range(N_CLASSES_Y4):
        if len(chosen) >= k:
            break
        candidates = [p for p in by_class[y4] if p not in chosen]
        if candidates:
            chosen.append(str(rng.choice(candidates)))

    remaining = [p for p in pool if p not in chosen]
    n_needed = k - len(chosen)
    if n_needed > 0 and remaining:
        extra = rng.choice(remaining, size=min(n_needed, len(remaining)), replace=False)
        chosen.extend(str(p) for p in np.atleast_1d(extra))
    return sorted(chosen)


def _aug_source_participants(rec: dict) -> set[str]:
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
    for rec in merged.values():
        if "y4" not in rec or rec["y4"] < 0:
            rec["y4"] = Y9_TO_Y4[int(rec["y9"])]
    return merged


def assert_no_leakage(train_trials, val_trials, test_trials, p_test, val_participants):
    val_set = {str(p) for p in val_participants}
    test_set = {str(p_test)}
    exclude = val_set | test_set

    def real_pids(trials):
        out = set()
        for k, v in trials.items():
            if v.get("is_augmented", False) or str(k).startswith("synth_"):
                continue
            pid = k[0] if isinstance(k, tuple) else str(k).split("_")[0]
            out.add(str(pid))
        return out

    assert real_pids(test_trials) == test_set
    assert real_pids(val_trials) == val_set
    assert real_pids(train_trials).isdisjoint(test_set)
    assert real_pids(train_trials).isdisjoint(val_set)

    for rec in val_trials.values():
        assert not rec.get("is_augmented", False)
    for rec in test_trials.values():
        assert not rec.get("is_augmented", False)

    for k, rec in train_trials.items():
        if not rec.get("is_augmented", False):
            continue
        sources = _aug_source_participants(rec)
        assert sources, f"synthetique {k} sans source"
        assert not (sources & exclude), f"fuite augmentation {k}"


def make_class_weights(
    items: List[dict],
    key: str,
    n_classes: int,
    include_augmented: bool = False,
) -> torch.Tensor:
    """Poids de classes equilibres ; classes absentes du fold -> poids 1.0."""
    if include_augmented:
        labels = [int(it[key]) for it in items]
    else:
        labels = [int(it[key]) for it in items if not it["is_augmented"]]
    if not labels:
        return torch.ones(n_classes, dtype=torch.float32)

    labels_arr = np.array(labels)
    present = np.unique(labels_arr)
    present_weights = compute_class_weight("balanced", classes=present, y=labels_arr)

    weights = np.ones(n_classes, dtype=np.float32)
    for cls, w in zip(present, present_weights):
        weights[int(cls)] = float(w)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_fold(
    train_trials: Dict,
    val_trials: Dict,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    alpha: float,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    fold_i: int = 0,
    n_folds: int = 1,
) -> Tuple[MultiTaskEVICEMS, dict]:
    mean, std = compute_norm_stats({k: v for k, v in train_trials.items() if not v.get("is_augmented")})
    train_items = trials_to_items(train_trials, mean, std)
    val_items = trials_to_items(val_trials, mean, std)

    w4 = make_class_weights(train_items, "y4", N_CLASSES_Y4).to(device)
    w9 = make_class_weights(train_items, "y9", N_CLASSES_Y9, include_augmented=True).to(device)
    ce4 = nn.CrossEntropyLoss(weight=w4, reduction="none")
    ce9 = nn.CrossEntropyLoss(weight=w9, reduction="none")

    model = MultiTaskEVICEMS(n_features=N_FEATURES, seq_len=SEQ_LEN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    val_loader = DataLoader(
        MultiTaskDataset(val_items, random_crop=False),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    best_val, best_state, wait = float("inf"), None, 0
    best_epoch = 0
    stopped_epoch = epochs
    epoch_logs: List[dict] = []

    _log(f"  [fold {fold_i + 1}/{n_folds}] demarrage entrainement ({epochs} epochs max, patience={patience})")

    for ep in range(1, epochs + 1):
        model.train()
        loader = DataLoader(
            MultiTaskDataset(build_stratified_items(train_items), random_crop=True),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )
        train_losses = []

        for xs, y4, y9, is_aug in loader:
            xs = xs.to(device)
            y4 = y4.to(device)
            y9 = y9.to(device)
            is_aug = is_aug.to(device)

            logits4, logits9 = model(xs)
            loss4 = ce4(logits4, y4)
            loss9 = ce9(logits9, y9)

            real_mask = ~is_aug
            loss = torch.tensor(0.0, device=device)
            if real_mask.any():
                loss = loss + alpha * loss4[real_mask].mean() + (1.0 - alpha) * loss9[real_mask].mean()
            if is_aug.any():
                loss = loss + (1.0 - alpha) * loss9[is_aug].mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        val_y4_true, val_y4_pred = [], []
        val_y9_true, val_y9_pred = [], []
        with torch.no_grad():
            for xs, y4, y9, is_aug in val_loader:
                xs = xs.to(device)
                y4 = y4.to(device)
                y9 = y9.to(device)
                logits4, logits9 = model(xs)
                vloss = alpha * ce4(logits4, y4).mean() + (1.0 - alpha) * ce9(logits9, y9).mean()
                val_losses.append(float(vloss.item()))
                val_y4_true.extend(y4.cpu().tolist())
                val_y4_pred.extend(logits4.argmax(dim=-1).cpu().tolist())
                val_y9_true.extend(y9.cpu().tolist())
                val_y9_pred.extend(logits9.argmax(dim=-1).cpu().tolist())

        tloss = float(np.mean(train_losses)) if train_losses else float("inf")
        vloss = float(np.mean(val_losses)) if val_losses else float("inf")
        val_acc4 = float(accuracy_score(val_y4_true, val_y4_pred)) if val_y4_true else 0.0
        val_acc9 = float(accuracy_score(val_y9_true, val_y9_pred)) if val_y9_true else 0.0
        improved = vloss < best_val - 1e-5

        epoch_logs.append({
            "epoch": ep,
            "train_loss": tloss,
            "val_loss": vloss,
            "val_acc_y4": val_acc4,
            "val_acc_y9": val_acc9,
            "best": improved,
        })

        marker = " *best*" if improved else f" (patience {wait + 1}/{patience})"
        _log(
            f"  [fold {fold_i + 1}/{n_folds}] epoch {ep:3d}/{epochs} | "
            f"train={tloss:.4f} | val={vloss:.4f} | "
            f"val_acc y4={val_acc4:.3f} y9={val_acc9:.3f}{marker}"
        )

        if improved:
            best_val = vloss
            best_epoch = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                stopped_epoch = ep
                _log(f"  [fold {fold_i + 1}/{n_folds}] early stopping a epoch {ep} (best={best_epoch}, val={best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model._norm_mean = mean
    model._norm_std = std
    return model, {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "stopped_epoch": stopped_epoch,
        "epoch_logs": epoch_logs,
    }


@torch.no_grad()
def predict_trial(model: MultiTaskEVICEMS, x: np.ndarray, device: torch.device) -> Tuple[int, int]:
    model.eval()
    x = pad_or_crop(x, SEQ_LEN, random_crop=False)
    xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
    logits4, logits9 = model(xt)
    return int(logits4.argmax(dim=-1).item()), int(logits9.argmax(dim=-1).item())


def run_lopo(
    dataset: Dict,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    alpha: float,
    max_folds: Optional[int],
    no_aug: bool,
    fast_aug: bool,
    seed: int,
) -> Tuple[pd.DataFrame, dict]:
    by_pid = defaultdict(list)
    for k in dataset:
        by_pid[k[0]].append(k)
    pids = sorted(by_pid.keys())
    if max_folds is not None:
        pids = pids[:max_folds]

    rows: List[dict] = []
    fold_summaries: List[dict] = []

    for fold_i, p_test in enumerate(pids):
        held_keys = set(by_pid[p_test])
        test_trials = {k: dataset[k] for k in held_keys}
        train_pool_real = {k: v for k, v in dataset.items() if k not in held_keys}
        train_pool_pids = sorted({k[0] for k in train_pool_real})

        val_participants = select_stratified_val_participants(
            train_pool_pids, dataset, fold_idx=fold_i, seed=seed,
        )
        val_set = set(val_participants)
        train_fit_real = {k: v for k, v in train_pool_real.items() if k[0] not in val_set}
        val_trials = {k: v for k, v in train_pool_real.items() if k[0] in val_set}

        _log(
            f"\n[LOPO fold {fold_i + 1}/{len(pids)}] TEST={p_test} | "
            f"train_reel={len(train_fit_real)} | val={len(val_trials)} | test={len(test_trials)}"
        )

        synth = {}
        if no_aug:
            _log(f"  [fold {fold_i + 1}] augmentation desactivee")
        else:
            _log(
                f"  [fold {fold_i + 1}] augmentation en cours "
                f"(DBA+jitter+timewarp — peut prendre 10 a 40 min, soyez patient)..."
            )
            t_aug = time.time()
            synth = augment_model2_fold(
                train_fit_real, fold_i, seed=seed, fast_mode=fast_aug,
            )
            _log(
                f"  [fold {fold_i + 1}] augmentation terminee: "
                f"+{len(synth)} synthetiques en {time.time() - t_aug:.1f}s"
            )

        for rec in synth.values():
            rec["y4"] = Y9_TO_Y4[int(rec["y9"])]

        exclude = val_set | {str(p_test)}
        train_trials = build_train_trials(train_fit_real, synth, exclude)
        assert_no_leakage(train_trials, val_trials, test_trials, p_test, val_participants)

        _log(
            f"  [fold {fold_i + 1}] pret pour entrainement | "
            f"train_total={len(train_trials)} (aug={sum(1 for v in train_trials.values() if v.get('is_augmented'))})"
        )

        model, train_info = train_one_fold(
            train_trials, val_trials, device,
            epochs=epochs, patience=patience, batch_size=batch_size, alpha=alpha,
            fold_i=fold_i, n_folds=len(pids),
        )
        mean, std = model._norm_mean, model._norm_std

        fold_rows = []
        for key in held_keys:
            rec = dataset[key]
            x = apply_norm(rec["X"], mean, std)
            pred4, pred9 = predict_trial(model, x, device)
            row = {
                "participant": str(p_test),
                "trial": str(key[1]),
                "true_y4": int(rec["y4"]),
                "pred_y4": pred4,
                "true_y9": int(rec["y9"]),
                "pred_y9": pred9,
                "expertise": rec["expertise"],
                "level": rec["level"],
                "fold": fold_i + 1,
            }
            rows.append(row)
            fold_rows.append(row)

        y4_true = [r["true_y4"] for r in fold_rows]
        y4_pred = [r["pred_y4"] for r in fold_rows]
        y9_true = [r["true_y9"] for r in fold_rows]
        y9_pred = [r["pred_y9"] for r in fold_rows]
        fold_summaries.append({
            "fold": fold_i + 1,
            "participant": str(p_test),
            "acc_y4": float(accuracy_score(y4_true, y4_pred)),
            "f1_y4": float(f1_score(y4_true, y4_pred, average="macro", zero_division=0)),
            "acc_y9": float(accuracy_score(y9_true, y9_pred)),
            "f1_y9": float(f1_score(y9_true, y9_pred, average="macro", zero_division=0)),
            **train_info,
        })
        _log(
            f"  fold {fold_i + 1} test metrics -> y4 acc={fold_summaries[-1]['acc_y4']:.3f} "
            f"f1={fold_summaries[-1]['f1_y4']:.3f} | "
            f"y9 acc={fold_summaries[-1]['acc_y9']:.3f} f1={fold_summaries[-1]['f1_y9']:.3f}"
        )

    preds_df = pd.DataFrame(rows)
    metrics = compute_global_metrics(preds_df, fold_summaries)
    return preds_df, metrics


def compute_global_metrics(preds_df: pd.DataFrame, fold_summaries: List[dict]) -> dict:
    if preds_df.empty:
        return {}

    y4_true = preds_df["true_y4"].to_numpy()
    y4_pred = preds_df["pred_y4"].to_numpy()
    y9_true = preds_df["true_y9"].to_numpy()
    y9_pred = preds_df["pred_y9"].to_numpy()

    metrics = {
        "n_trials": int(len(preds_df)),
        "n_folds": int(len(fold_summaries)),
        "acc_y4": float(accuracy_score(y4_true, y4_pred)),
        "f1_macro_y4": float(f1_score(y4_true, y4_pred, average="macro", zero_division=0)),
        "acc_y9": float(accuracy_score(y9_true, y9_pred)),
        "f1_macro_y9": float(f1_score(y9_true, y9_pred, average="macro", zero_division=0)),
        "confusion_y4": confusion_matrix(y4_true, y4_pred, labels=list(range(N_CLASSES_Y4))).tolist(),
        "confusion_y9": confusion_matrix(y9_true, y9_pred, labels=list(range(N_CLASSES_Y9))).tolist(),
        "report_y4": classification_report(
            y4_true, y4_pred,
            labels=list(range(N_CLASSES_Y4)),
            target_names=Y4_NAMES,
            zero_division=0,
            output_dict=True,
        ),
        "report_y9": classification_report(
            y9_true, y9_pred,
            labels=list(range(N_CLASSES_Y9)),
            target_names=Y9_NAMES,
            zero_division=0,
            output_dict=True,
        ),
        "fold_summaries": fold_summaries,
    }
    return metrics


def save_results(preds_df: pd.DataFrame, metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Resultats sauvegardes dans {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrainement multi-tache ICEMS (y4 + y9)")
    p.add_argument("--smoke-test", action="store_true", help="2 folds, 3 epochs, augmentation rapide")
    p.add_argument("--full", action="store_true", help="LOPO complet")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--alpha", type=float, default=0.4, help="Poids de la loss y4 (1-alpha pour y9)")
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--no-aug", action="store_true", help="Desactive l'augmentation y9")
    p.add_argument("--seed", type=int, default=NESTED_SEED)
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.smoke_test and not args.full:
        print("Precisez --smoke-test ou --full")
        sys.exit(1)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")

    _log("Chargement du dataset...")
    dataset = load_real_dataset()
    _log(f"Dataset reel charge : {len(dataset)} trials, {len({k[0] for k in dataset})} participants")

    epochs = 3 if args.smoke_test else args.epochs
    patience = 2 if args.smoke_test else args.patience
    max_folds = 2 if args.smoke_test else args.max_folds
    fast_aug = args.smoke_test

    _log(
        f"Config: epochs={epochs}, patience={patience}, batch={args.batch_size}, "
        f"alpha={args.alpha}, folds={max_folds or 'all'}, fast_aug={fast_aug}, no_aug={args.no_aug}"
    )
    _log("Demarrage LOPO...")

    preds_df, metrics = run_lopo(
        dataset=dataset,
        device=device,
        epochs=epochs,
        patience=patience,
        batch_size=args.batch_size,
        alpha=args.alpha,
        max_folds=max_folds,
        no_aug=args.no_aug,
        fast_aug=fast_aug,
        seed=args.seed,
    )

    print("\n=== METRIQUES GLOBALES ===", flush=True)
    print(f"y4 (Expertise) : acc={metrics['acc_y4']:.3f} | f1_macro={metrics['f1_macro_y4']:.3f}", flush=True)
    print(f"y9 (Formation) : acc={metrics['acc_y9']:.3f} | f1_macro={metrics['f1_macro_y9']:.3f}", flush=True)

    run_name = date.today().isoformat()
    suffix = "smoke" if args.smoke_test else "full"
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / f"{run_name}_{suffix}"
    save_results(preds_df, metrics, out_dir)


if __name__ == "__main__":
    main()
