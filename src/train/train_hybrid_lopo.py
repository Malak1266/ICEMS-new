"""
train_hybrid_lopo.py
====================
Entraînement LOPO nested — Hybrid LSTM-Transformer + loss hiérarchique.

Anti-fuite :
  - split par participant (LOPO)
  - scaler fit sur train_fit réels uniquement
  - early stopping sur k_inner participants réels internes
  - augmentation fold-interne uniquement (hook augment_fn)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import normalize_sublevel, sublevel_score, SUBLEVEL_TO_RANK, SUBLEVEL_TO_TIER
from eval.augment_extremes import augment_trial, jitter_sequence
from eval.expertise_metrics import compute_all, plot_confusion, plot_history, plot_scatter, save_metrics
from losses.hoel import HOEL
from models.hybrid_lstm_transformer import HybridConfig, HybridLSTMTransformer

# #region agent log
_DEBUG_LOG = Path(__file__).resolve().parents[2] / "debug-7e7e44.log"
def _dbg(location: str, message: str, data: dict, hypothesis_id: str = "load") -> None:
    import json as _json
    payload = {"sessionId": "7e7e44", "timestamp": int(time.time() * 1000),
               "location": location, "message": message, "data": data,
               "hypothesisId": hypothesis_id, "runId": "impl"}
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(_json.dumps(payload) + "\n")
    except OSError:
        pass
# #endregion

# ─── Features 13 canaux ──────────────────────────────────────────────────────
N_BASE_FEATURES = 10
VALID_COL = 9
N_FEATURES = 13

TIER_NAMES: Tuple[str, ...] = ("PGY", "Fellow", "Expert")
NESTED_LOPO_SEED = 42


def enrich_kinematic_features(X: np.ndarray) -> np.ndarray:
    vel = np.linalg.norm(X[:, [0, 3, 6]], axis=1).astype(np.float32)
    jerk = np.linalg.norm(X[:, [2, 5, 8]], axis=1).astype(np.float32)
    valid = X[:, VALID_COL].astype(np.float32)
    derived = np.stack([vel, jerk / (vel + 1e-6), valid], axis=1)
    return np.concatenate([X[:, :N_BASE_FEATURES], derived], axis=1).astype(np.float32)


@dataclass
class Trial:
    X: np.ndarray
    score: float
    tier: int
    rank: int
    participant: str
    sublevel: str
    trial_id: str = ""
    is_augmented: bool = False


@dataclass
class TrainConfig:
    seq_len: int = 4000
    crop_mode: str = "start"
    batch_size: int = 16
    max_epochs: int = 50
    patience: int = 8
    k_inner: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = NESTED_LOPO_SEED


def _load_participant_levels(csv_path: Path) -> Dict[str, str]:
    import pandas as pd
    df = pd.read_csv(csv_path, sep=";", encoding="latin-1")
    df = df.dropna(subset=["ID"])
    df["participant"] = df["ID"].astype(int).astype(str).str.zfill(8)
    df["level"] = df["Level"].astype(str).str.strip()
    return dict(zip(df["participant"], df["level"]))


def load_dataset_icems(
    pkl_path: Path | str = "data/continuous_per_trial.pkl",
    csv_path: Path | str = "data/Exvivo_trial_Participants(Sheet1).csv",
) -> List[Trial]:
    """Charge 136 trials réels ICEMS → list[Trial] pour LOPO hybrid."""
    pkl_path = Path(pkl_path)
    csv_path = Path(csv_path)
    pid_to_level = _load_participant_levels(csv_path)

    with open(pkl_path, "rb") as f:
        raw: dict = pickle.load(f)

    trials: List[Trial] = []
    missing_csv: Set[str] = set()
    tier_counts: Counter = Counter()

    for (pid, tid), rec in raw.items():
        pid = str(pid)
        tid = str(tid)
        level_raw = pid_to_level.get(pid) or rec.get("level", "")
        if pid not in pid_to_level:
            missing_csv.add(pid)

        score = sublevel_score(str(level_raw))
        canonical = normalize_sublevel(str(level_raw))
        tier = SUBLEVEL_TO_TIER[canonical]
        rank = SUBLEVEL_TO_RANK[canonical]
        X = enrich_kinematic_features(np.asarray(rec["X"], dtype=np.float32))
        tier_counts[tier] += 1

        trials.append(Trial(
            X=X,
            score=float(score),
            tier=int(tier),
            rank=int(rank),
            participant=pid,
            sublevel=canonical,
            trial_id=tid,
            is_augmented=False,
        ))

    if missing_csv:
        raise KeyError(
            f"{len(missing_csv)} participant(s) absents du CSV : "
            f"{sorted(missing_csv)[:5]}"
        )

    # #region agent log
    _dbg("train_hybrid_lopo.py:load_dataset_icems", "dataset loaded", {
        "n_trials": len(trials),
        "n_participants": len({t.participant for t in trials}),
        "n_features": int(trials[0].X.shape[1]) if trials else 0,
        "tier_counts": dict(tier_counts),
        "missing_csv": len(missing_csv),
    }, "H1")
    # #endregion

    return trials


# ─── Augmentation Option D (fold train uniquement) ───────────────────────────

def augment_fn(train_real: List[Trial], seed: int = 42) -> List[Trial]:
    """
    Option D : équilibrage Staff (jitter) puis ×6 (time-warp → magnitude-warp → jitter).
    Signature : augment_fn(train_real) -> list[Trial] (réels + synthétiques).
    """
    rng = np.random.default_rng(seed)
    real_only = [t for t in train_real if not t.is_augmented]

    ms_real = [t for t in real_only if t.sublevel == "ms"]
    staff_real = [t for t in real_only if t.sublevel == "staff"]
    n_ms, n_staff = len(ms_real), len(staff_real)
    n_to_create = max(0, n_ms - n_staff)

    synth: List[Trial] = []
    for _ in range(n_to_create):
        if not staff_real:
            break
        parent = staff_real[int(rng.integers(0, len(staff_real)))]
        X_new = jitter_sequence(parent.X, sigma=0.015, rng=rng)
        synth.append(Trial(
            X=X_new, score=parent.score, tier=parent.tier, rank=parent.rank,
            participant=parent.participant, sublevel=parent.sublevel,
            trial_id=f"{parent.trial_id}_eq_jitter",
            is_augmented=True,
        ))

    balanced = list(real_only) + synth
    global_multiplier = 6
    global_synth: List[Trial] = []
    for parent in balanced:
        for j in range(max(0, global_multiplier - 1)):
            X_new = augment_trial(parent.X, rng=rng)
            global_synth.append(Trial(
                X=X_new, score=parent.score, tier=parent.tier, rank=parent.rank,
                participant=parent.participant, sublevel=parent.sublevel,
                trial_id=f"{parent.trial_id}_aug_{j}",
                is_augmented=True,
            ))

    out = balanced + global_synth
    rng.shuffle(out)

    # #region agent log
    _dbg("train_hybrid_lopo.py:augment_fn", "augmentation done", {
        "n_real_in": len(real_only),
        "n_staff_eq": len(synth),
        "n_global_synth": len(global_synth),
        "n_out": len(out),
        "multiplier": global_multiplier,
    }, "H2")
    # #endregion

    return out


# ─── Prétraitement / dataset ─────────────────────────────────────────────────

def compute_norm_stats(trials: Sequence[Trial]) -> Tuple[np.ndarray, np.ndarray]:
    real = [t for t in trials if not t.is_augmented]
    kin = np.concatenate([t.X.reshape(-1, N_FEATURES) for t in real], axis=0)
    mean = kin.mean(axis=0).astype(np.float32)
    std = (kin.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std


def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def crop_sequence(X: np.ndarray, seq_len: int, mode: str = "start") -> np.ndarray:
    T, F = X.shape
    if T >= seq_len:
        if mode == "start":
            return X[:seq_len].astype(np.float32)
        start = np.random.randint(0, T - seq_len)
        return X[start:start + seq_len].astype(np.float32)
    pad = np.zeros((seq_len - T, F), dtype=np.float32)
    return np.concatenate([X, pad], axis=0)


class HybridTrialDataset(Dataset):
    def __init__(
        self,
        trials: Sequence[Trial],
        mean: np.ndarray,
        std: np.ndarray,
        cfg: TrainConfig,
        train: bool = True,
    ):
        self.trials = list(trials)
        self.mean = mean
        self.std = std
        self.cfg = cfg
        self.train = train

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx: int):
        t = self.trials[idx]
        x = apply_norm(t.X, self.mean, self.std)
        mode = self.cfg.crop_mode if self.train else "start"
        x = crop_sequence(x, self.cfg.seq_len, mode=mode)
        return (
            torch.from_numpy(x),
            torch.tensor(t.score, dtype=torch.float32),
            torch.tensor(t.tier, dtype=torch.long),
            torch.tensor(t.rank, dtype=torch.long),
        )


def collate_hybrid(batch):
    xs, scores, tiers, ranks = zip(*batch)
    return torch.stack(xs), torch.stack(scores), torch.stack(tiers), torch.stack(ranks)


# ─── Splits anti-fuite ───────────────────────────────────────────────────────

def _participant_tier_map(trials: Sequence[Trial], pids: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for pid in pids:
        for t in trials:
            if t.participant == pid and not t.is_augmented:
                out[pid] = t.tier
                break
    return out


def select_internal_val_participants(
    train_pool_pids: List[str],
    trials: Sequence[Trial],
    k: int,
    seed: int,
    fold_idx: int,
) -> List[str]:
    """k participants réels stratifiés par tier pour validation interne."""
    pool = sorted({str(p) for p in train_pool_pids})
    if len(pool) < k + 1:
        raise ValueError(
            f"Fold {fold_idx}: au moins {k + 1} participants requis, {len(pool)} dispo."
        )
    pid_tier = _participant_tier_map(trials, pool)
    by_tier: Dict[int, List[str]] = defaultdict(list)
    for p in pool:
        by_tier[pid_tier[p]].append(p)

    rng = np.random.default_rng(seed + fold_idx)
    chosen: List[str] = []
    for tier in range(3):
        if len(chosen) >= k:
            break
        cands = [p for p in by_tier[tier] if p not in chosen]
        if cands:
            chosen.append(str(rng.choice(cands)))

    remaining = [p for p in pool if p not in chosen]
    while len(chosen) < k and remaining:
        chosen.append(str(rng.choice(remaining)))
        remaining = [p for p in pool if p not in chosen]

    return sorted(chosen[:k])


def assert_no_leakage(
    train: List[Trial],
    internal_val: List[Trial],
    test: List[Trial],
    held_pid: str,
    int_val_pids: Set[str],
) -> None:
    train_pids = {t.participant for t in train if not t.is_augmented}
    for t in internal_val:
        assert not t.is_augmented, "FUITE: augmenté dans val interne"
        assert t.participant in int_val_pids
    for t in test:
        assert not t.is_augmented, "FUITE: augmenté dans test"
        assert t.participant == held_pid
    assert held_pid not in train_pids
    assert held_pid not in int_val_pids
    for t in train:
        if t.is_augmented:
            assert t.participant not in int_val_pids
            assert t.participant != held_pid


# ─── Entraînement ────────────────────────────────────────────────────────────

def train_one_fold(
    train_trials: List[Trial],
    val_trials: List[Trial],
    hybrid_cfg: HybridConfig,
    train_cfg: TrainConfig,
    loss_fn: HOEL,
    device: torch.device,
) -> Tuple[HybridLSTMTransformer, dict]:
    mean, std = compute_norm_stats(train_trials)
    train_ds = HybridTrialDataset(train_trials, mean, std, train_cfg, train=True)
    val_ds = HybridTrialDataset(val_trials, mean, std, train_cfg, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate_hybrid,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate_hybrid,
    )

    model = HybridLSTMTransformer(hybrid_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    history: List[dict] = []

    for epoch in range(1, train_cfg.max_epochs + 1):
        model.train()
        train_losses = []
        train_terms = {"mse": [], "asym": [], "pair": [], "top": [], "anchor": []}
        for xb, yb, tb, rb in train_loader:
            xb, yb, tb, rb = xb.to(device), yb.to(device), tb.to(device), rb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss_dict = loss_fn(pred, yb, tb, rb)
            loss = loss_dict["loss"]
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))
            for k in train_terms:
                train_terms[k].append(float(loss_dict[k].item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, tb, rb in val_loader:
                xb, yb, tb, rb = xb.to(device), yb.to(device), tb.to(device), rb.to(device)
                pred = model(xb)
                val_losses.append(float(loss_fn(pred, yb, tb, rb)["loss"].item()))

        tr = float(np.mean(train_losses)) if train_losses else float("nan")
        va = float(np.mean(val_losses)) if val_losses else float("nan")
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})

        term_means = {k: float(np.mean(v)) if v else float("nan") for k, v in train_terms.items()}
        print(
            f"  epoch {epoch}: train={tr:.4f} val={va:.4f} | "
            f"mse={term_means['mse']:.4f} asym={term_means['asym']:.4f} "
            f"pair={term_means['pair']:.4f} top={term_means['top']:.4f} "
            f"anchor={term_means['anchor']:.4f}"
        )

        if va < best_val:
            best_val = va
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= train_cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    info = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epochs_run": len(history),
        "early_stopped": len(history) < train_cfg.max_epochs,
        "history": history,
        "norm_mean": mean,
        "norm_std": std,
    }
    return model, info


@torch.no_grad()
def predict_trials(
    model: HybridLSTMTransformer,
    trials: Sequence[Trial],
    mean: np.ndarray,
    std: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
) -> List[float]:
    model.eval()
    ds = HybridTrialDataset(trials, mean, std, cfg, train=False)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_hybrid)
    preds: List[float] = []
    for xb, _, _, _ in loader:
        xb = xb.to(device)
        out = model(xb).squeeze(-1).cpu().numpy()
        preds.extend(out.tolist())
    return preds


def run_lopo(
    trials: List[Trial],
    hybrid_cfg: HybridConfig,
    train_cfg: TrainConfig,
    loss_fn: HOEL,
    device: torch.device,
    out_dir: Path,
    augment: Callable[[List[Trial], int], List[Trial]] = augment_fn,
    max_folds: Optional[int] = None,
    save_checkpoints: bool = False,
) -> Tuple[List[dict], dict]:
    """LOPO nested avec hook augment_fn et early stopping interne."""
    real_trials = [t for t in trials if not t.is_augmented]
    by_pid: Dict[str, List[Trial]] = defaultdict(list)
    for t in real_trials:
        by_pid[t.participant].append(t)

    pids = sorted(by_pid.keys())
    if max_folds is not None:
        pids = pids[:max_folds]

    out_dir.mkdir(parents=True, exist_ok=True)

    all_preds: List[dict] = []
    fold_histories: List[List[dict]] = []
    fold_infos: List[dict] = []

    for fold_i, held_pid in enumerate(pids):
        held_set = {held_pid}
        train_pool = [t for t in real_trials if t.participant not in held_set]
        test_trials = by_pid[held_pid]
        pool_pids = sorted({t.participant for t in train_pool})

        int_val_pids = select_internal_val_participants(
            pool_pids, real_trials, k=train_cfg.k_inner,
            seed=train_cfg.seed, fold_idx=fold_i,
        )
        int_val_set = set(int_val_pids)

        train_fit_real = [t for t in train_pool if t.participant not in int_val_set]
        internal_val = [t for t in train_pool if t.participant in int_val_set]

        assert_no_leakage(train_fit_real, internal_val, test_trials, held_pid, int_val_set)

        train_aug = augment(train_fit_real, seed=train_cfg.seed + fold_i)
        assert_no_leakage(train_aug, internal_val, test_trials, held_pid, int_val_set)

        print(
            f"\n[fold {fold_i + 1}/{len(pids)}] held={held_pid} | "
            f"int_val={int_val_pids} | train_real={len(train_fit_real)} | "
            f"train_aug={sum(1 for t in train_aug if t.is_augmented)}"
        )

        model, info = train_one_fold(
            train_aug, internal_val, hybrid_cfg, train_cfg, loss_fn, device,
        )
        fold_histories.append(info["history"])
        fold_infos.append({"fold": fold_i + 1, "held": held_pid, **info})

        # Meilleur modèle (early stopping) — rechargé dans train_one_fold
        torch.save(model.state_dict(), out_dir / f"fold_{fold_i + 1}_best.pt")

        if save_checkpoints:
            from dataclasses import asdict
            ckpt_dir = out_dir / "checkpoints" / f"fold_{fold_i + 1:02d}_{held_pid}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "hybrid_cfg": asdict(hybrid_cfg),
                "norm_mean": info["norm_mean"],
                "norm_std": info["norm_std"],
                "seq_len": train_cfg.seq_len,
            }, ckpt_dir / "model.pt")
            with open(ckpt_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump({"fold": fold_i + 1, "held": held_pid}, f)

        preds = predict_trials(
            model, test_trials, info["norm_mean"], info["norm_std"], train_cfg, device,
        )
        for t, pred in zip(test_trials, preds):
            all_preds.append({
                "participant": t.participant,
                "trial_id": t.trial_id,
                "sublevel": t.sublevel,
                "tier": t.tier,
                "score_true": t.score,
                "score_pred": float(pred),
            })

    metrics = compute_all(all_preds)
    aux = {"fold_infos": fold_infos, "fold_histories": fold_histories, "metrics": metrics}

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.pkl", "wb") as f:
        pickle.dump(all_preds, f)
    with open(out_dir / "fold_infos.pkl", "wb") as f:
        pickle.dump(fold_infos, f)
    save_metrics(metrics, out_dir)
    plot_scatter(all_preds, out_dir / "scatter.png")
    plot_confusion(all_preds, out_dir / "confusion_tier.png")
    plot_history(fold_histories, out_dir / "history.png")

    # #region agent log
    _dbg("train_hybrid_lopo.py:run_lopo", "lopo complete", {
        "n_folds": len(pids),
        "n_preds": len(all_preds),
        "r_participant": metrics.get("r_participant"),
        "mae": metrics.get("mae"),
    }, "H3")
    # #endregion

    return all_preds, aux


def make_hoel() -> HOEL:
    return HOEL(
        lambda_asym=0.20,
        lambda_pair=0.30,
        lambda_top=0.40,
        lambda_anchor=0.08,
        tier_weights={0: 1.0, 1: 2.5, 2: 4.0},
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid LSTM-Transformer LOPO + HOEL")
    ap.add_argument("--out", type=Path, default=Path("results/hybrid_hoel"))
    ap.add_argument("--pkl", type=Path, default=Path("data/continuous_per_trial.pkl"))
    ap.add_argument("--csv", type=Path, default=Path("data/Exvivo_trial_Participants(Sheet1).csv"))
    ap.add_argument("--seed", type=int, default=NESTED_LOPO_SEED)
    ap.add_argument("--smoke", action="store_true", help="2 folds, 2 epochs, max_folds=2")
    ap.add_argument("--save-checkpoints", action="store_true",
                    help="Sauvegarde les poids par fold (figures temporelles)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_cfg = TrainConfig(seed=args.seed)
    loss_fn = make_hoel()

    if args.smoke:
        train_cfg.max_epochs = 2
        train_cfg.patience = 1
        train_cfg.seq_len = 512  # réduit pour smoke local ; prod = 4000
        max_folds = 2
        print("[smoke] seq_len=512 (prod=4000), max_folds=2, max_epochs=2")
    else:
        max_folds = None

    hybrid_cfg = HybridConfig(
        n_features=13,
        seq_len=train_cfg.seq_len,
        lstm_hidden=128,
        d_model=128,
        nhead=4,
        key_dim=32,
        n_transformer_blocks=1,
        dropout=0.30,
        pos_encoding="sinusoidal",
        bidirectional=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} | loss=HOEL | smoke={args.smoke}")

    trials = load_dataset_icems(args.pkl, args.csv)
    print(f"[data] {len(trials)} trials, {len({t.participant for t in trials})} participants")

    out_dir = args.out if not args.smoke else args.out / "smoke"
    preds, aux = run_lopo(
        trials, hybrid_cfg, train_cfg, loss_fn, device, out_dir,
        max_folds=max_folds,
        save_checkpoints=args.save_checkpoints,
    )
    print(f"[metrics] {json.dumps(aux['metrics'], indent=2)}")
    print(f"[done] -> {out_dir}")


if __name__ == "__main__":
    main()
