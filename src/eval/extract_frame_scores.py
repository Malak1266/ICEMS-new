"""
extract_frame_scores.py
=======================
Extrait les scores d'expertise **par frame** depuis un run LOPO Hybrid + HOEL.

Principe (aligné sur hybrid1_evicems / HybridLSTMTransformer) :
    lstm → lstm_dropout → proj → + pos_embedding → blocks (Transformer)
    → **sans** mean(dim=1) → head appliquée à chaque pas de temps (B, T, d_model)

Chaque trial utilise le checkpoint du fold LOPO où le participant était held-out
(fold_{k}_best.pt + norm_mean/std du fold dans fold_infos.pkl).

Sortie : frame_predictions.pkl
------------------------------
{
    "meta": {
        "model": "HybridLSTMTransformer",
        "run_dir": "...",
        "seq_len": 4000,
        "n_entries": 136,
    },
    "entries": [
        {
            "participant": "01020614",
            "trial_id": "3",
            "fold": 1,
            "sublevel": "pgy3",
            "tier": 1,
            "score_true": -0.40,
            "trial_score_pred": -0.12,      # score trial-level LOPO (predictions.pkl)
            "trial_score_from_frames": -0.11,  # mean(frame_scores) — diagnostic
            "n_frames_raw": 3200,
            "n_frames_used": 3200,
            "time_norm": np.ndarray (T,),   # linspace [0, 1]
            "frame_scores": np.ndarray (T,), # ∈ [-1, 1]
        },
        ...
    ]
}

Usage :
    python -m eval.extract_frame_scores \\
        --run-dir results/hybrid_hoel_b \\
        --pkl data/continuous_per_trial.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.hybrid_lstm_transformer import HybridConfig, HybridLSTMTransformer
from train.train_hybrid_lopo import (
    N_FEATURES,
    apply_norm,
    crop_sequence,
    enrich_kinematic_features,
)

SUBLEVEL_TO_CLASS4 = {
    "ms": "student",
    "pgy1": "junior", "pgy2": "junior", "pgy3": "junior",
    "pgy4": "junior", "pgy5": "junior",
    "pgy6": "senior", "fellow": "senior",
    "staff": "expert",
}


def _sublevel_to_class4(sublevel: str) -> str:
    return SUBLEVEL_TO_CLASS4.get(sublevel.strip().lower(), "junior")


@torch.no_grad()
def forward_frame_scores(model: HybridLSTMTransformer, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass sans pooling global : score par frame.

    Reprend les couches nommées comme dans hybrid1_evicems.py :
      lstm, lstm_dropout, proj, pos_embedding, transformer/blocks, head
    """
    assert x.dim() == 3, f"attendu (B, T, F), reçu {tuple(x.shape)}"

    h, _ = model.lstm(x)
    h = model.lstm_dropout(h)
    h = model.proj(h)
    h = model._add_pos(h)
    for block in model.blocks:
        h = block(h)
    # (B, T, d_model) — pas de h.mean(dim=1)
    scores = model.head(h).squeeze(-1)  # (B, T)
    return scores


def _load_fold_infos(run_dir: Path) -> List[dict]:
    path = run_dir / "fold_infos.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable — relancer l'entraînement avec la version à jour de "
            "train_hybrid_lopo.py (sauvegarde fold_infos.pkl)."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_predictions(run_dir: Path) -> List[dict]:
    path = run_dir / "predictions.pkl"
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "preds" in payload:
        return list(payload["preds"])
    raise ValueError(f"Format predictions.pkl non reconnu : {path}")


def _resolve_checkpoint(
    run_dir: Path,
    fold_num: int,
    fold_info: dict,
) -> Tuple[dict, Optional[HybridConfig]]:
    """Charge state_dict depuis fold_{k}_best.pt ou checkpoints/fold_XX_*/model.pt."""
    simple = run_dir / f"fold_{fold_num}_best.pt"
    if simple.exists():
        return torch.load(simple, map_location="cpu", weights_only=False), None

    held = fold_info.get("held", "")
    rich_dir = run_dir / "checkpoints" / f"fold_{fold_num:02d}_{held}"
    rich_path = rich_dir / "model.pt"
    if rich_path.exists():
        ckpt = torch.load(rich_path, map_location="cpu", weights_only=False)
        cfg = HybridConfig(**ckpt["hybrid_cfg"])
        return ckpt, cfg

    raise FileNotFoundError(
        f"Aucun checkpoint pour fold {fold_num} dans {run_dir} "
        f"(attendu {simple.name} ou {rich_path})"
    )


def _build_hybrid_config(seq_len: int, ckpt_cfg: Optional[HybridConfig] = None) -> HybridConfig:
    if ckpt_cfg is not None:
        return ckpt_cfg
    return HybridConfig(
        n_features=N_FEATURES,
        seq_len=seq_len,
        lstm_hidden=128,
        d_model=128,
        nhead=4,
        key_dim=32,
        n_transformer_blocks=1,
        dropout=0.30,
        pos_encoding="sinusoidal",
        bidirectional=False,
    )


def _load_predictions_extremes(run_dir: Path) -> List[dict]:
    """Fusionne predictions_middle.pkl + predictions_extremes.pkl (protocole extrêmes)."""
    middle_path = run_dir / "predictions_middle.pkl"
    extremes_path = run_dir / "predictions_extremes.pkl"
    if not middle_path.exists():
        raise FileNotFoundError(f"{middle_path} introuvable")
    with open(middle_path, "rb") as f:
        preds = list(pickle.load(f))
    if extremes_path.exists():
        with open(extremes_path, "rb") as f:
            preds.extend(pickle.load(f))
    return preds


def _load_single_checkpoint(run_dir: Path, seq_len: int) -> Tuple[HybridLSTMTransformer, np.ndarray, np.ndarray]:
    """Charge model_best.pt + norm_stats.pkl pour le protocole extrêmes."""
    ckpt_path = run_dir / "model_best.pt"
    norm_path = run_dir / "norm_stats.pkl"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} introuvable")
    if not norm_path.exists():
        raise FileNotFoundError(f"{norm_path} introuvable")

    with open(norm_path, "rb") as f:
        norm = pickle.load(f)
    mean, std = norm["mean"], norm["std"]

    cfg = HybridConfig(n_features=N_FEATURES, seq_len=seq_len)
    model = HybridLSTMTransformer(cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    return model, mean, std


def extract_all_extremes(
    run_dir: Path,
    dataset_pkl: Path,
    device: torch.device,
    seq_len: Optional[int] = None,
) -> dict:
    """Extraction per-frame pour un run train-extrêmes (checkpoint unique)."""
    run_dir = Path(run_dir)
    preds = _load_predictions_extremes(run_dir)
    inferred_seq_len = seq_len or 4000

    model, mean, std = _load_single_checkpoint(run_dir, inferred_seq_len)
    model = model.to(device)
    model.eval()

    with open(dataset_pkl, "rb") as f:
        dataset = pickle.load(f)

    entries: List[dict] = []
    for pred in preds:
        pid = str(pred["participant"])
        tid = str(pred["trial_id"])
        key = _trial_key(dataset, pid, tid)
        if key is None:
            print(f"[warn] trial ({pid}, {tid}) absent du dataset — ignoré")
            continue

        T_raw = int(np.asarray(dataset[key]["X"]).shape[0])
        time_norm, frame_scores, n_used = extract_frame_scores_for_trial(
            model, dataset[key]["X"], mean, std, model.cfg.seq_len,
        )
        trial_from_frames = float(np.mean(frame_scores)) if len(frame_scores) else float("nan")
        sublevel = str(pred.get("sublevel", ""))

        entries.append({
            "participant": pid,
            "trial_id": tid,
            "fold": 0,
            "sublevel": sublevel,
            "class_4": pred.get("class_4", _sublevel_to_class4(sublevel)),
            "tier": int(pred.get("tier", 0)),
            "score_true": float(pred.get("score_true", np.nan)),
            "trial_score_pred": float(pred.get("score_pred", np.nan)),
            "trial_score_from_frames": trial_from_frames,
            "n_frames_raw": T_raw,
            "n_frames_used": n_used,
            "time_norm": time_norm,
            "frame_scores": frame_scores,
        })

    return {
        "meta": {
            "model": "HybridLSTMTransformer",
            "protocol": "extremes",
            "run_dir": str(run_dir.resolve()),
            "seq_len": int(model.cfg.seq_len),
            "n_entries": len(entries),
        },
        "entries": entries,
    }


def _trial_key(dataset: dict, participant: str, trial_id: str):
    for k in dataset:
        if str(k[0]) == str(participant) and str(k[1]) == str(trial_id):
            return k
    return None


def detect_protocol(run_dir: Path) -> str:
    """'lopo' si fold_infos.pkl, 'extremes' si model_best.pt, sinon erreur."""
    run_dir = Path(run_dir)
    if (run_dir / "fold_infos.pkl").exists():
        return "lopo"
    if (run_dir / "model_best.pt").exists():
        return "extremes"
    raise FileNotFoundError(
        f"Protocole inconnu dans {run_dir} — attendu fold_infos.pkl (LOPO) "
        f"ou model_best.pt (extrêmes)."
    )


def extract_frame_scores_for_trial(
    model: HybridLSTMTransformer,
    X_raw: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Retourne (time_norm, frame_scores, n_frames_used).

    n_frames_used = min(T_raw, seq_len) ; les frames au-delà du pad sont exclues.
    """
    X = enrich_kinematic_features(np.asarray(X_raw, dtype=np.float32))
    Xn = apply_norm(X, mean, std)
    T_raw = int(Xn.shape[0])

    chunk = crop_sequence(Xn, seq_len, mode="start")
    xb = torch.from_numpy(chunk).unsqueeze(0)
    scores_full = forward_frame_scores(model, xb).squeeze(0).cpu().numpy()

    n_used = min(T_raw, seq_len)
    frame_scores = scores_full[:n_used].astype(np.float32)
    time_norm = np.linspace(0.0, 1.0, n_used, dtype=np.float32)
    return time_norm, frame_scores, n_used


def extract_all(
    run_dir: Path,
    dataset_pkl: Path,
    device: torch.device,
    seq_len: Optional[int] = None,
) -> dict:
    run_dir = Path(run_dir)
    preds = _load_predictions(run_dir)
    fold_infos = _load_fold_infos(run_dir)
    held_to_fold: Dict[str, dict] = {str(f["held"]): f for f in fold_infos}

    with open(dataset_pkl, "rb") as f:
        dataset = pickle.load(f)

    # Déduire seq_len depuis le premier checkpoint riche ou défaut prod
    inferred_seq_len = seq_len or 4000
    for fi in fold_infos:
        try:
            _, cfg = _resolve_checkpoint(run_dir, int(fi["fold"]), fi)
            if cfg is not None:
                inferred_seq_len = cfg.seq_len
                break
        except FileNotFoundError:
            continue

    # Cache modèles par fold
    fold_models: Dict[int, HybridLSTMTransformer] = {}
    fold_norms: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    entries: List[dict] = []

    for pred in preds:
        pid = str(pred["participant"])
        tid = str(pred["trial_id"])
        if pid not in held_to_fold:
            raise KeyError(f"Participant {pid!r} absent de fold_infos.pkl")

        finfo = held_to_fold[pid]
        fold_num = int(finfo["fold"])

        if fold_num not in fold_models:
            state_or_ckpt, ckpt_cfg = _resolve_checkpoint(run_dir, fold_num, finfo)
            cfg = _build_hybrid_config(inferred_seq_len, ckpt_cfg)
            model = HybridLSTMTransformer(cfg).to(device)
            if isinstance(state_or_ckpt, dict) and "state_dict" in state_or_ckpt:
                model.load_state_dict(state_or_ckpt["state_dict"])
                mean, std = state_or_ckpt["norm_mean"], state_or_ckpt["norm_std"]
            else:
                model.load_state_dict(state_or_ckpt)
                mean, std = finfo["norm_mean"], finfo["norm_std"]
            model.eval()
            fold_models[fold_num] = model
            fold_norms[fold_num] = (mean, std)

        model = fold_models[fold_num]
        mean, std = fold_norms[fold_num]

        key = _trial_key(dataset, pid, tid)
        if key is None:
            print(f"[warn] trial ({pid}, {tid}) absent du dataset — ignoré")
            continue

        T_raw = int(np.asarray(dataset[key]["X"]).shape[0])
        time_norm, frame_scores, n_used = extract_frame_scores_for_trial(
            model, dataset[key]["X"], mean, std, model.cfg.seq_len,
        )

        trial_from_frames = float(np.mean(frame_scores)) if len(frame_scores) else float("nan")

        sublevel = str(pred.get("sublevel", ""))
        entries.append({
            "participant": pid,
            "trial_id": tid,
            "fold": fold_num,
            "sublevel": sublevel,
            "class_4": _sublevel_to_class4(sublevel),
            "tier": int(pred.get("tier", 0)),
            "score_true": float(pred.get("score_true", np.nan)),
            "trial_score_pred": float(pred.get("score_pred", np.nan)),
            "trial_score_from_frames": trial_from_frames,
            "n_frames_raw": T_raw,
            "n_frames_used": n_used,
            "time_norm": time_norm,
            "frame_scores": frame_scores,
        })

    payload = {
        "meta": {
            "model": "HybridLSTMTransformer",
            "run_dir": str(run_dir.resolve()),
            "seq_len": int(fold_models[next(iter(fold_models))].cfg.seq_len) if fold_models else inferred_seq_len,
            "n_entries": len(entries),
        },
        "entries": entries,
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extrait les scores par frame (Hybrid LSTM-Transformer, LOPO ou extrêmes).",
    )
    ap.add_argument(
        "--run-dir", type=Path, required=True,
        help="Dossier du run (LOPO: predictions.pkl + fold_* ; extrêmes: model_best.pt)",
    )
    ap.add_argument(
        "--protocol", choices=("auto", "lopo", "extremes"), default="auto",
        help="Protocole d'extraction (défaut: auto-détection)",
    )
    ap.add_argument("--pkl", type=Path, default=Path("data/continuous_per_trial.pkl"))
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Chemin de sortie (défaut: <run-dir>/frame_predictions.pkl)",
    )
    ap.add_argument("--seq-len", type=int, default=None, help="Override seq_len si absent du ckpt")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    out_path = args.out or (args.run_dir / "frame_predictions.pkl")

    print(f"[extract] run_dir={args.run_dir} device={device}")
    protocol = args.protocol if args.protocol != "auto" else detect_protocol(args.run_dir)
    print(f"[extract] protocol={protocol}")

    if protocol == "lopo":
        payload = extract_all(args.run_dir, args.pkl, device, seq_len=args.seq_len)
    else:
        payload = extract_all_extremes(args.run_dir, args.pkl, device, seq_len=args.seq_len)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    meta = payload["meta"]
    print(
        f"[done] {meta['n_entries']} trials -> {out_path} "
        f"(seq_len={meta['seq_len']})"
    )


if __name__ == "__main__":
    main()
