"""
extremes_protocol.py
====================
Protocole "extrêmes" EVICEMS : entraîner uniquement sur les niveaux extrêmes
(Medical Student → -1.0, Staff → +1.0), puis mesurer la VALIDITÉ PRÉDICTIVE sur
les niveaux intermédiaires (PGY1…PGY6, Fellow) JAMAIS vus à l'entraînement.

Choix de seq_len
----------------
La longueur temporelle est adaptée au modèle :
  1. Contrainte d'architecture : Hybrid1EVICEMS possède un pos_embedding
     paramétrique de forme (1, seq_len, d_model) — la longueur doit être constante.
  2. Couverture empirique : p10 des durées réelles = 2 250 frames → 90 % des trials
     ne subissent aucun padding (données mesurées sur continuous_per_trial.pkl).
     À 800 frames, seuls ~50 % de la durée médiane (~4 020 f) auraient été capturés.
  3. Capture de la phase "Middle" : un crop centré sur 2 000 frames inclut
     Early-fin + Middle complet + Late-début, soit la zone la plus discriminante
     pour l'évaluation de l'expertise chirurgicale.
  4. Compatibilité mémoire : O(T²) pour l'attention = 2000² = 4 M cases par tête,
     raisonnable sur GPU Narval (A100 80 GB).
Les séquences plus longues sont recadrées au centre, les plus courtes complétées
par des zéros à gauche (left-padding), puis normalisées (z-score TRAIN).

Contraintes respectées :
  - Aucune modification de CausalGRUScorer / run_corrected_lopo / run_lopo_corrected.py
  - Seeds fixes (numpy, torch, DataLoader generator)
  - Par défaut, tous les participants extrêmes sont utilisés en TRAIN ; l'ancien
    split GroupShuffleSplit 70/15/15 reste disponible via use_val_split=True.
"""
from __future__ import annotations

import json
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

# Imports projet — compatibles exécution directe ET import comme module.
try:
    from config import normalize_sublevel
    from icems_strategy import build_scorer
    from models.hybrid1_evicems import Hybrid1EVICEMS
except ImportError:  # pragma: no cover
    from src.config import normalize_sublevel
    from src.icems_strategy import build_scorer
    from src.models.hybrid1_evicems import Hybrid1EVICEMS

# ─── Constantes du protocole ─────────────────────────────────────────────────
TRAIN_SUBLEVELS = {"ms", "staff"}
MIDDLE_SUBLEVELS = {"pgy1", "pgy2", "pgy3", "pgy4", "pgy5", "pgy6", "fellow"}
SUBLEVEL_TARGET = {"ms": -1.0, "staff": +1.0}

# Année de formation (validité prédictive) : pgy_n → n, fellow → 7.
YEAR_BY_SUBLEVEL = {
    "pgy1": 1, "pgy2": 2, "pgy3": 3, "pgy4": 4,
    "pgy5": 5, "pgy6": 6, "fellow": 7,
}

DEFAULT_SEQ_LEN = 2000
# dropout = 0.30 pour les deux modèles (CausalGRU et Hybrid1EVICEMS).
# Choix méthodologique délibéré : la comparaison exige des conditions identiques.
# Note : CausalGRU utilise 0.15 dans le pipeline LOPO existant (step_B_classification.py
# TRAIN_DROPOUT = 0.15) — cette divergence est intentionnelle et documentée. Les résultats
# de ce protocole ne sont comparables qu'entre eux, pas aux métriques LOPO publiées.
DEFAULT_DROPOUT = 0.30


# ─── Chargement / normalisation des enregistrements ──────────────────────────
def _record_features(rec: dict) -> np.ndarray:
    """
    Extrait X (T, n_features) depuis un enregistrement.

    Formats supportés :
      A) champ "X" shape (T, F)          -> retourné directement
      B) champ "data" shape (C, T)        -> transposé en (T, C)
         Si C == 8 et rows 0-1 = entiers  -> data[2:, :].T  (6 features)
         Si C != 8                         -> data.T  (toutes les rows = features)
    """
    if "X" in rec:
        return np.asarray(rec["X"], dtype=np.float32)

    if "data" in rec:
        data = np.asarray(rec["data"], dtype=np.float32)
        # Détection automatique du format avec labels embarqués.
        # Convention augmented_v4 : si C==8, rows 0-1 sont des labels entiers (y4, y9).
        if data.shape[0] == 8:
            rows_are_labels = (
                np.all(data[0] == data[0].astype(np.int32)) and
                np.all(data[1] == data[1].astype(np.int32))
            )
            if rows_are_labels:
                return data[2:, :].T.astype(np.float32)  # (T, 6)
        return data.T.astype(np.float32)  # (T, C) cas général

    raise KeyError(
        f"Enregistrement sans champ 'X' ni 'data'. "
        f"Clés disponibles : {list(rec.keys())}"
    )


def _load_records(pkl_path: Path) -> List[dict]:
    """Charge le pkl et renvoie une liste normalisée.

    Chaque élément : {participant, trial, sublevel (canonique), X (T, n_features)}.
    Supporte le format dict {(pid, trial): rec} ET le format liste [rec, ...].
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    items: List[Tuple] = []
    if isinstance(raw, dict):
        for key, rec in raw.items():
            pid = str(rec.get("participant",
                      rec.get("name",
                      key[0] if isinstance(key, tuple) else str(key))))
            trial = str(rec.get("trial", key[1] if isinstance(key, tuple) and len(key) > 1 else "0"))
            items.append((pid, trial, rec))
    elif isinstance(raw, list):
        for i, rec in enumerate(raw):
            pid = str(rec.get("participant",
                      rec.get("name",
                      rec.get("id", f"p{i:03d}"))))
            trial = str(rec.get("trial", rec.get("name", "0")))
            items.append((pid, trial, rec))
    else:
        raise TypeError(f"Format pkl non supporté : {type(raw)}")

    records: List[dict] = []
    for pid, trial, rec in items:
        if rec.get("is_augmented", False):
            continue
        raw_sl = rec.get("sublevel") or rec.get("level", "")
        sublevel = normalize_sublevel(str(raw_sl))
        X = _record_features(rec)
        if X.ndim != 2 or X.shape[0] < 2:
            continue
        records.append({"participant": pid, "trial": trial, "sublevel": sublevel, "X": X})

    # Validation post-chargement
    n_ms    = sum(1 for r in records if r["sublevel"] == "ms")
    n_staff = sum(1 for r in records if r["sublevel"] == "staff")
    n_mid   = sum(1 for r in records if r["sublevel"] not in {"ms", "staff"})
    n_total = len(records)

    print(f"[LOAD] {n_total} records chargés")
    print(f"[LOAD]   ms={n_ms}  staff={n_staff}  milieu={n_mid}")
    print(f"[LOAD]   features shape exemple : {records[0]['X'].shape if records else 'N/A'}")

    assert n_ms > 0,    "ERREUR : aucun trial 'ms' trouvé — vérifier normalize_sublevel"
    assert n_staff > 0, "ERREUR : aucun trial 'staff' trouvé — vérifier normalize_sublevel"

    if records:
        X0 = records[0]["X"]
        assert X0.shape[1] >= 4, \
            f"Trop peu de features ({X0.shape[1]}) — vérifier _record_features"
        col0_unique = np.unique(X0[:, 0])
        assert len(col0_unique) > 5, \
            f"Col 0 ressemble a un label ({col0_unique}) — data[2:] manquant ?"

    return records


def _fit_length(
    X: np.ndarray,
    seq_len: int,
    crop_mode: str = "center",
) -> Tuple[np.ndarray, np.ndarray]:
    """Recadre/complète X (T, F) à seq_len ; left-pad zéros si court.

    Parameters
    ----------
    crop_mode : "center" | "start"
        "center" (défaut) — crop centré : start = (T - seq_len) // 2.
            Justification : la phase Middle (~33–66 %) est la plus discriminante
            pour l'expertise chirurgicale ; un crop centré capture fin-Early +
            Middle complet + début-Late.
        "start" — crop depuis le début : start = 0, frames 0:seq_len.
            Recommandé pour les modèles CAUSAUX (GRU gauche-droite) dont les
            états cachés dépendent du contexte initial : démarrer en plein milieu
            du trial introduit un biais de contexte manquant.

    Returns
    -------
    Xp : (seq_len, F)   séquence ajustée
    mask : (seq_len,)   1 = frame réelle, 0 = frame paddée
    """
    if crop_mode not in ("center", "start"):
        raise ValueError(f"crop_mode doit être 'center' ou 'start', reçu : {crop_mode!r}")

    T, F = X.shape
    if T == seq_len:
        return X.astype(np.float32), np.ones(seq_len, dtype=np.float32)
    if T > seq_len:
        start = 0 if crop_mode == "start" else (T - seq_len) // 2
        return X[start : start + seq_len].astype(np.float32), np.ones(seq_len, dtype=np.float32)
    # T < seq_len : padding à gauche (indépendant du crop_mode)
    pad = seq_len - T
    Xp = np.zeros((seq_len, F), dtype=np.float32)
    Xp[pad:] = X
    mask = np.zeros(seq_len, dtype=np.float32)
    mask[pad:] = 1.0
    return Xp, mask


def _compute_norm_stats(
    train_records: List[dict],
    seq_len: int,
    crop_mode: str = "center",
) -> Tuple[np.ndarray, np.ndarray]:
    """Moyenne/écart-type z-score sur les frames RÉELLES du TRAIN uniquement."""
    real_frames = []
    for r in train_records:
        Xp, mask = _fit_length(r["X"], seq_len, crop_mode=crop_mode)
        real_frames.append(Xp[mask > 0])
    allf = np.concatenate(real_frames, axis=0)
    mean = allf.mean(axis=0).astype(np.float32)
    std = (allf.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std


def _build_tensors(
    records: List[dict],
    mean: np.ndarray,
    std: np.ndarray,
    seq_len: int,
    crop_mode: str = "center",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construit (X, mask, y) normalisés pour une liste d'enregistrements."""
    Xs, masks, ys = [], [], []
    for r in records:
        Xp, mask = _fit_length(r["X"], seq_len, crop_mode=crop_mode)
        Xp = (Xp - mean) / std
        Xs.append(Xp)
        masks.append(mask)
        ys.append(float(r.get("y", np.nan)))
    X = torch.from_numpy(np.stack(Xs)).float()
    m = torch.from_numpy(np.stack(masks)).float()
    y = torch.tensor(ys, dtype=torch.float32)
    return X, m, y


# ─── Forward unifié (gère CausalGRU per-frame ET Hybrid1 per-trial) ──────────
def _forward_trial_scores(model: nn.Module, xb: torch.Tensor, mb: torch.Tensor) -> torch.Tensor:
    """Renvoie un score trial-level (B,) quel que soit le type de modèle.

    Dispatch par isinstance (robuste, indépendant de l'introspection) :
    - Hybrid1EVICEMS : forward(x) → (B, 1), retourné directement sans re-averaging.
    - Tout autre modèle (CausalGRU, HybridLSTMTransformer...) : forward(x, mask) →
      scores par frame (B, T), agrégés par moyenne pondérée sur les frames valides
      uniquement (frames paddées exclues via mb = 0).
    """
    if isinstance(model, Hybrid1EVICEMS):
        out = model(xb)   # (B, 1)
        return out.squeeze(-1)
    # Agrégation pondérée par le masque de validité : exclut les frames paddées.
    out = model(xb, mb)   # (B, T)
    w = mb.clamp(min=0)
    denom = w.sum(dim=1).clamp(min=1e-6)
    return (out * w).sum(dim=1) / denom


# ─── Split groupé 70/15/15 ───────────────────────────────────────────────────
def _grouped_split(
    records: List[dict], seed: int
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split 70/15/15 GROUPÉ PAR PARTICIPANT via GroupShuffleSplit (random_state=seed).

    Assertion : aucun participant partagé entre train / val / test.
    """
    groups = np.array([r["participant"] for r in records])
    idx = np.arange(len(records))

    # Étape 1 : 70 % train / 30 % temp
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss1.split(idx, groups=groups))

    # Étape 2 : temp → 50 % val / 50 % test  (= 15 % / 15 % du total)
    temp_groups = groups[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    rel_val, rel_test = next(gss2.split(temp_idx, groups=temp_groups))
    val_idx = temp_idx[rel_val]
    test_idx = temp_idx[rel_test]

    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    test = [records[i] for i in test_idx]

    # Assertion stricte de non-chevauchement (unité = participant)
    p_train = {r["participant"] for r in train}
    p_val = {r["participant"] for r in val}
    p_test = {r["participant"] for r in test}
    assert p_train.isdisjoint(p_val), f"Chevauchement train/val : {p_train & p_val}"
    assert p_train.isdisjoint(p_test), f"Chevauchement train/test : {p_train & p_test}"
    assert p_val.isdisjoint(p_test), f"Chevauchement val/test : {p_val & p_test}"

    return train, val, test


# ─── Entraînement ────────────────────────────────────────────────────────────
def _train_model(
    model: nn.Module,
    train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    seed: int,
) -> Tuple[nn.Module, dict]:
    """Adam + MSELoss.

    Si val_data est fourni : ancien comportement avec early stopping sur val_loss.
    Sinon : entraînement à epochs fixes, sans early stopping. C'est le mode par
    défaut du protocole extrêmes, car réserver 3-4 participants extrêmes pour une
    validation interne rend le pool TRAIN trop petit et instable.
    """
    model = model.to(device)
    if isinstance(model, Hybrid1EVICEMS):
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    Xtr, Mtr, Ytr = train_data
    if val_data is not None:
        Xva, Mva, Yva = val_data

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(Xtr, Mtr, Ytr),
        batch_size=batch_size, shuffle=True, generator=g,
    )

    best_val, best_state, wait = float("inf"), None, 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "early_stopping": val_data is not None,
        "state_source": "best_val" if val_data is not None else "last_epoch",
    }

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for xb, mb, yb in loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            opt.zero_grad()
            pred = _forward_trial_scores(model, xb, mb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history["train_loss"].append(train_loss)

        if val_data is None:
            if (epoch + 1) == 1 or (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
                print(f"[TRAIN] epoch {epoch + 1:03d}/{epochs} train_loss={train_loss:.6f}")
            continue

        model.eval()
        with torch.no_grad():
            pred_va = _forward_trial_scores(model, Xva.to(device), Mva.to(device))
            val_loss = criterion(pred_va, Yva.to(device)).item()

        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val, best_state, wait = val_loss, deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                history["stopped_epoch"] = epoch + 1
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        # Mode sans validation : le modèle courant est celui de la dernière epoch.
        history["stopped_epoch"] = epochs
    history["best_val_loss"] = best_val if val_data is not None else None
    return model, history


# ─── Protocole principal ─────────────────────────────────────────────────────
def run_extremes_protocol(
    pkl_path,
    model_name: str,
    seq_len: int = DEFAULT_SEQ_LEN,
    seed: int = 42,
    output_dir="results/extremes",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 15,  # val set = 3 participants → loss bruitée → patience longue
    dropout: float = DEFAULT_DROPOUT,
    augment: bool = True,
    global_multiplier: int = 3,
    device: Optional[torch.device] = None,
    crop_mode: Optional[str] = None,
    use_val_split: bool = False,
    auto_seq_len: bool = False,
) -> dict:
    """Exécute le protocole extrêmes pour un modèle donné.

    Parameters
    ----------
    crop_mode : "center" | "start" | None
        Stratégie de recadrage quand T > seq_len.
        Si None (défaut), dérivé automatiquement depuis model_name :
          - "causal_gru"     → "start"   (GRU causal : contexte initial requis)
          - tout autre modèle → "center"  (Hybrid1, etc. : phase Middle optimale)
    use_val_split : bool
        False (défaut) : tous les participants extrêmes sont utilisés en TRAIN,
        sans validation interne ni early stopping. True restaure l'ancien split
        GroupShuffleSplit 70/15/15.
    auto_seq_len : bool
        True (défaut) : si model_name == "causal_gru" et seq_len == 2000,
        utilise automatiquement seq_len=4000.

    Returns
    -------
    dict : {
        "model_name", "json_path", "results" (liste par participant test),
        "train_internal_mae", "history", "n_train/val/test", "seq_len", "seed",
        "crop_mode", "use_val_split"
    }
    """
    pkl_path = Path(pkl_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dérivation automatique du crop_mode si non fourni explicitement.
    if crop_mode is None:
        crop_mode = "start" if model_name == "causal_gru" else "center"

    if auto_seq_len:
        if model_name == "causal_gru":
            seq_len = 4000
            crop_mode = "start"
        else:
            seq_len = 2000
            crop_mode = "center"
        print(f"[PROTOCOL] auto_seq_len → {model_name}: seq_len={seq_len} crop={crop_mode}")

    model_label = "CausalGRU" if model_name == "causal_gru" else "Hybrid1"
    print(f"[PROTOCOL] {model_label:<11}: seq_len={seq_len}, crop_mode={crop_mode}, epochs={epochs}")

    # Seeds globaux
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. Chargement
    records = _load_records(pkl_path)

    # 2. Séparation des pools
    pool_train = [r for r in records if r["sublevel"] in TRAIN_SUBLEVELS]
    pool_test = [r for r in records if r["sublevel"] in MIDDLE_SUBLEVELS]
    if not pool_train:
        raise ValueError("POOL_TRAIN vide : aucun trial ms/staff trouvé.")
    if not pool_test:
        raise ValueError("POOL_TEST vide : aucun trial intermédiaire trouvé.")
    for r in pool_train:
        r["y"] = SUBLEVEL_TARGET[r["sublevel"]]

    n_features = pool_train[0]["X"].shape[1]

    train_participants_all = {r["participant"] for r in pool_train}
    test_participants_middle = {r["participant"] for r in pool_test}

    if use_val_split:
        # Rétrocompatibilité : ancien protocole 70/15/15 sur les extrêmes.
        train_recs, val_recs, test_train_recs = _grouped_split(pool_train, seed)
        print(
            f"[SPLIT] Ancien protocole 70/15/15 : train={len({r['participant'] for r in train_recs})} "
            f"val={len({r['participant'] for r in val_recs})} "
            f"test_interne={len({r['participant'] for r in test_train_recs})} participants"
        )
    else:
        # Nouveau protocole : tous les participants extrêmes en TRAIN.
        # Pool extrêmes trop petit pour réserver une validation interne stable ;
        # la validité externe est mesurée uniquement sur le milieu.
        train_recs = [r for r in pool_train]
        val_recs = []
        test_train_recs = []
        print(
            f"[SPLIT] TRAIN : {len(set(r['participant'] for r in train_recs))} "
            f"participants (tous extrêmes), {len(train_recs)} trials réels"
        )
        print("[SPLIT] val=0 test_interne=0 — epochs fixe sans early stopping")

    print(f"[TEST]  Pool milieu : {len(test_participants_middle)} participants (pgy1...fellow)")

    # ── Augmentation TRAIN (Option D) ─────────────────────────────────────
    # Appliquée APRÈS le split (pas de leakage) et AVANT la normalisation
    # (les stats z-score sont calculées sur les données augmentées).
    # N'agit QUE sur train_recs : val_recs / test_train_recs / pool_test intacts.
    if augment:
        try:
            from eval.augment_extremes import augment_pool
        except ImportError:  # pragma: no cover
            from src.eval.augment_extremes import augment_pool
        train_recs = augment_pool(
            train_recs,
            seed=seed,
            global_multiplier=global_multiplier,
            jitter_sigma_eq=0.015,
            jitter_sigma_global=0.020,
            warp_sigma=0.05,
            warp_knots=3,
        )
        n_aug_ms = sum(1 for r in train_recs if r["sublevel"] == "ms")
        n_aug_staff = sum(1 for r in train_recs if r["sublevel"] == "staff")
        print(
            f"[AUG]   Après augmentation ×{global_multiplier} : "
            f"{n_aug_ms} trials ms + {n_aug_staff} trials staff = {len(train_recs)} total"
        )
    # ── Fin augmentation ───────────────────────────────────────────────────

    print(
        f"n_train={len(train_recs)} n_val={len(val_recs)} "
        f"n_test_internal={len(test_train_recs)} n_middle={len(test_participants_middle)}"
    )

    # 4 + 5. Normalisation z-score (mean/std sur TRAIN uniquement)
    mean, std = _compute_norm_stats(train_recs, seq_len, crop_mode=crop_mode)
    train_data = _build_tensors(train_recs, mean, std, seq_len, crop_mode=crop_mode)
    val_data = (
        _build_tensors(val_recs, mean, std, seq_len, crop_mode=crop_mode)
        if val_recs else None
    )
    test_train_data = (
        _build_tensors(test_train_recs, mean, std, seq_len, crop_mode=crop_mode)
        if test_train_recs else None
    )

    # 6. Construction + entraînement
    model = build_scorer(model_name, n_features=n_features, dropout=dropout, seq_len=seq_len)
    model, history = _train_model(
        model, train_data, val_data, device,
        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, seed=seed,
    )

    # Sanity : avec use_val_split=False, MAE sur TRAIN extrêmes ; avec l'ancien
    # protocole, MAE sur le test interne ms/staff held-out.
    model.eval()
    with torch.no_grad():
        sanity_data = test_train_data if test_train_data is not None else train_data
        Xtt, Mtt, Ytt = sanity_data
        # Forward par mini-batchs : avec use_val_split=False le pool sanity = tout
        # le TRAIN augmenté (plusieurs centaines de trials). L'attention O(T²) de
        # Hybrid1 saturerait la mémoire si tout passait en un seul forward.
        preds = []
        for i in range(0, Xtt.shape[0], batch_size):
            xb = Xtt[i : i + batch_size].to(device)
            mb = Mtt[i : i + batch_size].to(device)
            preds.append(_forward_trial_scores(model, xb, mb).cpu().numpy())
        pred_tt = np.concatenate(preds) if preds else np.zeros(0, dtype=np.float32)
        train_internal_mae = float(np.mean(np.abs(pred_tt - Ytt.numpy())))

    # 7. Prédictions POOL_TEST : médiane par participant
    by_participant: Dict[str, List[float]] = {}
    sublevel_by_participant: Dict[str, str] = {}
    with torch.no_grad():
        for r in pool_test:
            Xp, mask = _fit_length(r["X"], seq_len, crop_mode=crop_mode)
            Xp = (Xp - mean) / std
            xb = torch.from_numpy(Xp).float().unsqueeze(0).to(device)
            mb = torch.from_numpy(mask).float().unsqueeze(0).to(device)
            score = float(_forward_trial_scores(model, xb, mb).cpu().numpy()[0])
            by_participant.setdefault(r["participant"], []).append(score)
            sublevel_by_participant[r["participant"]] = r["sublevel"]

    # 8. Sauvegarde JSON (un objet par participant test)
    results = []
    for pid, scores in by_participant.items():
        sl = sublevel_by_participant[pid]
        results.append({
            "participant_id": pid,
            "sublevel": sl,
            "pred_score": float(np.median(scores)),
            "year_of_training": YEAR_BY_SUBLEVEL[sl],
            "n_trials": len(scores),
        })
    results.sort(key=lambda d: (d["year_of_training"], d["participant_id"]))

    json_path = output_dir / f"results_{model_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return {
        "model_name": model_name,
        "json_path": str(json_path),
        "results": results,
        "train_internal_mae": train_internal_mae,
        "best_val_loss": history.get("best_val_loss"),
        "n_train": len(train_recs),
        "n_val": len(val_recs),
        "n_test_internal": len(test_train_recs),
        "n_test_middle_participants": len(results),
        "seq_len": seq_len,
        "seed": seed,
        "crop_mode": crop_mode,
        "use_val_split": use_val_split,
        "auto_seq_len": auto_seq_len,
    }
