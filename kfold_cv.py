"""
kfold_cv.py
===========
Cross-validation K-fold PAR PARTICIPANT pour le scorer d'expertise (baseline LSTM).

Répond directement à la question méthodologique du prof :
    « Combien de runs as-tu faits ? Le 0.35 est-il UN seul run ou une moyenne ? »

Principe (cf. réunion 26 mai) :
    1. On ne garde que les classes EXTRÊMES (0 = Student, 8 = Staff) pour train/val.
    2. On découpe les PARTICIPANTS (pas les fenêtres) en K folds → aucune fuite.
    3. Pour chaque fold :
         - train sur K-1 folds, validation sur le fold restant (extrêmes, jamais vus)
         - évaluation BLIND sur les classes 1-7 (intermédiaires, jamais entraînées)
    4. On rapporte Pearson_val et Pearson_blind : moyenne ± écart-type sur les K folds.

Sortie :
    - tableau console fold par fold
    - results/kfold_cv/kfold_results.csv  (1 ligne par fold + ligne agrégée)

Ce script est volontairement SANS MAE (baseline pure). L'ablation avec MAE est
traitée séparément une fois la compatibilité 4ch/10ch résolue.

Usage (depuis le dossier ICEMS-main) :
    python kfold_cv.py
    python kfold_cv.py --folds 5 --epochs 25 --lstm-units 64
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# UTF-8 pour les consoles Windows (évite les crash sur ±, ², →, etc.).
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# On réutilise UNIQUEMENT les helpers de normalisation de continuous_scorer
# (import sans effet de bord : son main() est protégé par __main__).
sys.path.insert(0, str(Path(__file__).parent / "src"))
from continuous_scorer import compute_train_norm_stats, apply_norm  # noqa: E402

# ── Hyperparamètres (identiques à la baseline B.4 pour comparabilité) ───────
N_CONTEXT  = 300      # 30 s à 10 Hz : taille de la fenêtre glissante
HOP        = 50       # 5 s : décalage entre deux fenêtres
N_FEATURES = 10       # 3 instruments × 3 cinématiques + valid_ratio
LSTM_UNITS = 64
EPOCHS     = 25
BATCH_SIZE = 64
SEED       = 42


# ── Helpers fenêtres (redéfinis ici pour éviter tout import à effet de bord) ─
def extract_sliding_windows(X: np.ndarray, N: int = N_CONTEXT, hop: int = HOP):
    """Découpe une séquence (T, F) en fenêtres glissantes (n_win, N, F)."""
    T = X.shape[0]
    if T < N:
        return None, []
    starts = list(range(0, T - N + 1, hop))
    windows = np.stack([X[s:s + N] for s in starts], axis=0)
    return windows, starts


def build_window_dataset(trials: Dict[Tuple[str, str], dict]):
    """Concatène les fenêtres de tous les trials (X déjà normalisé en amont)."""
    Xs, ys, y9s = [], [], []
    for rec in trials.values():
        windows, _ = extract_sliding_windows(rec["X"])
        if windows is None:
            continue
        n = windows.shape[0]
        Xs.append(windows)
        ys.append(np.full(n, rec["y_reg"], dtype=np.float32))
        y9s.append(np.full(n, rec["y9"], dtype=np.int32))
    if not Xs:
        return None, None, None
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(y9s))


def balanced_sample_weights(y9: np.ndarray):
    """Poids inversement proportionnels à la fréquence de classe (style sklearn)."""
    counts = Counter(y9.tolist())
    n_total, n_classes = len(y9), len(counts)
    wpc = {c: n_total / (n_classes * n_c) for c, n_c in counts.items()}
    return np.array([wpc[int(y)] for y in y9], dtype=np.float32), wpc


# ── Modèle (architecture identique à la baseline B.4) ───────────────────────
def build_window_scorer(keras, layers):
    inp = keras.Input(shape=(N_CONTEXT, N_FEATURES), name="window_input")
    x = layers.LSTM(LSTM_UNITS, return_sequences=False, name="lstm")(inp)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="tanh", name="score")(x)
    return keras.Model(inp, out, name="window_scorer")


def per_trial_median_predictions(model, trials, mean, std):
    """Prédit un score agrégé (médiane des fenêtres) par trial."""
    preds = {}
    for key, rec in trials.items():
        X = apply_norm(rec["X"], mean, std)
        windows, _ = extract_sliding_windows(X)
        if windows is None:
            continue
        scores = model.predict(windows, batch_size=BATCH_SIZE, verbose=0).flatten()
        preds[key] = {"y_reg": rec["y_reg"], "y9": rec["y9"],
                      "score": float(np.median(scores))}
    return preds


def pearson_and_r2(preds) -> Tuple[float, float, int]:
    if len(preds) < 2:
        return float("nan"), float("nan"), len(preds)
    y_true = np.array([p["y_reg"] for p in preds.values()])
    y_pred = np.array([p["score"] for p in preds.values()])
    if y_pred.std() < 1e-9:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
    return pearson, 1.0 - ss_res / ss_tot, len(preds)


# ── Boucle K-fold par participant ───────────────────────────────────────────
def participants_of(trials: Dict[Tuple[str, str], dict]) -> List[str]:
    return sorted({pid for (pid, _tid) in trials.keys()})


def run_kfold(dataset_path: Path, out_dir: Path, folds: int, epochs: int,
              lstm_units: int):
    global LSTM_UNITS, EPOCHS
    LSTM_UNITS, EPOCHS = lstm_units, epochs

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.model_selection import StratifiedGroupKFold

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    extremes = {k: v for k, v in dataset.items() if v["y9"] in (0, 8)}
    blind    = {k: v for k, v in dataset.items() if v["y9"] in (1, 2, 3, 4, 5, 6, 7)}
    parts = participants_of(extremes)

    print("=" * 72)
    print(f" K-FOLD CROSS-VALIDATION ({folds} folds) — split PAR PARTICIPANT")
    print("=" * 72)
    print(f"  Trials extrêmes (0+8) : {len(extremes)}  | participants : {len(parts)}")
    print(f"  Trials blind   (1-7)  : {len(blind)}")
    print(f"  Hyperparams : N_CONTEXT={N_CONTEXT}, HOP={HOP}, "
          f"LSTM={LSTM_UNITS}, epochs={EPOCHS}, batch={BATCH_SIZE}\n")

    if len(parts) < folds:
        raise ValueError(f"Pas assez de participants ({len(parts)}) pour {folds} folds.")

    # StratifiedGroupKFold : ne coupe jamais un participant (group) ET garantit
    # les DEUX classes (0 et 8) dans chaque pli de validation → plus de Pearson nan.
    trial_keys = list(extremes.keys())
    groups = [k[0] for k in trial_keys]                  # participant
    labels = [extremes[k]["y9"] for k in trial_keys]     # classe 0 ou 8
    sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = []

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(trial_keys, labels, groups), start=1):
        tf.keras.utils.set_random_seed(SEED + fold)

        tr_keys = [trial_keys[i] for i in tr_idx]
        va_keys = [trial_keys[i] for i in va_idx]
        tr_pids = {k[0] for k in tr_keys}
        va_pids = {k[0] for k in va_keys}
        assert tr_pids.isdisjoint(va_pids), "Fuite : participant des 2 côtés !"

        train_trials = {k: extremes[k] for k in tr_keys}
        val_trials   = {k: extremes[k] for k in va_keys}

        # Normalisation calculée SUR LE TRAIN UNIQUEMENT (jamais sur val/blind).
        mean, std = compute_train_norm_stats(train_trials, n_features=N_FEATURES)
        train_norm = {k: {**v, "X": apply_norm(v["X"], mean, std)}
                      for k, v in train_trials.items()}

        X_tr, y_tr, y9_tr = build_window_dataset(train_norm)
        if X_tr is None:
            print(f"[Fold {fold}] aucune fenêtre (trials trop courts) — sauté.")
            continue
        sw, _ = balanced_sample_weights(y9_tr)

        model = build_window_scorer(keras, layers)
        model.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss=keras.losses.Huber(delta=1.0), metrics=["mae"])
        model.fit(X_tr, y_tr, sample_weight=sw, epochs=EPOCHS,
                  batch_size=BATCH_SIZE, verbose=0)

        # per_trial_median_predictions normalise en interne → on lui passe le RAW.
        p_val   = pearson_and_r2(per_trial_median_predictions(model, val_trials, mean, std))
        p_blind = pearson_and_r2(per_trial_median_predictions(model, blind, mean, std))

        n_tr0 = sum(1 for v in train_trials.values() if v["y9"] == 0)
        n_tr8 = sum(1 for v in train_trials.values() if v["y9"] == 8)
        rows.append({
            "fold": fold, "n_train_part": len(tr_pids), "n_val_part": len(va_pids),
            "n_train_trials": len(train_trials), "train_c0": n_tr0, "train_c8": n_tr8,
            "pearson_val": p_val[0], "r2_val": p_val[1], "n_val": p_val[2],
            "pearson_blind": p_blind[0], "r2_blind": p_blind[1], "n_blind": p_blind[2],
        })
        print(f"[Fold {fold}] val(extrêmes): Pearson={p_val[0]:+.3f} R²={p_val[1]:+.3f} "
              f"(n={p_val[2]})  |  blind(1-7): Pearson={p_blind[0]:+.3f} "
              f"R²={p_blind[1]:+.3f} (n={p_blind[2]})")

    # ── Agrégation ──────────────────────────────────────────────────────────
    pv = np.array([r["pearson_val"]   for r in rows], dtype=float)
    pb = np.array([r["pearson_blind"] for r in rows], dtype=float)
    print("\n" + "=" * 72)
    print(" RÉSULTAT AGRÉGÉ")
    print("=" * 72)
    print(f"  Pearson VAL (extrêmes) : {np.nanmean(pv):+.3f} ± {np.nanstd(pv):.3f}")
    print(f"  Pearson BLIND (1-7)    : {np.nanmean(pb):+.3f} ± {np.nanstd(pb):.3f}")
    print(f"  → C'est CE chiffre (moyenne ± std sur {len(rows)} folds) à présenter "
          f"au prof, PAS un run unique.")

    # ── Sauvegarde CSV ────────────────────────────────────────────────────────
    csv_path = out_dir / "kfold_results.csv"
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        agg = {"fold": "MEAN±STD",
               "pearson_val": f"{np.nanmean(pv):.3f}±{np.nanstd(pv):.3f}",
               "pearson_blind": f"{np.nanmean(pb):.3f}±{np.nanstd(pb):.3f}"}
        df = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)
        df.to_csv(csv_path, index=False)
    except ImportError:
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\n✅ Résultats sauvegardés : {csv_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="K-fold CV par participant (baseline LSTM).")
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/continuous_per_trial.pkl"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/kfold_cv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lstm-units", type=int, default=LSTM_UNITS)
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {args.dataset}\n"
            f"Génère-le d'abord : python src/build_continuous_dataset.py")

    run_kfold(args.dataset, args.out_dir, args.folds, args.epochs, args.lstm_units)


if __name__ == "__main__":
    main()
