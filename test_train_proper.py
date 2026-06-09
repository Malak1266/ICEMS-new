"""B.4 — Entraînement avec split PAR PARTICIPANT + validation aveugle.

Différences avec B.3 :
1. Split train/val par PARTICIPANT (pas par fenêtre) → pas de fuite.
2. Set blind séparé : classes 1-7, jamais vu pendant le training.
3. Plus d'epochs (par défaut 25).
4. Sauvegarde de l'historique pour tracer les courbes.
"""
import sys
import pickle
import json
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, "src")
from continuous_scorer import compute_train_norm_stats, apply_norm
from test_build_windows import (
    extract_sliding_windows,
    build_window_dataset,
    compute_balanced_sample_weights,
    N_CONTEXT,
)

# ────────────────────────────────────────────────────────────────────────────
N_FEATURES   = 10
LSTM_UNITS   = 64
EPOCHS       = 25
BATCH_SIZE   = 64
SEED         = 42

OUT_DIR = Path("artifacts/B4_proper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_window_scorer():
    inp = keras.Input(shape=(N_CONTEXT, N_FEATURES), name="window_input")
    x = layers.LSTM(LSTM_UNITS, return_sequences=False, name="lstm")(inp)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="tanh", name="score")(x)
    return keras.Model(inp, out, name="window_scorer")


# ────────────────────────────────────────────────────────────────────────────
# 1. Split PAR PARTICIPANT (le cœur de B.4)
# ────────────────────────────────────────────────────────────────────────────
def split_by_participant(trials_dict, val_ratio=0.2, seed=SEED):
    """Sépare des trials en train/val SANS jamais mettre 2 trials d'un même
    participant des 2 côtés."""
    by_participant = defaultdict(list)
    for (pid, tid), rec in trials_dict.items():
        by_participant[pid].append(((pid, tid), rec))

    participants = sorted(by_participant.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(participants)

    n_val = max(1, int(len(participants) * val_ratio))
    val_pids = set(participants[:n_val])
    tr_pids  = set(participants[n_val:])

    train = {k: v for pid in tr_pids  for k, v in by_participant[pid]}
    val   = {k: v for pid in val_pids for k, v in by_participant[pid]}
    return train, val, tr_pids, val_pids


def per_trial_predictions(model, trials_dict, mean, std, batch_size=BATCH_SIZE):
    """Prédit un score AGRÉGÉ (médiane) par trial."""
    preds = {}
    for key, rec in trials_dict.items():
        X = apply_norm(rec["X"], mean, std)
        windows, _ = extract_sliding_windows(X)
        if windows is None:
            continue
        scores = model.predict(windows, batch_size=batch_size, verbose=0).flatten()
        preds[key] = {
            "score_median": float(np.median(scores)),
            "score_mean":   float(scores.mean()),
            "n_windows":    len(scores),
            "y_reg":        rec["y_reg"],
            "y9":           rec["y9"],
            "level":        rec.get("level", "?"),
        }
    return preds


def compute_metrics(preds, mode="median"):
    key = f"score_{mode}"
    y_true = np.array([p["y_reg"] for p in preds.values()])
    y_pred = np.array([p[key]      for p in preds.values()])
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {"pearson": pearson, "r2": r2, "n_trials": len(y_true)}


# ────────────────────────────────────────────────────────────────────────────
# 2. Pipeline complet
# ────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(" B.4 — Split par participant + entraînement long + blind val")
    print("=" * 70)

    with open("data/continuous_per_trial.pkl", "rb") as f:
        dataset = pickle.load(f)

    extremes = {k: v for k, v in dataset.items() if v["y9"] in (0, 8)}
    blind    = {k: v for k, v in dataset.items() if v["y9"] in (1, 2, 3, 4, 5, 6, 7)}

    print(f"\n[1] Trials extrêmes (classes 0+8) : {len(extremes)}")
    print(f"    Trials blind     (classes 1-7) : {len(blind)}")

    train_trials, val_trials, tr_pids, val_pids = split_by_participant(extremes, val_ratio=0.2)
    print(f"\n[2] Split PAR PARTICIPANT :")
    print(f"    Train  : {len(train_trials)} trials  ({len(tr_pids)} participants)")
    print(f"    Val    : {len(val_trials)} trials  ({len(val_pids)} participants)")
    assert tr_pids.isdisjoint(val_pids), "Fuite : un participant des 2 côtés !"
    print(f"    Vérif fuite : OK (participants disjoints)")

    mean, std = compute_train_norm_stats(train_trials, n_features=N_FEATURES)
    train_norm = {k: {**v, "X": apply_norm(v["X"], mean, std)} for k, v in train_trials.items()}
    val_norm   = {k: {**v, "X": apply_norm(v["X"], mean, std)} for k, v in val_trials.items()}
    print(f"\n[3] Normalisation calculée sur train uniquement")

    X_tr, y_tr, y9_tr, _ = build_window_dataset(train_norm)
    X_va, y_va, y9_va, _ = build_window_dataset(val_norm)
    print(f"\n[4] Fenêtres :")
    print(f"    Train : {X_tr.shape}  (Class 0: {(y9_tr == 0).sum()}, Class 8: {(y9_tr == 8).sum()})")
    print(f"    Val   : {X_va.shape}  (Class 0: {(y9_va == 0).sum()}, Class 8: {(y9_va == 8).sum()})")

    sw_tr, wpc = compute_balanced_sample_weights(y9_tr)
    print(f"\n[5] sample_weight : Class 0 → {wpc[0]:.3f}, Class 8 → {wpc[8]:.3f}")

    model = build_window_scorer()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.Huber(delta=1.0),
        metrics=["mae"],
    )
    print(f"\n[6] Modèle : {model.count_params():,} paramètres")

    print(f"\n[7] Entraînement {EPOCHS} epochs (batch={BATCH_SIZE})...\n")
    history = model.fit(
        X_tr, y_tr,
        sample_weight=sw_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    print("\n" + "=" * 70)
    print(" Évaluation par trial (médiane des scores fenêtres)")
    print("=" * 70)

    print("\n[Val]  classes 0 + 8 (participants jamais vus pendant l'entraînement)")
    preds_val = per_trial_predictions(model, val_trials, mean, std)
    metrics_val = compute_metrics(preds_val, mode="median")
    print(f"  Pearson = {metrics_val['pearson']:+.3f}  |  R² = {metrics_val['r2']:+.3f}  "
          f"|  n_trials = {metrics_val['n_trials']}")
    for c in (0, 8):
        scs = [p["score_median"] for p in preds_val.values() if p["y9"] == c]
        if scs:
            print(f"  Class {c} (cible {-1 if c==0 else +1:+d}) : "
                  f"médiane des scores = {np.mean(scs):+.3f} ± {np.std(scs):.3f}  (n={len(scs)})")

    print("\n[Blind]  classes 1-7 (intermédiaires, JAMAIS vues)")
    preds_blind = per_trial_predictions(model, blind, mean, std)
    metrics_blind = compute_metrics(preds_blind, mode="median")
    print(f"  Pearson = {metrics_blind['pearson']:+.3f}  |  R² = {metrics_blind['r2']:+.3f}  "
          f"|  n_trials = {metrics_blind['n_trials']}")
    for c in range(1, 8):
        scs = [p["score_median"] for p in preds_blind.values() if p["y9"] == c]
        if scs:
            target_reg = -1 + 2 * c / 8
            print(f"  Class {c} (cible y_reg={target_reg:+.2f}) : "
                  f"médiane = {np.mean(scs):+.3f} ± {np.std(scs):.3f}  (n={len(scs)})")

    np.save(OUT_DIR / "norm_mean.npy", mean)
    np.save(OUT_DIR / "norm_std.npy",  std)
    with open(OUT_DIR / "history.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({"val": metrics_val, "blind": metrics_blind}, f, indent=2)
    model.save(OUT_DIR / "window_scorer.keras")
    print(f"\n[Sauvegarde] artifacts dans {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
