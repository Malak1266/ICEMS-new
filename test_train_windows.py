"""B.3 — Mini-entraînement LSTM sur le dataset windows-level.

Objectif :
- Construire un petit LSTM causal qui prend une fenêtre (300, 10) et sort un score dans [-1, +1].
- Entraîner sur le train_set (classes 0 + 8) avec sample_weight='balanced'.
- Vérifier que la loss Huber descend → preuve que le modèle apprend.
"""
import sys
import pickle
import numpy as np
from collections import Counter

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Réutilise les utilitaires déjà écrits
sys.path.insert(0, "src")
from continuous_scorer import compute_train_norm_stats, apply_norm

# Et les fonctions de B.2
from test_build_windows import (
    extract_sliding_windows,
    build_window_dataset,
    compute_balanced_sample_weights,
    N_CONTEXT,
)

# ────────────────────────────────────────────────────────────────────────────
# Hyperparamètres
# ────────────────────────────────────────────────────────────────────────────
N_FEATURES = 10
LSTM_UNITS = 64        # petit pour ce test
EPOCHS     = 5
BATCH_SIZE = 64
HUBER_DELTA = 1.0
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ────────────────────────────────────────────────────────────────────────────
# Modèle : LSTM causal → score unique par fenêtre
# ────────────────────────────────────────────────────────────────────────────
def build_window_scorer(n_features=N_FEATURES, lstm_units=LSTM_UNITS):
    """
    Input  : (batch, 300, n_features)
    Output : (batch, 1)  ∈ [-1, +1]
    """
    inp = keras.Input(shape=(N_CONTEXT, n_features), name="window_input")
    x = layers.LSTM(lstm_units, return_sequences=False, name="lstm")(inp)
    x = layers.Dense(32, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    out = layers.Dense(1, activation="tanh", name="score")(x)
    model = keras.Model(inp, out, name="window_scorer")
    return model


# ────────────────────────────────────────────────────────────────────────────
# Pipeline complète
# ────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(" B.3 — Mini-entraînement LSTM sur windows-level")
    print("=" * 70)

    # 1. Charger le dataset
    with open("data/continuous_per_trial.pkl", "rb") as f:
        dataset = pickle.load(f)
    train_set = {k: v for k, v in dataset.items() if v["y9"] in (0, 8)}
    print(f"\n[1] Train set : {len(train_set)} trials (classes 0 + 8)")

    # 2. Normalisation Z-score (stats calculées UNIQUEMENT sur train)
    mean, std = compute_train_norm_stats(train_set, n_features=N_FEATURES)
    train_set_norm = {
        k: {**v, "X": apply_norm(v["X"], mean, std)}
        for k, v in train_set.items()
    }
    print(f"[2] Normalisation Z-score appliquée — "
          f"mean[:3]={np.round(mean[:3], 2)}, std[:3]={np.round(std[:3], 2)}")

    # 3. Construire le dataset windows-level
    X_w, y_w, y9_w, keys = build_window_dataset(train_set_norm)
    print(f"[3] Windows : X={X_w.shape}, y={y_w.shape}")

    # 4. Sample weights balanced
    sw, wpc = compute_balanced_sample_weights(y9_w)
    print(f"[4] sample_weight : Class 0 → {wpc[0]:.3f}, Class 8 → {wpc[8]:.3f}")

    # 5. Mélange + split simple 90/10 pour suivre la val_loss
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X_w))
    n_val = len(X_w) // 10
    idx_val, idx_tr = idx[:n_val], idx[n_val:]

    X_tr, y_tr, sw_tr = X_w[idx_tr], y_w[idx_tr], sw[idx_tr]
    X_va, y_va, sw_va = X_w[idx_val], y_w[idx_val], sw[idx_val]
    print(f"[5] Split : train={len(X_tr)}, val={len(X_va)}")

    # 6. Construire le modèle
    model = build_window_scorer()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.Huber(delta=HUBER_DELTA),
        metrics=["mae"],
    )
    print(f"\n[6] Modèle : {model.count_params():,} paramètres")
    model.summary()

    # 7. Entraînement
    print(f"\n[7] Entraînement sur {EPOCHS} epochs (batch={BATCH_SIZE})...\n")
    history = model.fit(
        X_tr, y_tr,
        sample_weight=sw_tr,
        validation_data=(X_va, y_va, sw_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    # 8. Vérification : la loss a-t-elle diminué ?
    print("\n" + "=" * 70)
    print(" Résumé")
    print("=" * 70)
    loss_first = history.history["loss"][0]
    loss_last  = history.history["loss"][-1]
    val_first  = history.history["val_loss"][0]
    val_last   = history.history["val_loss"][-1]
    print(f"  Train loss : {loss_first:.4f} → {loss_last:.4f}  "
          f"(Δ = {loss_first - loss_last:+.4f})")
    print(f"  Val   loss : {val_first:.4f} → {val_last:.4f}  "
          f"(Δ = {val_first - val_last:+.4f})")

    # 9. Prédictions par classe (sanity check)
    preds = model.predict(X_va, batch_size=BATCH_SIZE, verbose=0).flatten()
    y9_va = y9_w[idx_val]
    print(f"\n  Prédictions moyennes par classe sur val :")
    for c in (0, 8):
        mask = y9_va == c
        if mask.sum() > 0:
            print(f"    Class {c} (cible {-1 if c == 0 else +1:+d}) : "
                  f"pred_mean = {preds[mask].mean():+.3f}  "
                  f"(n={mask.sum()})")


if __name__ == "__main__":
    main()
