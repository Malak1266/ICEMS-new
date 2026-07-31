"""
train_long.py
=============
Entraînement LONG de production du scorer d'expertise (baseline, sans MAE).

Différences avec la baseline 25-epochs :
    - epochs configurables (défaut 50) + EarlyStopping sur val_loss
    - pondération de classe configurable (corrige le déséquilibre Student/Staff)
    - suivi du Pearson BLIND à chaque epoch (callback dédié)
    - sauvegarde du modèle final + courbes (loss & Pearson blind)

Produit le modèle utilisé ensuite par plot_score_vs_time.py (P3).

Usage (depuis ICEMS-main) :
    python train_long.py
    python train_long.py --epochs 80 --lstm-units 128 --staff-weight 1.75
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# Réutilise tout le pipeline de kfold_cv (pas de duplication).
from kfold_cv import (  # noqa: E402
    extract_sliding_windows, build_window_dataset, build_window_scorer,
    per_trial_median_predictions, pearson_and_r2,
    N_CONTEXT, HOP, N_FEATURES, SEED,
)
from continuous_scorer import compute_train_norm_stats, apply_norm  # noqa: E402

import pickle  # noqa: E402


def split_by_participant_holdout(trials, val_ratio=0.2, seed=SEED):
    """Sépare train/val SANS qu'un participant soit des deux côtés."""
    by_pid = {}
    for (pid, tid), rec in trials.items():
        by_pid.setdefault(pid, []).append(((pid, tid), rec))
    pids = sorted(by_pid)
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n_val = max(1, int(len(pids) * val_ratio))
    val_pids, tr_pids = set(pids[:n_val]), set(pids[n_val:])
    train = {k: v for p in tr_pids for k, v in by_pid[p]}
    val = {k: v for p in val_pids for k, v in by_pid[p]}
    return train, val


def class_weighted_sample_weights(y9, weight_map):
    """sample_weight = poids défini par classe (régression → on passe par sample_weight)."""
    return np.array([weight_map.get(int(c), 1.0) for c in y9], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="Entraînement long du scorer (baseline).")
    ap.add_argument("--dataset", type=Path, default=Path("data/continuous_per_trial.pkl"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/train_long"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lstm-units", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=12, help="EarlyStopping patience.")
    ap.add_argument("--staff-weight", type=float, default=1.75,
                    help="Poids de la classe 8 (Staff) ; classe 0 = 1.0.")
    args = ap.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(
            f"{args.dataset} introuvable. Lance d'abord : "
            f"python src/build_continuous_dataset.py")

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    tf.keras.utils.set_random_seed(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # On force la taille du LSTM utilisée par build_window_scorer via le module.
    import kfold_cv
    kfold_cv.LSTM_UNITS = args.lstm_units

    with open(args.dataset, "rb") as f:
        dataset = pickle.load(f)
    extremes = {k: v for k, v in dataset.items() if v["y9"] in (0, 8)}
    blind = {k: v for k, v in dataset.items() if v["y9"] in (1, 2, 3, 4, 5, 6, 7)}

    train_trials, val_trials = split_by_participant_holdout(extremes)
    print(f"[Split] train={len(train_trials)} trials, val={len(val_trials)} trials, "
          f"blind={len(blind)} trials")

    mean, std = compute_train_norm_stats(train_trials, n_features=N_FEATURES)
    train_norm = {k: {**v, "X": apply_norm(v["X"], mean, std)} for k, v in train_trials.items()}
    val_norm = {k: {**v, "X": apply_norm(v["X"], mean, std)} for k, v in val_trials.items()}

    X_tr, y_tr, y9_tr = build_window_dataset(train_norm)
    X_va, y_va, _ = build_window_dataset(val_norm)
    sw = class_weighted_sample_weights(y9_tr, {0: 1.0, 8: args.staff_weight})
    print(f"[Données] fenêtres train={X_tr.shape}, val={X_va.shape} | "
          f"poids classe {{0:1.0, 8:{args.staff_weight}}}")

    model = build_window_scorer(keras, layers)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=keras.losses.Huber(delta=1.0), metrics=["mae"])
    print(f"[Modèle] {model.count_params():,} paramètres\n")

    # Callback : suit le Pearson BLIND (classes 1-7) à chaque epoch.
    blind_history = []

    class BlindPearsonCB(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            preds = per_trial_median_predictions(self.model, blind, mean, std)
            p, r2, _ = pearson_and_r2(preds)
            blind_history.append(p)
            print(f"      [blind] Pearson={p:+.3f}  R²={r2:+.3f}")

    early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                          restore_best_weights=True)
    hist = model.fit(X_tr, y_tr, sample_weight=sw, validation_data=(X_va, y_va),
                     epochs=args.epochs, batch_size=args.batch_size, verbose=2,
                     callbacks=[BlindPearsonCB(), early])

    # Évaluation finale.
    p_val = pearson_and_r2(per_trial_median_predictions(model, val_trials, mean, std))
    p_blind = pearson_and_r2(per_trial_median_predictions(model, blind, mean, std))
    print("\n" + "=" * 60)
    print(f"  Val (extrêmes) : Pearson={p_val[0]:+.3f}  R²={p_val[1]:+.3f}")
    print(f"  Blind (1-7)    : Pearson={p_blind[0]:+.3f}  R²={p_blind[1]:+.3f}")
    print("=" * 60)

    # Sauvegardes.
    model.save(args.out_dir / "scorer.keras")
    np.save(args.out_dir / "norm_mean.npy", mean)
    np.save(args.out_dir / "norm_std.npy", std)
    with open(args.out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump({"loss": [float(x) for x in hist.history["loss"]],
                   "val_loss": [float(x) for x in hist.history["val_loss"]],
                   "blind_pearson": [float(x) for x in blind_history],
                   "final_val": p_val[0], "final_blind": p_blind[0]}, f, indent=2)

    # Courbes.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
        a1.plot(hist.history["loss"], label="train")
        a1.plot(hist.history["val_loss"], label="val")
        a1.set_title("Loss (Huber)"); a1.set_xlabel("epoch"); a1.legend(); a1.grid(alpha=0.3)
        a2.plot(blind_history, color="#d95f02")
        a2.axhline(0, color="gray", ls="--", lw=0.6)
        a2.set_title("Pearson BLIND par epoch"); a2.set_xlabel("epoch"); a2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / "curves.png", dpi=120)
        print(f"  → Courbes : {args.out_dir / 'curves.png'}")
    except Exception as e:  # pragma: no cover
        print(f"  (plot ignoré : {e})")

    print(f"\n✅ Modèle + artefacts sauvegardés dans {args.out_dir}/")


if __name__ == "__main__":
    main()
