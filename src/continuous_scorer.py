"""
continuous_scorer.py
====================
Scoring temporel continu de l'expertise chirurgicale.

Implémente l'architecture décrite dans PROJECT_PROGRESSION.md PARTIE 4 :
  - LSTM causal unidirectionnel
  - Sortie tanh ∈ [-1, +1] (cohérent avec la Tâche H : axe Y normalisé)
  - Scoring streaming : à chaque instant t, score(t) = f(X[0:t])
  - Mise à jour du score par pas de STRIDE_INFERENCE frames
  - Mémoire LSTM théoriquement infinie (variant sliding 1 min disponible en option)

Ce module est le **remplaçant** du paradigme « fenêtres fixes 32 frames + moyenne »
qui causait le biais central 1.2-2.7 (cf. PROJECT_PROGRESSION.md §2.4).

Cible :
  - Entraînement : régression sur classes extrêmes uniquement (Tâche D)
        Classe 0 (Medical student) → y = -1
        Classe 8 (Staff)            → y = +1
  - Validation aveugle : projection linéaire des classes intermédiaires (PGY1..6, Fellow).

Usage :
    python continuous_scorer.py                       # test synthétique
    python continuous_scorer.py --self-test           # idem (alias explicite)

Une fois le pipeline Narval rapatrié :
    from src.continuous_scorer import build_continuous_scorer, score_streaming
    model = build_continuous_scorer(n_features=4, lstm_units=128)
    scores_t = score_streaming(model, X_trial)        # X_trial shape (T, 4) → scores (T, 1)
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Force UTF-8 sur stdout/stderr pour les consoles Windows (PowerShell, cmd)
# qui utilisent par défaut cp1252 et plantent sur les caractères Unicode (→, ², ³, etc.).
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# Imports TF/Keras — encapsulés dans une fonction pour garder l'import lazy.
def _import_keras():
    """Import lazy de TF/Keras pour éviter le coût quand on n'instancie pas de modèle."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    return tf, keras, layers


# ── Configuration par défaut (cf. PROJECT_PROGRESSION.md §4.2.4) ───────────
@dataclass
class ScorerConfig:
    n_features: int        = 10             # 9 cinématiques (3 instr × 3 metrics) + valid_ratio
    lstm_units: int        = 128
    dense_hidden: int      = 64
    dropout: float         = 0.20
    learning_rate: float   = 1e-3
    huber_delta: float     = 0.5            # robuste aux outliers (gestes dangereux isolés)
    stride_inference: int  = 5              # pas de mise à jour du score (frames)
    memory_mode: str       = "full"         # "full" | "sliding_60s"
    sampling_rate_hz: float = 10.0          # fréquence Atracsys (sert à convertir 60s → frames)
    use_mae_encoder: bool  = False          # variante d'ablation Tâche I
    # Tâche D : split classes
    train_classes: Tuple[int, ...] = (0, 8)  # Medical student + Staff
    val_classes: Tuple[int, ...]   = (1, 2, 3, 4, 5, 6, 7)
    # Tâche F : filtrage occlusion
    min_valid_ratio_trial: float = 0.30     # rejette le trial si valid_ratio moyen < ce seuil
    # Sous-échantillonnage temporel optionnel (réduit le coût d'entraînement)
    decimation: int        = 1              # 1 = pas de décimation ; 5 = 1 frame sur 5


# ── Construction du modèle ─────────────────────────────────────────────────
def build_continuous_scorer(config: Optional[ScorerConfig] = None):
    """
    Construit le modèle Keras de scoring continu.

    Architecture :
        Input (None, n_features)
            → LSTM(lstm_units, return_sequences=True, stateful=False)
            → Dense(dense_hidden, relu) appliqué à chaque pas de temps
            → Dropout
            → Dense(1, tanh)
        Output (None, 1) ∈ [-1, +1]

    Le `return_sequences=True` est CRUCIAL : on veut un score PAR FRAME, pas un
    score unique pour toute la séquence. C'est ce qui permet la visualisation
    de l'évolution temporelle (Tâche H).

    Le LSTM est `stateful=False` à l'entraînement (batch indépendants) mais on
    fournit `score_streaming()` pour l'inférence incrémentale en production.
    """
    if config is None:
        config = ScorerConfig()
    _, keras, layers = _import_keras()

    inp = layers.Input(shape=(None, config.n_features), name="features_seq")

    x = layers.Masking(mask_value=0.0, name="mask_padding")(inp)
    x = layers.LSTM(
        config.lstm_units,
        return_sequences=True,
        name="causal_lstm",
    )(x)
    x = layers.TimeDistributed(
        layers.Dense(config.dense_hidden, activation="relu"),
        name="td_dense_hidden",
    )(x)
    x = layers.Dropout(config.dropout, name="dropout")(x)
    out = layers.TimeDistributed(
        layers.Dense(1, activation="tanh"),
        name="td_score",
    )(x)

    model = keras.Model(inp, out, name="continuous_expertise_scorer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=keras.losses.Huber(delta=config.huber_delta),
        metrics=["mae"],
    )
    return model


# ── Scoring streaming — inférence incrémentale ─────────────────────────────
def score_streaming(
    model,
    X_trial: np.ndarray,
    config: Optional[ScorerConfig] = None,
) -> np.ndarray:
    """
    Calcule le score d'expertise à chaque pas de temps pour un trial complet.

    Args:
        model    : Keras model produit par build_continuous_scorer().
        X_trial  : ndarray shape (T, n_features) — séquence complète d'un trial.
        config   : ScorerConfig (utilise stride_inference et memory_mode).

    Returns:
        scores : ndarray shape (T,) — score à chaque frame, dans [-1, +1].
                 Pour les frames intermédiaires entre deux mises à jour
                 (gap = stride_inference), on duplique la valeur précédente.

    Note importante :
        Cette implémentation passe la séquence complète d'un coup au LSTM
        (`return_sequences=True`), ce qui est mathématiquement équivalent à
        appliquer le modèle à chaque préfixe X[0:t] grâce à la causalité du LSTM
        unidirectionnel — et beaucoup plus rapide.

        Pour un VRAI streaming en production (frame-par-frame, sans connaître
        la séquence complète à l'avance), il faudra basculer sur un LSTM
        `stateful=True` et un buffer circulaire — voir TODO en bas de fichier.
    """
    if config is None:
        config = ScorerConfig()
    if X_trial.ndim != 2:
        raise ValueError(f"X_trial doit être (T, n_features), reçu {X_trial.shape}")
    T, F = X_trial.shape
    if F != config.n_features:
        raise ValueError(
            f"X_trial.shape[1] = {F} ≠ config.n_features = {config.n_features}. "
            f"Vérifier que le slicing 6→4 (Tâche B) a bien été appliqué."
        )

    if config.memory_mode == "sliding_60s":
        # Tronque l'historique à 60 s (mode d'ablation Tâche I).
        max_frames = int(config.sampling_rate_hz * 60.0)
        if T > max_frames:
            X_trial = X_trial[-max_frames:]
            T = X_trial.shape[0]

    # Inférence batch unique sur la séquence complète.
    X_batch = X_trial[np.newaxis, ...]                          # (1, T, F)
    scores_full = model.predict(X_batch, verbose=0)[0, :, 0]    # (T,)

    # Sous-échantillonnage selon stride_inference puis ré-injection.
    if config.stride_inference <= 1:
        return scores_full

    sampled_idx = np.arange(0, T, config.stride_inference)
    sampled_scores = scores_full[sampled_idx]

    # Hold-last : pour les frames intermédiaires, on garde la dernière valeur calculée.
    scores_held = np.empty(T, dtype=np.float32)
    j = 0
    for t in range(T):
        if j + 1 < len(sampled_idx) and t >= sampled_idx[j + 1]:
            j += 1
        scores_held[t] = sampled_scores[j]
    return scores_held


# ── Visualisation (Tâche H) ────────────────────────────────────────────────
def plot_score_evolution(
    scores: np.ndarray,
    title: str = "Évolution du score d'expertise",
    ax=None,
    color: str = "#2166ac",
    label: Optional[str] = None,
):
    """
    Trace l'évolution du score d'un trial sur un graphe normalisé :
        X ∈ [0, 1]  (début → fin du trial)
        Y ∈ [-1, +1] (score d'expertise)

    Conforme à PROJECT_PROGRESSION.md §4.4 (Tâche H).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    T = len(scores)
    x_norm = np.linspace(0.0, 1.0, T)

    ax.plot(x_norm, scores, color=color, lw=1.2, label=label)
    ax.axhline(0.0, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax.axhline(+1.0, color="green", lw=0.4, ls=":", alpha=0.4)
    ax.axhline(-1.0, color="red", lw=0.4, ls=":", alpha=0.4)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Temps normalisé (0 = début du trial, 1 = fin)")
    ax.set_ylabel("Score d'expertise")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend(loc="best")
    return ax


# ── Chargement et préparation du dataset (Tâches D, E, F) ──────────────────
def load_continuous_dataset(
    pickle_path: str | Path = "data/continuous_per_trial.pkl",
) -> Dict[Tuple[str, str], dict]:
    """Charge le dict {(participant, trial): {X, y9, y_reg, level, T, fs}}."""
    pickle_path = Path(pickle_path)
    if not pickle_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {pickle_path}\n"
            f"Lancer d'abord : python src/build_continuous_dataset.py"
        )
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def filter_occluded(
    dataset: Dict[Tuple[str, str], dict],
    min_valid_ratio: float = 0.30,
    valid_ratio_idx: int = -1,
    verbose: bool = True,
) -> Dict[Tuple[str, str], dict]:
    """
    Tâche F : élimine les trials dont la fraction de frames valides est < seuil.

    Le canal `valid_ratio` est par convention la dernière colonne (index -1) du
    tenseur X. La moyenne sur tout le trial donne la fraction de frames actives.

    Args:
        dataset         : dict trial-level produit par load_continuous_dataset().
        min_valid_ratio : seuil minimal (cohérent avec la directive prof §2.4.2).
        valid_ratio_idx : index du canal valid_ratio dans X (default -1 = dernier).

    Returns:
        Sous-dict des trials retenus.
    """
    kept, dropped = {}, []
    for key, rec in dataset.items():
        vr_mean = float(rec["X"][:, valid_ratio_idx].mean())
        if vr_mean >= min_valid_ratio:
            kept[key] = rec
        else:
            dropped.append((key, vr_mean))

    if verbose:
        print(f"\n[Tâche F] Filtrage occlusion (seuil={min_valid_ratio:.2f}) : "
              f"gardés={len(kept)}, rejetés={len(dropped)}")
        for (pid, tid), vr in dropped[:5]:
            print(f"    drop  participant={pid}  trial={tid}  valid_ratio={vr:.3f}")
    return kept


def split_extremes_vs_blind(
    dataset: Dict[Tuple[str, str], dict],
    train_classes: Tuple[int, ...] = (0, 8),
    val_classes: Tuple[int, ...]   = (1, 2, 3, 4, 5, 6, 7),
    verbose: bool = True,
) -> Tuple[Dict, Dict]:
    """
    Tâche D : découpe le dataset en TRAIN (classes extrêmes uniquement) et
    VAL aveugle (classes intermédiaires).

    Aucune supervision n'est donnée au modèle sur les classes intermédiaires —
    le R² aveugle se calcule a posteriori (cf. compute_blind_metrics).
    """
    train_set = {k: v for k, v in dataset.items() if v["y9"] in train_classes}
    val_set   = {k: v for k, v in dataset.items() if v["y9"] in val_classes}

    if verbose:
        print(f"\n[Tâche D] Split entraînement aux extrêmes :")
        print(f"  TRAIN  ({len(train_set):3d} trials) : classes {train_classes}")
        for c in train_classes:
            n = sum(1 for v in train_set.values() if v["y9"] == c)
            print(f"            class {c} : {n} trials")
        print(f"  VAL    ({len(val_set):3d} trials) : classes {val_classes} (aveugle)")
        for c in val_classes:
            n = sum(1 for v in val_set.values() if v["y9"] == c)
            print(f"            class {c} : {n} trials")
    return train_set, val_set


def compute_train_norm_stats(
    train_set: Dict[Tuple[str, str], dict],
    n_features: int,
    valid_ratio_idx: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule mean/std par feature sur l'ensemble d'entraînement uniquement
    (jamais sur la validation, sinon fuite de données).

    Le canal valid_ratio (index -1) reste dans [0, 1] et n'est PAS normalisé.
    """
    if valid_ratio_idx < 0:
        valid_ratio_idx = n_features + valid_ratio_idx

    all_frames = np.concatenate([v["X"] for v in train_set.values()], axis=0)  # (sum_T, F)
    mean = all_frames.mean(axis=0).astype(np.float32)
    std  = all_frames.std(axis=0).astype(np.float32)
    std  = np.where(std < 1e-8, 1.0, std)

    # Annule la normalisation pour la colonne valid_ratio.
    mean[valid_ratio_idx] = 0.0
    std[valid_ratio_idx]  = 1.0
    return mean, std


def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def decimate(X: np.ndarray, k: int) -> np.ndarray:
    """Sous-échantillonnage temporel uniforme : 1 frame sur k."""
    if k <= 1:
        return X
    return X[::k]


# ── Préparation Keras (padding pour batches de séquences variables) ────────
def make_padded_arrays(
    trials: Dict[Tuple[str, str], dict],
    mean: np.ndarray,
    std: np.ndarray,
    decimation: int = 1,
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[Tuple[str, str]]]:
    """
    Empile tous les trials dans un tenseur dense (n_trials, T_max, F)
    avec padding 0 + masque automatique via la couche Masking.

    Returns:
        X_pad    : (n_trials, T_max, F)
        y_seq    : (n_trials, T_max, 1) — cible répliquée à chaque frame
        Ts       : longueurs réelles (avant padding)
        keys     : ordre des (participant, trial)
    """
    keys = list(trials.keys())
    Xs = [decimate(apply_norm(trials[k]["X"], mean, std), decimation) for k in keys]
    Ts = [x.shape[0] for x in Xs]
    n_trials = len(keys)
    T_max = max(Ts)
    F = Xs[0].shape[1]

    X_pad = np.full((n_trials, T_max, F), pad_value, dtype=np.float32)
    y_seq = np.zeros((n_trials, T_max, 1), dtype=np.float32)
    for i, (k, x) in enumerate(zip(keys, Xs)):
        T = x.shape[0]
        X_pad[i, :T] = x
        y_seq[i, :T, 0] = trials[k]["y_reg"]
    return X_pad, y_seq, Ts, keys


# ── Agrégation et métriques (Tâches E + D) ─────────────────────────────────
def aggregate_trial_score(scores: np.ndarray, mode: str = "median") -> float:
    """
    Tâche E : score global d'un trial à partir de la time-series score(t).

    Le prof a explicitement demandé d'abandonner la moyenne (qui souffre du
    « biais central » à cause des temps morts statiques). La médiane est
    plus robuste aux gestes dangereux isolés et aux pauses.
    """
    if mode == "median":
        return float(np.median(scores))
    if mode == "mean":
        return float(np.mean(scores))
    if mode == "p75":
        return float(np.percentile(scores, 75))
    raise ValueError(f"Mode d'agrégation inconnu : {mode}")


def compute_blind_metrics(
    predictions: Dict[Tuple[str, str], float],
    val_set: Dict[Tuple[str, str], dict],
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Tâche D + H : Pearson + R² aveugle sur les classes intermédiaires (PGY1-Fellow).

    Le R² aveugle ne doit PAS être sur-optimisé (cf. PROJECT_PROGRESSION.md §2.5.2).
    Il indique simplement comment les données épousent la dynamique du modèle.
    """
    keys = sorted(set(predictions.keys()) & set(val_set.keys()))
    if not keys:
        return {"pearson": float("nan"), "r2": float("nan"), "n": 0}

    y_pred = np.array([predictions[k]    for k in keys], dtype=np.float64)
    y_true = np.array([val_set[k]["y_reg"] for k in keys], dtype=np.float64)

    if y_pred.std() < 1e-9 or y_true.std() < 1e-9:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(y_pred, y_true)[0, 1])

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else float("nan")

    metrics = {"pearson": pearson, "r2": r2, "n": len(keys)}

    if verbose:
        print(f"\n=== Métriques aveugles (classes intermédiaires) ===")
        print(f"  n_trials       : {metrics['n']}")
        print(f"  Pearson r      : {metrics['pearson']:+.4f}")
        print(f"  R²             : {metrics['r2']:+.4f}")
        print(f"  (Cibles connues du prof : Pearson > 0 attend la convergence du modèle.)")

    return metrics


def predict_all_trials(
    model,
    dataset: Dict[Tuple[str, str], dict],
    mean: np.ndarray,
    std: np.ndarray,
    config: ScorerConfig,
    aggregate_mode: str = "median",
) -> Dict[Tuple[str, str], dict]:
    """
    Inférence sur tous les trials du dataset.
    Retourne {key: {"scores": (T,), "agg_score": float}}.
    """
    out = {}
    for key, rec in dataset.items():
        X_norm = apply_norm(rec["X"], mean, std)
        X_norm = decimate(X_norm, config.decimation)
        scores = score_streaming(model, X_norm, config)
        out[key] = {
            "scores":    scores,
            "agg_score": aggregate_trial_score(scores, mode=aggregate_mode),
            "y_reg":     rec["y_reg"],
            "y9":        rec["y9"],
            "level":     rec["level"],
        }
    return out


# ── Boucle d'entraînement (Tâche D) ────────────────────────────────────────
def train_on_extremes(
    model,
    train_set: Dict[Tuple[str, str], dict],
    val_set:   Dict[Tuple[str, str], dict],
    config: ScorerConfig,
    epochs: int = 10,
    batch_size: int = 4,
    verbose: int = 1,
):
    """
    Entraîne le modèle uniquement sur les classes extrêmes (Tâche D).
    Calcule à chaque epoch les métriques aveugles sur le val_set.

    Returns :
        history (dict) : {"train_loss": [...], "val_pearson": [...], "val_r2": [...]}
    """
    _, keras, _ = _import_keras()
    np.random.seed(42)

    # 1. Statistiques de normalisation calculées TRAIN-only.
    mean, std = compute_train_norm_stats(train_set, n_features=config.n_features)
    print(f"\n[Norm] mean[:3]={mean[:3]}, std[:3]={std[:3]} "
          f"(valid_ratio mean={mean[-1]:.2f}, std={std[-1]:.2f} — non normalisé)")

    # 2. Padding + tenseurs pour Keras.
    X_tr, y_tr, T_tr, _ = make_padded_arrays(train_set, mean, std, config.decimation)
    X_va, y_va, T_va, _ = make_padded_arrays(val_set,   mean, std, config.decimation)
    print(f"\n[Tensors] Train shape : X={X_tr.shape}, y={y_tr.shape}")
    print(f"          Val   shape : X={X_va.shape}, y={y_va.shape}")
    print(f"          T train     : min={min(T_tr)}, médiane={int(np.median(T_tr))}, max={max(T_tr)}")
    print(f"          T val       : min={min(T_va)}, médiane={int(np.median(T_va))}, max={max(T_va)}")

    # 3. Callback de validation aveugle (Pearson + R²).
    class BlindValidationCB(keras.callbacks.Callback):
        def __init__(self, val_set, mean, std, config):
            super().__init__()
            self.val_set = val_set
            self.mean = mean
            self.std = std
            self.config = config
            self.history = {"val_pearson": [], "val_r2": []}

        def on_epoch_end(self, epoch, logs=None):
            preds = predict_all_trials(
                self.model, self.val_set, self.mean, self.std, self.config,
            )
            agg = {k: v["agg_score"] for k, v in preds.items()}
            metrics = compute_blind_metrics(agg, self.val_set, verbose=False)
            self.history["val_pearson"].append(metrics["pearson"])
            self.history["val_r2"].append(metrics["r2"])
            print(f"      [blind] Pearson={metrics['pearson']:+.4f}  R²={metrics['r2']:+.4f}")

    blind_cb = BlindValidationCB(val_set, mean, std, config)

    # 4. Entraînement.
    hist = model.fit(
        X_tr, y_tr,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=[blind_cb],
        shuffle=True,
    )

    return {
        "train_loss":  hist.history.get("loss", []),
        "val_pearson": blind_cb.history["val_pearson"],
        "val_r2":      blind_cb.history["val_r2"],
        "norm_mean":   mean,
        "norm_std":    std,
    }


# ── Auto-test sur données synthétiques ─────────────────────────────────────
def _generate_synthetic_trial(
    n_frames: int,
    expertise_target: float,
    seed: int = 0,
) -> np.ndarray:
    """
    Génère un trial synthétique (T, 4) avec une signature cinématique cohérente :
      - expertise_target = +1 (Staff) → mouvements lents, peu de jerk, valid_ratio haut
      - expertise_target = -1 (Student) → mouvements saccadés, jerk élevé, plus d'occlusions

    Sert UNIQUEMENT à valider la topologie du modèle. Aucune prétention de réalisme.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n_frames)

    smoothness = 0.5 * (1.0 + expertise_target)               # 0..1 ; 1 = expert lisse
    velocity = 50.0 * (1.0 - 0.7 * smoothness) * np.sin(t) + rng.normal(0, 5, n_frames)
    accel    = np.gradient(velocity)
    jerk     = np.gradient(accel)
    jerk    += rng.normal(0, 8 * (1.0 - smoothness), n_frames)

    valid_ratio = 0.85 + 0.10 * smoothness + rng.normal(0, 0.03, n_frames)
    valid_ratio = np.clip(valid_ratio, 0.0, 1.0)

    return np.stack([velocity, accel, jerk, valid_ratio], axis=1).astype(np.float32)


def self_test(verbose: bool = True):
    """
    Validation de bout-en-bout sans aucune dépendance aux données Narval :
      1. Construit un modèle.
      2. Génère 4 trials synthétiques (2 Students, 2 Staffs) de longueurs variables.
      3. Vérifie que les shapes/plages de sortie sont conformes.
      4. (Optionnel) Affiche un graphe de chaque trial avec label vs score.

    Ce test ne fait PAS d'apprentissage : avec un modèle non entraîné, les scores
    sont aléatoires. Il valide uniquement l'INTÉGRITÉ DE LA PIPELINE.
    """
    cfg = ScorerConfig(n_features=4, lstm_units=64, stride_inference=5)
    model = build_continuous_scorer(cfg)
    if verbose:
        model.summary()
        print(f"\n{'='*60}\nAuto-test de la pipeline de scoring continu\n{'='*60}")

    trials = [
        ("Student-1", -1.0, 320),
        ("Staff-1",   +1.0, 480),
        ("Student-2", -1.0, 250),
        ("Staff-2",   +1.0, 600),
    ]

    results = []
    for name, target, T in trials:
        X = _generate_synthetic_trial(T, target, seed=hash(name) & 0xFFFF)
        scores = score_streaming(model, X, cfg)

        assert scores.shape == (T,), f"Shape invalide : {scores.shape} ≠ ({T},)"
        assert np.all(scores >= -1.0) and np.all(scores <= 1.0), \
            f"Score hors [-1, +1] : min={scores.min()}, max={scores.max()}"

        if verbose:
            print(
                f"  {name:12s} (target={target:+.1f}, T={T:4d} frames) "
                f"→ score moyen={scores.mean():+.3f}, "
                f"min={scores.min():+.3f}, max={scores.max():+.3f}"
            )
        results.append((name, target, X, scores))

    if verbose:
        print(f"\n✅ Pipeline OK : 4 trials synthétiques traités, scores ∈ [-1, +1].")
        print("   (Le modèle n'est pas entraîné, donc les scores n'ont AUCUN sens "
              "sémantique. Ce test valide seulement la topologie.)")

    return results


def _maybe_plot(results, save_path: Optional[str] = None):
    """Plot optionnel des 4 trials synthétiques."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (name, target, _X, scores) in zip(axes.flatten(), results):
        plot_score_evolution(
            scores, title=f"{name} (cible={target:+.1f})", ax=ax,
            color="#1a9850" if target > 0 else "#d73027",
        )
    fig.suptitle(
        "Auto-test scoring continu — modèle NON entraîné (scores aléatoires)",
        fontsize=11,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"  → Figure sauvegardée : {save_path}")
    else:
        plt.show()


# ── Pipeline d'entraînement réel sur données labellisées (Tâches D + E + F) ─
def main_train(
    dataset_path: str = "data/continuous_per_trial.pkl",
    epochs: int = 5,
    batch_size: int = 2,
    decimation: int = 5,
    lstm_units: int = 64,
    min_valid_ratio: float = 0.30,
    save_dir: str = "results_continuous",
):
    """
    Pipeline complet : charge → filtre → split → entraîne → valide → sauvegarde.

    Hyperparamètres par défaut volontairement modestes (epochs=5, batch_size=2,
    lstm_units=64, decimation=5) pour permettre un tour complet en quelques
    minutes sur CPU. Pour un entraînement de production, augmenter ces valeurs.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Pipeline scoring continu — entraînement Tâche D + validation aveugle")
    print("=" * 70)

    # 1. Charger le dataset.
    dataset = load_continuous_dataset(dataset_path)
    print(f"\n[Load] {len(dataset)} trials chargés depuis {dataset_path}")

    # Valider la dimension de features.
    sample_X = next(iter(dataset.values()))["X"]
    n_features = sample_X.shape[1]
    print(f"       n_features détecté : {n_features}")

    # 2. Tâche F — filtrage occlusion.
    dataset_f = filter_occluded(dataset, min_valid_ratio=min_valid_ratio)

    # 3. Tâche D — split.
    train_set, val_set = split_extremes_vs_blind(dataset_f)
    if not train_set or not val_set:
        raise RuntimeError(
            "Train ou Val vide après filtrage. Réduire min_valid_ratio."
        )

    # 4. Construire le modèle.
    config = ScorerConfig(
        n_features=n_features,
        lstm_units=lstm_units,
        decimation=decimation,
        min_valid_ratio_trial=min_valid_ratio,
    )
    model = build_continuous_scorer(config)
    print(f"\n[Model] Total params : {model.count_params():,}")

    # 5. Entraîner.
    history = train_on_extremes(
        model, train_set, val_set, config,
        epochs=epochs, batch_size=batch_size,
    )

    # 6. Évaluation finale + sauvegarde.
    print("\n" + "=" * 70)
    print("  Évaluation finale")
    print("=" * 70)
    preds = predict_all_trials(
        model, val_set, history["norm_mean"], history["norm_std"], config,
    )
    agg = {k: v["agg_score"] for k, v in preds.items()}
    metrics = compute_blind_metrics(agg, val_set, verbose=True)

    # Sauvegarder modèle + paramètres + métriques.
    model.save(save_path / "scorer.keras")
    np.save(save_path / "norm_mean.npy", history["norm_mean"])
    np.save(save_path / "norm_std.npy",  history["norm_std"])
    with open(save_path / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Pearson r : {metrics['pearson']:+.4f}\n")
        f.write(f"R²        : {metrics['r2']:+.4f}\n")
        f.write(f"n_val     : {metrics['n']}\n")
        f.write(f"\nTrain history :\n")
        for i, (loss, p, r) in enumerate(zip(
            history["train_loss"], history["val_pearson"], history["val_r2"]
        )):
            f.write(f"  epoch {i+1:2d}  loss={loss:.4f}  pearson={p:+.4f}  r²={r:+.4f}\n")
    print(f"\n✅ Sauvegardé dans {save_path}/")

    return model, history, metrics, preds


def plot_blind_predictions(
    preds: Dict[Tuple[str, str], dict],
    save_path: Optional[str] = None,
):
    """
    Trace un nuage de points (y_reg cible, agg_score prédit) pour la validation aveugle,
    coloré par classe. Permet de visualiser la dynamique du modèle (cf. §2.5.3).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    classes = sorted({v["y9"] for v in preds.values()})
    cmap = plt.get_cmap("RdYlGn")

    for c in classes:
        keys = [k for k, v in preds.items() if v["y9"] == c]
        xs = [preds[k]["y_reg"]      for k in keys]
        ys = [preds[k]["agg_score"]  for k in keys]
        ax.scatter(xs, ys, label=f"Class {c} ({preds[keys[0]]['level']})",
                   color=cmap(c / 8.0), s=60, alpha=0.75, edgecolors="black", lw=0.5)

    ax.plot([-1, 1], [-1, 1], "k--", lw=0.8, alpha=0.4, label="y = x (idéal)")
    ax.axhline(0, color="gray", lw=0.4, ls=":")
    ax.axvline(0, color="gray", lw=0.4, ls=":")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Cible y_reg (basée sur la classe)")
    ax.set_ylabel("Score agrégé (médiane)")
    ax.set_title("Validation aveugle — agg_score vs y_reg (classes intermédiaires)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        print(f"  → Figure sauvegardée : {save_path}")
    return ax


# ── Point d'entrée CLI ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    # Mode self-test (par défaut, conservé)
    p_test = sub.add_parser("self-test", help="Auto-test sur données synthétiques.")
    p_test.add_argument("--plot", action="store_true")
    p_test.add_argument("--save", type=str, default=None)

    # Mode training réel
    p_train = sub.add_parser("train", help="Entraînement sur données labellisées (Tâches D+E+F).")
    p_train.add_argument("--dataset", type=str, default="data/continuous_per_trial.pkl")
    p_train.add_argument("--epochs",     type=int, default=5)
    p_train.add_argument("--batch-size", type=int, default=2)
    p_train.add_argument("--decimation", type=int, default=5,
                         help="Sous-échantillonnage temporel (1=aucun, 5=1 frame sur 5).")
    p_train.add_argument("--lstm-units", type=int, default=64)
    p_train.add_argument("--min-valid",  type=float, default=0.30)
    p_train.add_argument("--out-dir",    type=str, default="results_continuous")
    p_train.add_argument("--plot",       action="store_true")

    args = parser.parse_args()

    # Mode par défaut : self-test (rétrocompatibilité).
    if args.cmd is None or args.cmd == "self-test":
        results = self_test(verbose=True)
        if getattr(args, "plot", False) or getattr(args, "save", None):
            _maybe_plot(results, save_path=getattr(args, "save", None))
        return

    if args.cmd == "train":
        model, history, metrics, preds = main_train(
            dataset_path=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            decimation=args.decimation,
            lstm_units=args.lstm_units,
            min_valid_ratio=args.min_valid,
            save_dir=args.out_dir,
        )
        if args.plot:
            plot_blind_predictions(preds, save_path=str(Path(args.out_dir) / "blind_scatter.png"))
        return


if __name__ == "__main__":
    main()


# ─── Notes pour la suite ───────────────────────────────────────────────────
#
# TODO (post-Narval, après rapatriement des données labellisées) :
#   1. Charger continuous_per_trial.pkl (cf. PROJECT_PROGRESSION.md §4.5).
#   2. Implémenter le LOOCV par trial sur les classes extrêmes (Tâche D).
#   3. Brancher l'encoder MAE gelé en option (Tâche I, flag use_mae_encoder).
#   4. Streaming temps-réel : convertir vers stateful=True + buffer circulaire.
#   5. Ajouter le calcul du R² aveugle sur les classes intermédiaires (§4.3).
