"""
build_clf_datasets.py
=====================
Construit et sauvegarde les deux datasets de classification à partir
de continuous_per_trial.pkl et filtered_data.json.

══════════════════════════════════════════════════════════════════════
  MODÈLE 1  —  Classification Expertise (4 classes)
══════════════════════════════════════════════════════════════════════
  Input  : features statistiques extraites des séries temporelles
           91 features par trial :
             • 9 canaux cinématiques × 10 stats = 90 features
             • 1 feature valid_ratio (mean)
  Target : y4 ∈ {0=Student, 1=Junior, 2=Senior, 3=Expert}
  Output : data/clf_model1.pkl
  Format : dict { (pid, tid) → {X_feat, y4, y_reg, y9, level} }

══════════════════════════════════════════════════════════════════════
  MODÈLE 2  —  Classification Niveau de formation (9 classes)
══════════════════════════════════════════════════════════════════════
  Input  : séries temporelles brutes (T, 10) — même format que
           continuous_per_trial.pkl — avec augmentation des classes
           minoritaires appliquée globalement (exploration uniquement).
           ⚠ Pour la validation LOPO, utiliser augment_model2_fold()
             de src/clf_model2_augment.py dans le fold train uniquement.
  Target : y9 ∈ {0=ms, 1=pgy1, 2=pgy2, 3=pgy3, 4=pgy4,
                  5=pgy5, 6=pgy6, 7=fellow, 8=staff}
  Output : data/clf_model2_aug.pkl
  Format : dict { key → {X, y9, y_reg, y4, level, is_augmented, ...} }

Usage :
    python src/build_clf_datasets.py
    python -m src.build_clf_datasets
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

# Assure que src/ est dans le path (compatible run direct et run depuis la racine)
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ─── Chemins ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

PATH_PKL     = DATA_DIR / "continuous_per_trial.pkl"
PATH_JSON    = DATA_DIR / "filtered_data.json"
PATH_MODEL1  = DATA_DIR / "clf_model1.pkl"
PATH_MODEL2  = DATA_DIR / "clf_model2_aug.pkl"

# ─── Mapping expertise → y4 ───────────────────────────────────────────────────
EXPERTISE_TO_Y4: dict[str, int] = {
    "Student": 0,
    "Junior":  1,
    "Senior":  2,
    "Expert":  3,
}
Y4_NAMES: list[str] = ["Student", "Junior", "Senior", "Expert"]

# ─── Canaux cinématiques (indices 0-8 de X) ────────────────────────────────────
KIN_CHANNEL_NAMES: list[str] = [
    "bipolar.vel",   "bipolar.acc",   "bipolar.jerk",
    "scissors.vel",  "scissors.acc",  "scissors.jerk",
    "cavitron.vel",  "cavitron.acc",  "cavitron.jerk",
]  # canal 9 = valid_ratio

# ─── Noms des features statistiques (91 features) ─────────────────────────────
_STAT_NAMES = ["mean", "std", "min", "max", "median",
               "p25", "p75", "iqr", "skew", "kurtosis"]

FEATURE_NAMES: list[str] = (
    [f"{ch}.{s}" for ch in KIN_CHANNEL_NAMES for s in _STAT_NAMES]
    + ["valid_ratio.mean"]
)  # 9×10 + 1 = 91 features


# ══════════════════════════════════════════════════════════════════════════════
#  FONCTIONS UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def _extract_stat_features(X: np.ndarray) -> np.ndarray:
    """Extrait 91 features statistiques depuis une série temporelle (T, 10).

    Parameters
    ----------
    X : np.ndarray, shape (T, 10)
        Série temporelle d'un trial (9 canaux cinématiques + valid_ratio).

    Returns
    -------
    np.ndarray, shape (91,)
        Vecteur de features : [mean, std, min, max, median, p25, p75, iqr,
        skewness, kurtosis] pour chaque canal cinématique + mean(valid_ratio).
    """
    feats: list[float] = []

    # 9 canaux cinématiques (indices 0-8)
    for ch in range(9):
        col = X[:, ch].astype(np.float64)
        feats.extend([
            float(np.mean(col)),
            float(np.std(col)),
            float(np.min(col)),
            float(np.max(col)),
            float(np.median(col)),
            float(np.percentile(col, 25)),
            float(np.percentile(col, 75)),
            float(np.percentile(col, 75) - np.percentile(col, 25)),   # IQR
            float(sp_stats.skew(col)),
            float(sp_stats.kurtosis(col)),
        ])

    # Canal 9 : valid_ratio — uniquement la moyenne (proportion d'activité)
    feats.append(float(np.mean(X[:, 9])))

    return np.array(feats, dtype=np.float32)


def _build_participant_expertise_map(json_path: Path) -> dict[str, str]:
    """Construit le mapping participant_id → expertise depuis filtered_data.json.

    Returns
    -------
    dict[str, str]
        { "01020614": "Senior", "02090612": "Student", ... }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    n = len(raw["participant"])
    pid_to_exp: dict[str, str] = {}
    for i in range(n):
        pid = str(raw["participant"][str(i)])
        exp = str(raw["expertise"][str(i)]).strip()
        pid_to_exp[pid] = exp
    return pid_to_exp


# ══════════════════════════════════════════════════════════════════════════════
#  MODÈLE 1 — Features statistiques + y4
# ══════════════════════════════════════════════════════════════════════════════

def build_model1_dataset(
    pkl: dict,
    pid_to_expertise: dict[str, str],
) -> dict:
    """Construit le dataset du Modèle 1 (features statistiques, 4 classes).

    Parameters
    ----------
    pkl : dict
        Contenu de continuous_per_trial.pkl.
    pid_to_expertise : dict
        Mapping participant_id → expertise (str).

    Returns
    -------
    dict
        { (pid, tid) → {X_feat, y4, y_reg, y9, level} }
        X_feat : np.ndarray (91,)
        y4     : int ∈ [0, 3]
    """
    dataset: dict = {}
    skipped = 0

    for (pid, tid), trial in pkl.items():
        expertise = pid_to_expertise.get(str(pid))
        if expertise is None or expertise not in EXPERTISE_TO_Y4:
            skipped += 1
            continue

        y4 = EXPERTISE_TO_Y4[expertise]
        X_feat = _extract_stat_features(trial["X"])

        dataset[(pid, tid)] = {
            "X_feat":    X_feat,
            "y4":        y4,
            "expertise": expertise,
            "y_reg":     trial["y_reg"],
            "y9":        trial["y9"],
            "level":     trial["level"],
        }

    if skipped:
        print(f"  [!] {skipped} trials ignorés (expertise non reconnue).")

    return dataset


def _print_model1_summary(dataset: dict) -> None:
    counts = Counter(v["y4"] for v in dataset.values())
    print(f"\n{'Expertise':<12} {'Trials':>8}")
    print("-" * 22)
    for y4, name in enumerate(Y4_NAMES):
        print(f"  {name:<10} {counts.get(y4, 0):>8}")
    print(f"  {'TOTAL':<10} {sum(counts.values()):>8}")


# ══════════════════════════════════════════════════════════════════════════════
#  MODÈLE 2 — Séries temporelles brutes + augmentation globale
# ══════════════════════════════════════════════════════════════════════════════

def build_model2_dataset(
    pkl: dict,
    pid_to_expertise: dict[str, str],
    seed: int = 42,
) -> dict:
    """Construit le dataset du Modèle 2 avec augmentation globale des classes minoritaires.

    Ajoute y4 (expertise) à chaque enregistrement pour la compatibilité.
    Les échantillons synthétiques sont marqués ``is_augmented=True``.

    ⚠  Ce dataset est destiné à l'exploration et à l'entraînement global.
       Pour une validation LOPO rigoureuse, utiliser augment_model2_fold()
       de src/clf_model2_augment.py dans le fold train uniquement.

    Parameters
    ----------
    pkl : dict
        Contenu de continuous_per_trial.pkl.
    pid_to_expertise : dict
        Mapping participant_id → expertise (str).
    seed : int
        Graine pour la reproductibilité de l'augmentation.

    Returns
    -------
    dict
        { key → {X, y9, y_reg, y4, level, is_augmented, ...} }
    """
    from clf_model2_augment import augment_model2_fold, print_augmentation_summary

    # 1) Construire les enregistrements réels enrichis de y4
    real_trials: dict = {}
    for (pid, tid), trial in pkl.items():
        expertise = pid_to_expertise.get(str(pid), "Unknown")
        y4 = EXPERTISE_TO_Y4.get(expertise, -1)
        real_trials[(pid, tid)] = {
            "X":           trial["X"],
            "y9":          trial["y9"],
            "y_reg":       trial["y_reg"],
            "y4":          y4,
            "level":       trial["level"],
            "is_augmented": False,
            "aug_type":    None,
        }

    # 2) Augmentation — fold_i=0 pour le dataset global (toutes les données en train)
    synth_trials = augment_model2_fold(real_trials, fold_i=0, seed=seed)

    # Ajouter y4 aux synthétiques (propagé depuis la classe y9)
    y9_to_y4_map = _build_y9_to_y4_map(real_trials)
    for rec in synth_trials.values():
        rec["y4"] = y9_to_y4_map.get(rec["y9"], -1)

    # 3) Résumé
    print_augmentation_summary(real_trials, synth_trials)

    # 4) Fusionner
    dataset = {**real_trials, **synth_trials}
    return dataset


def _build_y9_to_y4_map(real_trials: dict) -> dict[int, int]:
    """Déduit le mapping y9 → y4 depuis les trials réels."""
    mapping: dict[int, int] = {}
    for rec in real_trials.values():
        y9 = rec["y9"]
        y4 = rec.get("y4", -1)
        if y9 not in mapping and y4 >= 0:
            mapping[y9] = y4
    return mapping


def _print_model2_summary(dataset: dict) -> None:
    from clf_model2_augment import Y9_NAMES
    counts_real = Counter(
        int(v["y9"]) for v in dataset.values() if not v.get("is_augmented", False)
    )
    counts_all = Counter(int(v["y9"]) for v in dataset.values())
    print(f"\n{'Classe':<12} {'Réels':>8} {'Total':>8}")
    print("-" * 32)
    for cls, name in enumerate(Y9_NAMES):
        print(f"  {name:<10} {counts_real.get(cls, 0):>8} {counts_all.get(cls, 0):>8}")
    print(f"  {'TOTAL':<10} {sum(counts_real.values()):>8} {sum(counts_all.values()):>8}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  build_clf_datasets.py")
    print("=" * 60)

    # ── Chargement des sources ─────────────────────────────────────────────────
    print(f"\n[1/4] Chargement de {PATH_PKL.name} ...")
    with open(PATH_PKL, "rb") as f:
        pkl = pickle.load(f)
    print(f"      {len(pkl)} trials chargés.")

    print(f"\n[2/4] Chargement de {PATH_JSON.name} ...")
    pid_to_expertise = _build_participant_expertise_map(PATH_JSON)
    print(f"      {len(pid_to_expertise)} participants mappés.")

    # ── Modèle 1 ──────────────────────────────────────────────────────────────
    print("\n[3/4] Construction du dataset Modèle 1 (features statistiques) ...")
    model1 = build_model1_dataset(pkl, pid_to_expertise)
    print(f"      {len(model1)} trials avec {len(FEATURE_NAMES)} features chacun.")
    _print_model1_summary(model1)

    with open(PATH_MODEL1, "wb") as f:
        pickle.dump(model1, f, protocol=4)
    print(f"\n  [OK] Sauvegarde : {PATH_MODEL1}")

    # ── Modèle 2 ──────────────────────────────────────────────────────────────
    print("\n[4/4] Construction du dataset Modèle 2 (séries brutes + augmentation) ...")
    model2 = build_model2_dataset(pkl, pid_to_expertise)
    print(f"      {len(model2)} enregistrements totaux (réels + synthétiques).")
    _print_model2_summary(model2)

    with open(PATH_MODEL2, "wb") as f:
        pickle.dump(model2, f, protocol=4)
    print(f"\n  [OK] Sauvegarde : {PATH_MODEL2}")

    print("\n" + "=" * 60)
    print("  Datasets prêts.")
    print("  clf_model1.pkl     -> Modele 1 : XGBoost / Random Forest")
    print("  clf_model2_aug.pkl -> Modele 2 : Hybrid1EVICEMS (softmax)")
    print("=" * 60)


if __name__ == "__main__":
    main()
