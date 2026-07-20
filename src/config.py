"""
ICEMS — Configuration centralisée.

Mapping sublevel → score de régression (9 niveaux non-linéaires).
Ce module est la SOURCE UNIQUE DE VÉRITÉ pour Y9_TO_REG.

Tous les composants du pipeline (step_B_classification, icems_v3_train,
diagnose_senior_expert, sublevel_analysis) doivent importer Y9_TO_REG
depuis ce module plutôt que de le redéfinir localement.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List

import numpy as np

logger = logging.getLogger(__name__)

# ─── Mapping 9-niveaux non-linéaire ──────────────────────────────────────────
# Clés en minuscules — toujours accéder via sublevel.strip().lower() ou
# via la fonction normalize_sublevel().
#
# Justification des valeurs (non-linéarité intentionnelle) :
#   • MS → PGY5 : zone "non-expert" étendue, toujours négative
#   • PGY6 (+0.10) : pivot post-résidence, juste au-dessus de zéro
#   • Fellow (+0.55) → Staff (+1.00) : progression monotone, écart franc
#     pour éviter la rupture fellow→staff observée sur la Figure 7
SUBLEVEL_TO_SCORE: Dict[str, float] = {
    "ms":     -1.00,
    "pgy1":   -0.80,
    "pgy2":   -0.60,
    "pgy3":   -0.40,
    "pgy4":   -0.20,
    "pgy5":   +0.10,
    "pgy6":   +0.40,
    "fellow": +0.75,
    "staff":  +1.00,
}

# Ordre canonique aligné sur y9 ∈ [0, 8]
SUBLEVEL_ORDER: List[str] = [
    "ms", "pgy1", "pgy2", "pgy3", "pgy4", "pgy5", "pgy6", "fellow", "staff",
]

SUBLEVEL_TO_RANK: Dict[str, int] = {k: i for i, k in enumerate(SUBLEVEL_ORDER)}

SUBLEVEL_TO_TIER: Dict[str, int] = {
    "ms": 0,
    "pgy1": 0, "pgy2": 0, "pgy3": 0, "pgy4": 0, "pgy5": 0,
    "pgy6": 1,
    "fellow": 1,
    "staff": 2,
}

# Sanity-check interne : SUBLEVEL_ORDER doit couvrir exactement SUBLEVEL_TO_SCORE
assert set(SUBLEVEL_ORDER) == set(SUBLEVEL_TO_SCORE), (
    "SUBLEVEL_ORDER et SUBLEVEL_TO_SCORE sont désynchronisés — vérifier config.py"
)

# Y9_TO_REG dérivé de SUBLEVEL_TO_SCORE (source unique de vérité).
# Remplace toutes les définitions locales dispersées dans les modules src/.
Y9_TO_REG: np.ndarray = np.array(
    [SUBLEVEL_TO_SCORE[k] for k in SUBLEVEL_ORDER], dtype=np.float32
)

# ─── Table de normalisation des formes de sublevel ───────────────────────────
# Toutes les variantes orthographiques connues → clé canonique de SUBLEVEL_TO_SCORE.
# Utilisée par normalize_sublevel() pour accepter indifféremment :
#   - formes courtes  : "MS", "PGY1", "Fellow"
#   - formes longues  : "Medical student", "Staff"
#   - formes papier   : "Medical Student", "Resident PGY1", "Neurosurgeon"
_SUBLEVEL_ALIAS: Dict[str, str] = {
    # ── Formes courtes (SUBLEVEL_SHORT) ──────────────────────────────────────
    "ms":     "ms",
    "pgy1":   "pgy1",
    "pgy2":   "pgy2",
    "pgy3":   "pgy3",
    "pgy4":   "pgy4",
    "pgy5":   "pgy5",
    "pgy6":   "pgy6",
    "fellow": "fellow",
    "staff":  "staff",
    # ── Formes longues (SUBLEVEL_NAMES / build_continuous_dataset.py) ────────
    "medical student":  "ms",
    "pgy 1":            "pgy1",
    "pgy 2":            "pgy2",
    "pgy 3":            "pgy3",
    "pgy 4":            "pgy4",
    "pgy 5":            "pgy5",
    "pgy 6":            "pgy6",
    # ── Formes papier (PAPER_SUBLEVEL_BY_Y9 / enrich_baseline_pkl.py) ────────
    "medical student":  "ms",          # déjà ci-dessus, confirme
    "resident pgy1":    "pgy1",
    "resident pgy2":    "pgy2",
    "resident pgy3":    "pgy3",
    "resident pgy4":    "pgy4",
    "resident pgy5":    "pgy5",
    "resident pgy6":    "pgy6",
    "neurosurgeon":     "staff",
}


def normalize_sublevel(raw: str) -> str:
    """Normalise toute forme de sublevel vers la clé canonique de SUBLEVEL_TO_SCORE.

    Pipeline : raw.strip().lower() → lookup exact → lookup sous-chaîne (décroissant).

    Parameters
    ----------
    raw : str
        Valeur brute du champ sublevel ou level (ex : "Medical Student", "PGY1",
        "Resident PGY3", "Neurosurgeon").

    Returns
    -------
    str
        Clé canonique (ex : "ms", "pgy3", "staff").
        Si non reconnue, retourne raw.strip().lower() — KeyError levée en aval.
    """
    key = raw.strip().lower()
    if key in _SUBLEVEL_ALIAS:
        return _SUBLEVEL_ALIAS[key]
    # Fallback par sous-chaîne (alias les plus longs en premier pour précision maximale)
    for alias in sorted(_SUBLEVEL_ALIAS, key=len, reverse=True):
        if alias in key:
            return _SUBLEVEL_ALIAS[alias]
    return key


def sublevel_score(raw: str) -> float:
    """Retourne le score de régression pour un sous-niveau (toute forme acceptée).

    Parameters
    ----------
    raw : str
        Valeur brute du champ sublevel ou level.

    Returns
    -------
    float
        Score de régression dans [-1.0, +1.0].

    Raises
    ------
    KeyError
        Si le sous-niveau n'est pas reconnu dans SUBLEVEL_TO_SCORE, avec message
        explicite listant les valeurs attendues.
    """
    canonical = normalize_sublevel(raw)
    if canonical not in SUBLEVEL_TO_SCORE:
        raise KeyError(
            f"Sublevel inconnu : {raw!r} (normalisé : {canonical!r}). "
            f"Valeurs attendues : {list(SUBLEVEL_TO_SCORE.keys())}"
        )
    return SUBLEVEL_TO_SCORE[canonical]


def assert_all_sublevels_known(records: Iterable[dict]) -> None:
    """Vérifie que tous les sublevels présents dans les records sont dans SUBLEVEL_TO_SCORE.

    Priorité de lookup : champ ``sublevel`` > champ ``level`` > validation ``y9`` range.
    Lève KeyError explicite si un sublevel est inconnu.

    Parameters
    ----------
    records : Iterable[dict]
        Enregistrements du dataset (values() du dict retourné par _normalize_dataset).

    Raises
    ------
    KeyError
        Liste tous les sublevels non reconnus en une seule erreur.
    """
    unknown: set[str] = set()
    for rec in records:
        raw_sl: str = rec.get("sublevel") or rec.get("level", "")
        if raw_sl:
            canonical = normalize_sublevel(str(raw_sl))
            if canonical not in SUBLEVEL_TO_SCORE:
                unknown.add(raw_sl)
        # Toujours vérifier que y9 est dans [0, 8] (clé primaire toujours présente)
        y9 = rec.get("y9")
        if y9 is not None and not (0 <= int(y9) <= 8):
            raise KeyError(
                f"y9={y9!r} hors plage [0, 8] dans l'enregistrement {rec.get('name', '?')!r}."
            )
    if unknown:
        raise KeyError(
            f"Sublevels inconnus dans le dataset : {sorted(unknown)}.\n"
            f"Valeurs attendues : {list(SUBLEVEL_TO_SCORE.keys())}\n"
            f"Vérifiez normalize_sublevel() ou _SUBLEVEL_ALIAS dans src/config.py."
        )


def log_y_distribution(y_values: Iterable[float], label: str = "y_reg") -> None:
    """Logue et affiche la distribution des y après mapping (vérification visuelle).

    Pour chaque valeur unique de y, affiche :
      - la clé sublevel correspondante dans SUBLEVEL_TO_SCORE
      - la valeur numérique
      - le nombre de trials

    Parameters
    ----------
    y_values : Iterable[float]
        Valeurs de y_reg après application du mapping SUBLEVEL_TO_SCORE.
    label : str
        Nom du champ pour l'affichage (défaut : "y_reg").
    """
    arr = np.asarray(list(y_values), dtype=np.float32)
    if arr.size == 0:
        msg = f"[config] log_y_distribution : aucune valeur dans {label!r}."
        logger.warning(msg)
        print(msg)
        return

    unique_vals, counts = np.unique(arr, return_counts=True)

    header = f"[config] Distribution {label} ({arr.size} trials) :"
    logger.info(header)
    print(header)

    for val, cnt in zip(unique_vals, counts):
        # Trouver la clé sublevel correspondante (correspondance exacte à 1e-4 près)
        sl = next(
            (k for k, v in SUBLEVEL_TO_SCORE.items() if abs(float(v) - float(val)) < 1e-4),
            "?",
        )
        line = f"  {sl.ljust(8)} (y={val:+.2f}) : {cnt:3d} trials"
        logger.info(line)
        print(line)

    stats = (
        f"  -> mean={arr.mean():+.4f}  std={arr.std():.4f}  "
        f"min={arr.min():+.4f}  max={arr.max():+.4f}"
    )
    logger.info(stats)
    print(stats)
