"""
calibration.py — Recalibration BORNÉE du score d'expertise ICEMS.

Contexte
--------
La tête du modèle est Dense(1, tanh) -> score_brut ∈ (-1, +1).
La moyenne des essais / des 10 paires reste, elle aussi, dans (-1, +1).
=> Le score composite BRUT ne peut PAS dépasser [-1, +1].

Le dépassement observé sur Panel A (valeurs à -1.6, +1.4, ...) provient
UNIQUEMENT de la calibration affine post-hoc  y = a*x + b  (pente a > 1,
moyennes des groupes extrêmes ancrées sur *exactement* ±1). Une droite
non bornée appliquée à des points déjà proches de ±1 les pousse dehors.

Ce module remplace cette calibration par un équivalent BORNÉ. Il s'applique
APRÈS le modèle gelé : il ne touche NI l'architecture, NI le prétraitement,
NI le split, NI l'ensemble de test. La comparabilité avec le papier est
préservée.

Trois modes
-----------
  "raw"     : identité. Scores tanh bruts, déjà dans (-1,1). = protocole du papier.
  "bounded" : recalibration affine dans l'espace logit atanh -> tanh.
              Décompresse/recentre COMME l'affine, mais reste dans (-1,1)
              par construction (recommandé pour une figure de soutenance).
  "clip"    : rustine np.clip(-1, 1). Borne mais empile les points sur ±1
              (moche, à réserver au dépannage). À déclarer si utilisé.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-4  # marge pour éviter atanh(±1) = ±inf


def _atanh_safe(x) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0 + EPS, 1.0 - EPS)
    return np.arctanh(x)


def fit_bounded_affine(raw_scores, groups,
                       group_lo: str = "Novice", group_hi: str = "Expert",
                       anchor_lo: float = -0.9, anchor_hi: float = 0.9):
    """
    Ajuste (a, b) dans l'espace logit pour que la MOYENNE du groupe bas tombe
    (après re-squash tanh) sur `anchor_lo`, et celle du groupe haut sur
    `anchor_hi`. |anchor| < 1  =>  sortie garantie dans (-1, 1).

    Choix de `anchor_lo/hi`
      * ±0.9  : ancres intérieures neutres (défaut).
      * -0.80 / +0.75 : aligne sur les moyennes publiées Hybrid 1 (comparaison directe).
      * Rien du tout : préfère alors mode="raw".
    """
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    groups = np.asarray(groups)
    z = _atanh_safe(raw_scores)
    z_lo = z[groups == group_lo].mean()
    z_hi = z[groups == group_hi].mean()
    if not (np.isfinite(z_lo) and np.isfinite(z_hi)) or abs(z_hi - z_lo) < 1e-9:
        raise ValueError("Ancrage impossible : moyennes des groupes extrêmes trop proches / manquantes.")
    t_lo, t_hi = _atanh_safe(anchor_lo), _atanh_safe(anchor_hi)
    a = float((t_hi - t_lo) / (z_hi - z_lo))
    b = float(t_lo - a * z_lo)
    return a, b


def apply_bounded_affine(raw_scores, a: float, b: float) -> np.ndarray:
    """y = tanh(a * atanh(x) + b) ∈ (-1, 1) par construction, monotone si a > 0."""
    return np.tanh(a * _atanh_safe(raw_scores) + b)


def calibrate(raw_scores, groups=None, mode: str = "bounded", **kw) -> np.ndarray:
    """Point d'entrée unique. Renvoie les scores calibrés (bornés sauf mode='raw' déjà borné)."""
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    if mode == "raw":
        return raw_scores
    if mode == "clip":
        return np.clip(raw_scores, -1.0, 1.0)
    if mode == "bounded":
        if groups is None:
            raise ValueError("mode='bounded' exige `groups` (pour ancrer les moyennes).")
        a, b = fit_bounded_affine(raw_scores, groups, **kw)
        return apply_bounded_affine(raw_scores, a, b)
    raise ValueError(f"mode inconnu : {mode!r} (attendu : raw | bounded | clip)")


if __name__ == "__main__":
    # Auto-test : la sortie bornée ne dépasse jamais [-1, 1].
    rng = np.random.default_rng(0)
    raw = np.tanh(rng.normal(0, 1.2, size=200))
    grp = np.array((["Novice"] * 50) + (["Junior"] * 50) + (["Senior"] * 50) + (["Expert"] * 50))
    y = calibrate(raw, grp, mode="bounded")
    assert y.max() <= 1.0 and y.min() >= -1.0, "ÉCHEC bornage"
    print(f"OK bounded : min={y.min():.4f}  max={y.max():.4f}  (dans [-1, 1])")
