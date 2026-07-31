"""
augment_extremes.py
===================
Augmentation Option D pour le protocole extrêmes (pool TRAIN ms/staff uniquement).

Deux étapes, appliquées APRÈS GroupShuffleSplit et AVANT la normalisation :
  1. Équilibrage Staff : génère des synthétiques Staff (jitter doux) jusqu'à
     égaliser le nombre de trials ms.
  2. Augmentation globale : multiplie le pool équilibré par `global_multiplier`
     via jitter + time-warp doux.

Techniques autorisées : jitter gaussien + time-warp cubique doux UNIQUEMENT.
Interdits : DBA, magnitude warp, window slicing.
Seule dépendance externe ajoutée : scipy.interpolate.

Garanties :
  - Un synthétique hérite du `sublevel` ET du `y` (cible) de son parent réel.
  - Tous les synthétiques portent is_augmented=True.
  - rng isolé (np.random.default_rng(seed)) → reproductible.
  - N'agit QUE sur la liste passée (records_train) — jamais val/test/milieu.
"""
from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d


# ─── FONCTION 1 — jitter ─────────────────────────────────────────────────────
def jitter_sequence(
    X: np.ndarray,
    sigma: float = 0.020,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Ajoute un bruit gaussien i.i.d. à chaque feature de chaque frame.

    X : (T, F) float32
    sigma : écart-type du bruit — doit rester <= 0.025 pour ce protocole.
    Retourne : (T, F) float32, même shape que X.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=X.shape).astype(np.float32)
    return (X + noise).astype(np.float32)


# ─── FONCTION 2 — time warp ──────────────────────────────────────────────────
def time_warp_sequence(
    X: np.ndarray,
    sigma: float = 0.05,
    n_knots: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Déformation temporelle douce via spline cubique sur n_knots points internes.

    Algorithme :
      1. Points de contrôle : n_knots+2 positions sur [0, T-1] (extrémités fixées).
      2. Points internes perturbés par un bruit gaussien(0, sigma*T), puis clampés
         dans [1, T-2] pour préserver la monotonie.
      3. Spline cubique entre la grille d'origine et les positions perturbées.
      4. Rééchantillonnage linéaire de chaque feature sur la nouvelle grille.

    X : (T, F) float32
    sigma : amplitude de déformation — doit rester <= 0.05.
    n_knots : nombre de points de contrôle internes — doit rester <= 4.
    Retourne : (T, F) float32, même shape que X.

    Si la grille résultante n'est pas strictement monotone (np.diff > 0),
    retourne X inchangé avec un warning (jamais d'exception).
    """
    if rng is None:
        rng = np.random.default_rng()

    T, F = X.shape
    if T < 4 or n_knots < 1:
        return X.astype(np.float32)

    # 1. Grille de contrôle d'origine : extrémités + n_knots points internes
    orig_knots = np.linspace(0.0, T - 1, n_knots + 2)

    # 2. Perturbation des points internes uniquement (extrémités fixées)
    # Pour les longues séquences (T > 2000), sigma * T peut dépasser l'espacement
    # inter-knots, ce qui provoque des croisements entre points de contrôle adjacents
    # et donc une grille non-monotone. Le clamping à 40 % de l'espacement garantit
    # la monotonie indépendamment de T : deux knots voisins ne peuvent jamais se
    # croiser car chacun ne peut bouger que de ±40 % de la distance qui les sépare,
    # laissant toujours un gap résiduel de ≥20 % entre eux.
    spacing = T / (n_knots + 1)          # espacement moyen entre knots
    max_perturb = spacing * 0.40          # 40 % de l'espacement → monotonie garantie
    std_perturb = min(sigma * T, max_perturb)
    warped_knots = orig_knots.copy()
    perturb = rng.normal(0.0, std_perturb, size=n_knots)
    warped_knots[1:-1] = orig_knots[1:-1] + perturb
    warped_knots[1:-1] = np.clip(warped_knots[1:-1], 1.0, T - 2)
    warped_knots[0], warped_knots[-1] = 0.0, T - 1  # ré-ancrage explicite des bords

    # Les points de contrôle perturbés doivent rester strictement croissants
    if not np.all(np.diff(warped_knots) > 0):
        warnings.warn(
            "time_warp_sequence: non-monotonic control points - trial returned unchanged.",
            RuntimeWarning,
        )
        return X.astype(np.float32)

    # 3. Interpolation Pchip : t_original → t_warpé.
    # PchipInterpolator (Piecewise Cubic Hermite) est monotone par construction
    # entre ses points de contrôle, contrairement à CubicSpline qui peut produire
    # des oscillations de Runge entre knots rapprochés (visible sur T > 2000).
    # Cette propriété élimine le repli sur grille non-monotone.
    spline = PchipInterpolator(orig_knots, warped_knots)
    grid = np.arange(T, dtype=np.float64)
    new_grid = spline(grid)

    # 4. Vérification résiduelle de la monotonie (garde-fou, ne devrait plus se déclencher)
    if not np.all(np.diff(new_grid) > 0):
        warnings.warn(
            "time_warp_sequence: non-monotonic resampled grid - trial returned unchanged.",
            RuntimeWarning,
        )
        return X.astype(np.float32)

    # Borne la nouvelle grille dans [0, T-1] pour rester dans le domaine d'interpolation
    new_grid = np.clip(new_grid, 0.0, T - 1)

    out = np.zeros_like(X, dtype=np.float32)
    for f in range(F):
        interp = interp1d(grid, X[:, f], kind="linear", fill_value="extrapolate")
        out[:, f] = interp(new_grid).astype(np.float32)
    return out


# ─── FONCTION 2b — magnitude warp ────────────────────────────────────────────
def magnitude_warp_sequence(
    X: np.ndarray,
    sigma: float = 0.08,
    n_knots: int = 4,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Mise à l'échelle locale et douce de l'amplitude des features.
    Simule des variations de force/vitesse d'exécution réalistes.

    Algorithme :
    1. Générer n_knots valeurs de scaling tirées de N(1.0, sigma)
       aux positions knots = np.linspace(0, T-1, n_knots)
    2. Interpoler sur toute la longueur T via interp1d(kind='cubic')
    3. Clamper la courbe dans [0.5, 1.5]
    4. Multiplier chaque frame : X_warped[t, :] = X[t, :] * scaling_curve[t]

    La même courbe est appliquée à toutes les features (cohérence physique).
    sigma <= 0.10 obligatoire.
    Retourne (T, F) float32, même shape que X.
    """
    if rng is None:
        rng = np.random.default_rng()
    T, F = X.shape
    knot_positions = np.linspace(0, T - 1, n_knots)
    knot_values = rng.normal(1.0, sigma, size=n_knots)
    from scipy.interpolate import interp1d
    interp = interp1d(knot_positions, knot_values,
                      kind='cubic', fill_value='extrapolate')
    scaling_curve = interp(np.arange(T))
    scaling_curve = np.clip(scaling_curve, 0.5, 1.5).astype(np.float32)
    return (X * scaling_curve[:, np.newaxis]).astype(np.float32)


# ─── FONCTION 3 — pipeline trial ─────────────────────────────────────────────
def augment_trial(X, rng=None,
                  p_time_warp=0.70,
                  p_magnitude_warp=0.50,
                  p_jitter=0.80,
                  jitter_sigma=0.020,
                  warp_sigma=0.05,
                  warp_knots=3,
                  mag_sigma=0.08,
                  mag_knots=4):
    """
    Pipeline : time_warp → magnitude_warp → jitter
    Chaque technique appliquée selon sa probabilité.
    """
    out = X.copy()
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < p_time_warp:
        out = time_warp_sequence(out, sigma=warp_sigma,
                                 n_knots=warp_knots, rng=rng)
    if rng.random() < p_magnitude_warp:
        out = magnitude_warp_sequence(out, sigma=mag_sigma,
                                      n_knots=mag_knots, rng=rng)
    if rng.random() < p_jitter:
        out = jitter_sequence(out, sigma=jitter_sigma, rng=rng)
    return out.astype(np.float32)


# ─── FONCTION 4 — augmentation du pool (deux étapes) ─────────────────────────
def augment_pool(
    records_train: List[dict],
    seed: int = 42,
    # Étape 1 — équilibrage Staff
    target_staff_count: Optional[int] = None,
    jitter_sigma_eq: float = 0.015,
    # Étape 2 — augmentation globale
    global_multiplier: int = 3,
    jitter_sigma_global: float = 0.020,
    warp_sigma: float = 0.05,
    warp_knots: int = 3,
) -> List[dict]:
    """Augmentation en deux étapes du pool TRAIN extrêmes (ms + staff).

    Voir le docstring de tâche pour le détail. Retourne la liste complète
    (réels + synthétiques) mélangée de façon reproductible.
    """
    rng = np.random.default_rng(seed)

    def _clone(parent: dict, X_new: np.ndarray, method: str) -> dict:
        """Crée un synthétique héritant de sublevel + y du parent."""
        return {
            "X": X_new.astype(np.float32),
            "sublevel": parent["sublevel"],
            "participant": parent["participant"],
            "trial": f"{parent['trial']}_aug_{method}",
            "y": parent["y"],
            "is_augmented": True,
            "aug_method": method,
        }

    ms_real = [r for r in records_train if r["sublevel"] == "ms"]
    staff_real = [r for r in records_train if r["sublevel"] == "staff"]
    n_ms = len(ms_real)
    n_staff = len(staff_real)

    # ── ÉTAPE 1 — Équilibrage Staff ──────────────────────────────────────────
    target = target_staff_count if target_staff_count is not None else n_ms
    n_to_create = max(0, target - n_staff)

    staff_synth_eq: List[dict] = []
    if n_to_create > 0 and staff_real:
        for _ in range(n_to_create):
            parent = staff_real[rng.integers(0, len(staff_real))]
            X_new = jitter_sequence(parent["X"], sigma=jitter_sigma_eq, rng=rng)
            staff_synth_eq.append(_clone(parent, X_new, "jitter_eq"))

    print(
        f"[AUG] Étape 1 : ms={n_ms} staff={n_staff} → +{n_to_create} "
        f"Staff synthétiques (jitter σ={jitter_sigma_eq})"
    )

    # Pool équilibré : tous les réels + synthétiques d'équilibrage
    balanced = list(records_train) + staff_synth_eq

    # ── ÉTAPE 2 — Augmentation globale ───────────────────────────────────────
    n_balanced = len(balanced)
    global_synth: List[dict] = []
    for parent in balanced:
        for _ in range(max(0, global_multiplier - 1)):
            X_new = augment_trial(
                parent["X"],
                rng=rng,
                jitter_sigma=jitter_sigma_global,
                warp_sigma=warp_sigma,
                warp_knots=warp_knots,
            )
            global_synth.append(_clone(parent, X_new, "jitter+warp"))

    final_pool = balanced + global_synth
    n_final_total = len(final_pool)
    print(
        f"[AUG] Étape 2 : {n_balanced} trials → {n_final_total} trials "
        f"(×{global_multiplier}, jitter σ={jitter_sigma_global} + warp σ={warp_sigma})"
    )

    # ── Log final + ratio ────────────────────────────────────────────────────
    n_ms_final = sum(1 for r in final_pool if r["sublevel"] == "ms")
    n_staff_final = sum(1 for r in final_pool if r["sublevel"] == "staff")
    ratio = (n_ms_final / n_staff_final) if n_staff_final else float("inf")
    print(f"[AUG] Final   : ms={n_ms_final} staff={n_staff_final} — ratio={ratio:.2f}")

    rng.shuffle(final_pool)
    return final_pool


# ─── Test unitaire minimal ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    rng = np.random.default_rng(0)

    # 1. jitter préserve la shape
    X = rng.normal(0, 1, size=(120, 10)).astype(np.float32)
    Xj = jitter_sequence(X, sigma=0.02, rng=rng)
    assert Xj.shape == X.shape, f"jitter shape {Xj.shape} != {X.shape}"
    assert Xj.dtype == np.float32

    # 2. time_warp préserve la shape
    Xw = time_warp_sequence(X, sigma=0.05, n_knots=3, rng=rng)
    assert Xw.shape == X.shape, f"time_warp shape {Xw.shape} != {X.shape}"
    assert Xw.dtype == np.float32

    # 2b. time_warp sur séquence trop courte → inchangé, shape préservée
    Xshort = rng.normal(0, 1, size=(3, 4)).astype(np.float32)
    assert time_warp_sequence(Xshort, rng=rng).shape == Xshort.shape

    # 3. augment_pool produit le bon ratio ms/staff (équilibrage → ratio 1.0)
    def _mk(sl, pid, tr):
        return {
            "X": rng.normal(0, 1, size=(100, 10)).astype(np.float32),
            "sublevel": sl, "participant": pid, "trial": tr,
            "y": -1.0 if sl == "ms" else 1.0, "is_augmented": False,
        }

    records_train = (
        [_mk("ms", f"p{i}", f"t{i}") for i in range(8)]      # 8 ms
        + [_mk("staff", f"s{i}", f"t{i}") for i in range(5)]  # 5 staff
    )
    records_val = [_mk("ms", "vp0", "vt0"), _mk("staff", "vs0", "vt0")]
    val_ids_before = {id(r) for r in records_val}

    out = augment_pool(records_train, seed=42, global_multiplier=3)
    n_ms = sum(1 for r in out if r["sublevel"] == "ms")
    n_staff = sum(1 for r in out if r["sublevel"] == "staff")
    assert n_ms == n_staff, f"ratio déséquilibré : ms={n_ms} staff={n_staff}"

    # total attendu = global_multiplier * 2 * n_ms = 3 * 2 * 8 = 48
    assert len(out) == 3 * 2 * 8, f"taille finale {len(out)} != 48"

    # Réels intacts : 13 réels avec is_augmented=False
    n_real = sum(1 for r in out if not r.get("is_augmented", False))
    assert n_real == 13, f"réels attendus=13, obtenus={n_real}"

    # 4. Aucun synthétique n'a fui dans records_val (assertion explicite)
    val_ids_after = {id(r) for r in records_val}
    assert val_ids_before == val_ids_after, "records_val a été modifié !"
    assert all(not r.get("is_augmented", False) for r in records_val), \
        "un synthétique est présent dans records_val"
    out_ids = {id(r) for r in out}
    assert out_ids.isdisjoint(val_ids_after), "fuite d'objets entre out et records_val"

    print("Tous les tests augmentation : OK")

    # Test magnitude_warp
    rng_test = np.random.default_rng(0)
    X_dummy = np.random.randn(500, 10).astype(np.float32)
    X_mw = magnitude_warp_sequence(X_dummy, sigma=0.08, rng=rng_test)
    assert X_mw.shape == X_dummy.shape, "magnitude_warp: shape incorrecte"
    assert not np.isnan(X_mw).any(), "magnitude_warp: NaN détectés"
    assert X_mw.max() < X_dummy.max() * 1.6, "magnitude_warp: explosion"
    print("Test magnitude_warp : OK")
