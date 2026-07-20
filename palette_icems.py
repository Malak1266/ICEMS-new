"""
palette_icems.py — Palette ORDINALE pour les 4 niveaux d'expertise + style publication.

POURQUOI PAS DE COULEURS CATEGORIELLES
--------------------------------------
L'expertise est une variable ORDINALE (Novice < Junior < Senior < Expert).
Une palette categorielle (rouge / jaune / bleu / vert, comme dans
fig1_temporal_stability.png) encode 4 categories SANS ordre : le lecteur doit
consulter la legende pour reconstruire la progression. Une rampe sequentielle
encode l'ordre de facon pre-attentive — on lit la progression sans legende.
C'est une regle standard de semiologie graphique (Bertin ; Ware, "Information
Visualization" ; Crameri et al. 2020, Nature Comm., sur les colormaps en science).

LA RAMPE RETENUE : sable chaud -> bleu profond froid
  Novice  #E8C49A  sable clair    | clair + chaud = "peu dense", debutant
  Junior  #C98F6B  terre          |
  Senior  #6B8FA8  bleu moyen     | bascule chaud->froid a mi-parcours
  Expert  #1F3D52  bleu nuit      | fonce + froid = "dense", maitrise

Trois canaux redondants, donc lisible meme degrade :
  1. LUMINOSITE decroissante monotone (L* 82 -> 26) -> survit au noir & blanc
  2. TEINTE chaud -> froid                          -> lecture rapide en couleur
  3. SATURATION stable                              -> aucun groupe ne "saute"

Le canal luminosite seul suffit : c'est ce qui rend la palette robuste au
daltonisme (deuteranopie/protanopie ~8 % des hommes) et a l'impression N&B.
"""
from __future__ import annotations
import numpy as np

GROUPS = ["Novice", "Junior", "Senior", "Expert"]
PALETTE = {"Novice": "#E8C49A", "Junior": "#C98F6B",
           "Senior": "#6B8FA8", "Expert": "#1F3D52"}
INK = "#2B2B2B"
GRID = "#D9DEE2"

# Variante monochrome bleue (si un relecteur juge la bichromie trop marquee)
PALETTE_MONO = {"Novice": "#C9D3D9", "Junior": "#9FB4BF",
                "Senior": "#5F7D8C", "Expert": "#2F4A58"}

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "axes.edgecolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "axes.axisbelow": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,   # polices vectorielles editables (exigence de nombreux journaux)
    "ps.fonttype": 42,
}

# Reglages courbes / bandes
LINEWIDTH = 2.4          # assez epais pour survivre a une reduction en colonne
BAND_ALPHA = 0.18        # bande visible sans masquer les courbes voisines
BAND_EDGE_ALPHA = 0.45   # liseré: delimite la bande quand plusieurs se superposent


def apply_style():
    import matplotlib
    from matplotlib import rcParams
    rcParams.update(STYLE)


def relative_luminance(hexcolor: str) -> float:
    """Luminance relative WCAG — sert a verifier la monotonie de la rampe."""
    c = hexcolor.lstrip("#")
    rgb = [int(c[i:i+2], 16) / 255 for i in (0, 2, 4)]
    lin = [v/12.92 if v <= 0.03928 else ((v+0.055)/1.055)**2.4 for v in rgb]
    return 0.2126*lin[0] + 0.7152*lin[1] + 0.0722*lin[2]


def simulate_cvd(hexcolor: str, kind: str = "deuteranopia") -> str:
    """Simulation approchee (Vienot et al. 1999) pour controle daltonisme."""
    c = hexcolor.lstrip("#")
    r, g, b = [int(c[i:i+2], 16)/255 for i in (0, 2, 4)]
    if kind == "deuteranopia":
        M = np.array([[0.625, 0.375, 0.0], [0.700, 0.300, 0.0], [0.0, 0.300, 0.700]])
    elif kind == "protanopia":
        M = np.array([[0.567, 0.433, 0.0], [0.558, 0.442, 0.0], [0.0, 0.242, 0.758]])
    else:
        M = np.array([[0.950, 0.050, 0.0], [0.0, 0.433, 0.567], [0.0, 0.475, 0.525]])
    out = np.clip(M @ np.array([r, g, b]), 0, 1)
    return "#" + "".join(f"{int(v*255):02X}" for v in out)


def check() -> bool:
    """Verifie la monotonie de luminosite (le critere qui rend la rampe robuste)."""
    L = [relative_luminance(PALETTE[g]) for g in GROUPS]
    mono = all(L[i] > L[i+1] for i in range(3))
    print("Luminance relative (doit DECROITRE strictement) :")
    for g, l in zip(GROUPS, L):
        print(f"  {g:<8} {PALETTE[g]}  L={l:.3f}   deuteranopie -> {simulate_cvd(PALETTE[g])}")
    print(f"  => monotone : {'OUI' if mono else 'NON'}")
    print(f"  contraste Novice/Expert : {(L[0]+0.05)/(L[3]+0.05):.1f}:1  (>=3:1 recommande)")
    return mono


if __name__ == "__main__":
    assert check(), "rampe non monotone"
