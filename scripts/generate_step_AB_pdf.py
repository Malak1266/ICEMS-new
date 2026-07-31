"""
Génère docs/ICEMS_Step_A_B_Schemas.pdf — schémas explicatifs Step A & Step B.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "ICEMS_Step_A_B_Schemas.pdf"

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
GRAY = "#666666"
LIGHT = "#f0f4f8"
LOCK = "#8e44ad"


def box(ax, x, y, w, h, text, fc="white", ec=BLUE, fontsize=9, bold=False):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fc, edgecolor=ec, linewidth=1.5,
    )
    ax.add_patch(p)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, wrap=True)


def arrow(ax, x1, y1, x2, y2, color=GRAY):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=12, color=color, linewidth=1.2,
    ))


def page_title(fig, title, subtitle=""):
    fig.text(0.5, 0.96, title, ha="center", fontsize=16, weight="bold")
    if subtitle:
        fig.text(0.5, 0.92, subtitle, ha="center", fontsize=10, color=GRAY)


def page_step_a_pipeline(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    page_title(fig, "Step A — Pipeline DBA + Jitter",
               "step_A_v3_dba_jitter.py  ·  continuous_per_trial.pkl → augmented_v4.pkl")

    y = 9.0
    box(ax, 2.5, y, 5, 0.7, "continuous_per_trial.pkl\n136 trials réels", fc=LIGHT, bold=True)
    arrow(ax, 5, y, 5, y - 0.5)
    y -= 1.2
    box(ax, 2.0, y, 6, 0.7, "Pour chaque classe : Student · Junior · Senior · Expert", fc="#fff3e0", ec=ORANGE)
    arrow(ax, 5, y, 5, y - 0.5)
    y -= 1.2
    box(ax, 1.5, y, 7, 0.8, "① Tirer N_PARENTS = 6 trials réels (même classe)\n② Extraire lignes 2→7 · rééchantillonner max 500 frames", fc=LIGHT)
    arrow(ax, 5, y, 5, y - 0.5)
    y -= 1.3
    box(ax, 2.0, y, 6, 0.8, "③ DBA — dtw_barycenter_averaging\nMoyenne DTW → 1 séquence synthétique", fc="#e8f5e9", ec=GREEN, bold=True)
    arrow(ax, 5, y, 5, y - 0.5)
    y -= 1.3
    box(ax, 2.0, y, 6, 0.8, "④ Reconstruire data : lignes 0-1 = labels parent\nlignes 2-7 = résultat DBA", fc=LIGHT)
    arrow(ax, 5, y, 5, y - 0.5)
    y -= 1.3
    box(ax, 2.0, y, 6, 0.8, "⑤ Jitter — bruit gaussien σ = 0.03 × std_canal\nUniquement sur lignes 2-7", fc="#fff8e1", ec=ORANGE)
    arrow(ax, 5, y, 5, y - 0.5)
    y -= 1.3
    box(ax, 2.5, y, 5, 0.8, "augmented_v4.pkl\n136 réels + 16 DBA + 16 jitter = 168", fc=LIGHT, ec=GREEN, bold=True)

    ax.text(0.3, 0.5, "Règle LOPO : participant = 1er parent  ·  is_augmented = True  ·  jamais mélanger les classes",
            fontsize=9, color=GRAY, style="italic")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_step_a_data(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    page_title(fig, "Step A — Structure de data (C × T)",
               "Matrice canaux × temps  ·  T ≈ 150–500 frames (rééchantillonné à 500 pour DBA)")

    rows = [
        (0, "Ligne 0", "Label expertise_idx", LOCK, "CONSTANT"),
        (1, "Ligne 1", "Label level_idx", LOCK, "CONSTANT"),
        (2, "Ligne 2", "Position Magnitude", BLUE, "DBA + Jitter"),
        (3, "Ligne 3", "Velocity", BLUE, "DBA + Jitter"),
        (4, "Ligne 4", "Acceleration", BLUE, "DBA + Jitter"),
        (5, "Ligne 5", "Jerk (discriminant)", RED, "DBA + Jitter"),
        (6, "Ligne 6", "Distance Bipolar–Cavitron", BLUE, "DBA + Jitter"),
        (7, "Ligne 7", "Distance Bipolar–Scissors", BLUE, "DBA + Jitter"),
    ]
    y = 8.5
    for _, label, desc, color, zone in rows:
        box(ax, 0.5, y, 2.2, 0.65, label, fc=color, ec=color, fontsize=8)
        box(ax, 2.9, y, 4.5, 0.65, desc, fc="white", ec=color, fontsize=9)
        box(ax, 7.6, y, 2.0, 0.65, zone, fc="#f5f5f5", ec=GRAY, fontsize=8)
        y -= 0.85

    box(ax, 0.5, 1.2, 9, 1.5,
        "Signal d'expertise :  Novice (Student) → jerk élevé, gestes saccadés → rugosité HAUTE\n"
        "                      Expert          → jerk faible, gestes lisses    → rugosité BASSE\n\n"
        "Ordre attendu :  Student > Junior > Senior > Expert  (médiane rugosité)",
        fc=LIGHT, ec=GRAY, fontsize=9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_step_a_dba_example(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    page_title(fig, "Step A — Exemple : 6 parents → 1 DBA → 1 Jitter",
               "Canal Jerk (ligne 5)  ·  une classe à la fois")

    for i, xi in enumerate([0.8, 2.2, 3.6, 5.0, 6.4, 7.8]):
        box(ax, xi, 7.5, 1.0, 0.6, f"Parent\n{i+1}", fc="#eeeeee", ec=GRAY, fontsize=8)
        arrow(ax, xi + 0.5, 7.5, 5, 6.5)

    box(ax, 3.5, 5.8, 3, 0.9, "DBA\nMoyenne DTW", fc="#ffebee", ec=RED, bold=True)
    arrow(ax, 5, 5.8, 5, 4.8)
    box(ax, 3.5, 3.8, 3, 0.9, "Séquence DBA\n(participant = Parent 1)", fc="#e3f2fd", ec=BLUE)
    arrow(ax, 5, 3.8, 5, 2.8)
    box(ax, 3.5, 1.8, 3, 0.9, "DBA + Jitter\naug_type = dba+jitter", fc="#fff3e0", ec=ORANGE, bold=True)

    ax.text(5, 0.8, "× n_dba_per_class = 4 par classe  ·  × 4 classes  ·  seed = 42",
            ha="center", fontsize=9, color=GRAY)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_step_b_lopo(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    page_title(fig, "Step B — LOPO (Leave-One-Participant-Out)",
               "step_B_classification.py  ·  47 participants  ·  GRU Causal + score continu")

    box(ax, 3, 8.5, 4, 0.7, "136 trials · 47 participants", fc=LIGHT, bold=True)
    arrow(ax, 5, 8.5, 5, 8.0)
    box(ax, 2.5, 7.1, 5, 0.7, "Fold k : tenir OUT participant P_k", fc="#fff3e0", ec=ORANGE, bold=True)

    box(ax, 0.3, 4.8, 4.2, 2.0, "TRAIN\n46 participants\n(trials réels)", fc="#e8f5e9", ec=GREEN, fontsize=9)
    box(ax, 5.5, 4.8, 4.2, 2.0, "TEST\nP_k seulement\n(trials réels)", fc="#ffebee", ec=RED, fontsize=9)

    arrow(ax, 2.4, 4.8, 2.4, 4.2)
    box(ax, 0.3, 3.0, 4.2, 1.0, "+ DBA+jitter inline\néquilibrer classes", fc="#fff8e1", ec=ORANGE, fontsize=9)
    arrow(ax, 2.4, 3.0, 2.4, 2.4)
    box(ax, 0.3, 1.2, 4.2, 1.0, "Entraîner GRU\n40 epochs", fc="#e3f2fd", ec=BLUE, fontsize=9, bold=True)
    arrow(ax, 4.5, 1.7, 5.5, 5.5)
    box(ax, 5.5, 1.2, 4.2, 1.0, "Prédire score\nsur TEST réel", fc="#f3e5f5", ec=LOCK, fontsize=9, bold=True)

    arrow(ax, 7.6, 1.2, 7.6, 0.5)
    box(ax, 5.5, 0.1, 4.2, 0.7, "× 47 folds → métriques", fc=LIGHT, ec=GREEN, bold=True)

    ax.text(5, 3.5, "→", fontsize=24, ha="center", color=GRAY)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_step_b_inline_aug(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    page_title(fig, "Step B — Augmentation inline par fold",
               "Exemple fold 1 : participant 01020614 tenu out")

    box(ax, 1, 8.2, 8, 0.7, "Train réel : 135 trials (136 − 3 du participant exclu)", fc=LIGHT)
    box(ax, 1, 7.0, 8, 1.0,
        "Comptage :  Student=41  Junior=40  Senior=29  Expert=25\n"
        "Cible = majoritaire = 41 trials/classe",
        fc="#fff3e0", ec=ORANGE, fontsize=9)

    items = [
        ("Student", "41 → 41", "0 synth", GREEN),
        ("Junior",  "40 → 41", "+1 synth", ORANGE),
        ("Senior",  "29 → 41", "+12 synth", ORANGE),
        ("Expert",  "25 → 41", "+16 synth", ORANGE),
    ]
    y = 5.8
    for cls, change, synth, col in items:
        box(ax, 1, y, 2.5, 0.55, cls, fc=col, ec=col, fontsize=9)
        box(ax, 3.7, y, 2.5, 0.55, change, fc="white", ec=col, fontsize=9)
        box(ax, 6.3, y, 2.7, 0.55, synth, fc="#f5f5f5", ec=GRAY, fontsize=9)
        y -= 0.7

    arrow(ax, 5, 5.0, 5, 4.5)
    box(ax, 1, 3.2, 8, 1.0,
        "Pour chaque synthétique :\n"
        "① 6 parents (même classe, train seulement)  ② DBA  ③ Jitter 3%  ④ participant = synth_XXX",
        fc="#e8f5e9", ec=GREEN, fontsize=9)
    arrow(ax, 5, 3.2, 5, 2.7)
    box(ax, 2, 1.5, 6, 0.9, "[aug] +35 synthétiques  →  Train final ≈ 170 trials",
        fc=LIGHT, ec=BLUE, bold=True)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_step_b_rules(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    page_title(fig, "Step B — Règles anti-fuite & architecture")

    box(ax, 0.4, 6.8, 4.5, 2.5,
        "AUTORISÉ\n\n"
        "• DBA sur train du fold seulement\n"
        "• Parents = trials réels du train\n"
        "• Synthétiques : participant = synth_XXX\n"
        "• Normalisation sur train du fold\n"
        "• Test = réels du participant tenu out",
        fc="#e8f5e9", ec=GREEN, fontsize=9)
    box(ax, 5.1, 6.8, 4.5, 2.5,
        "INTERDIT\n\n"
        "• Utiliser P_k pour générer des DBA\n"
        "• Synthétiques en test\n"
        "• Normalisation globale\n"
        "• Mélanger classes dans un DBA\n"
        "• Modifier lignes 0-1 (labels)",
        fc="#ffebee", ec=RED, fontsize=9)

    box(ax, 0.4, 3.5, 9.2, 2.8,
        "Architecture GRU Causal\n\n"
        "X (T, 10)  →  Linear+GELU  →  GRU 2 couches (64)  →  score/frame ∈ [-1,+1]\n"
        "                              ↓\n"
        "                    Agrégation pondérée (valid_ratio)  →  1 score / trial\n"
        "                              ↓\n"
        "              Loss = MSE + pénalité ordinale (Expert > Senior > Junior > Student)",
        fc="#e3f2fd", ec=BLUE, fontsize=9)

    box(ax, 0.4, 0.8, 9.2, 2.2,
        "Classes & scores cibles\n\n"
        "Student  y_reg = -1.00   (42 trials)     Junior   y_reg = -0.33   (41 trials)\n"
        "Senior   y_reg = +0.33   (31 trials)     Expert   y_reg = +1.00   (22 trials)\n\n"
        "Métriques finales : Pearson r  ·  Spearman r  ·  Accuracy 4 classes",
        fc=LIGHT, ec=GRAY, fontsize=9)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_comparison(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    page_title(fig, "Step A vs Step B — Comparaison",
               "ICEMS — Évaluation automatique d'expertise chirurgicale")

    box(ax, 0.5, 5.5, 4.3, 3.5,
        "STEP A\nstep_A_v3_dba_jitter.py\n\n"
        "Quand : AVANT le ML\n"
        "Où : tout le dataset\n"
        "Sortie : augmented_v4.pkl\n"
        "Rôle : exploration, visualisation\n"
        "Paramètres : n_parents=6, n_dba=4/classe",
        fc="#fff3e0", ec=ORANGE, fontsize=9)
    box(ax, 5.2, 5.5, 4.3, 3.5,
        "STEP B\nstep_B_classification.py\n\n"
        "Quand : PENDANT chaque fold LOPO\n"
        "Où : train du fold seulement\n"
        "Sortie : results/level1_v4/\n"
        "Rôle : évaluation rigoureuse\n"
        "Run Narval : 47 folds × 40 epochs",
        fc="#e8f5e9", ec=GREEN, fontsize=9)

    box(ax, 2.5, 4.2, 5, 0.7, "continuous_per_trial.pkl (136 trials réels)", fc=LIGHT, bold=True)
    arrow(ax, 3.5, 4.2, 2.5, 5.5)
    arrow(ax, 6.5, 4.2, 7.5, 5.5)

    box(ax, 0.5, 1.5, 9, 2.2,
        "Timeline d'un fold (Narval, CPU)\n\n"
        "[LOPO fold k/47]  →  [aug] +N synthétiques (~1-3 min)  →  Entraînement 40 epochs (~30-90 min, silencieux)\n"
        "  →  Early stopping / fin fold  →  fold suivant\n\n"
        "Commande : python src/step_B_classification.py --data data/continuous_per_trial.pkl "
        "--out results/level1_v4 --epochs 40",
        fc=LIGHT, ec=BLUE, fontsize=8)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        d = pdf.infodict()
        d["Title"] = "ICEMS — Schémas Step A & Step B"
        d["Author"] = "ICEMS Project"
        d["Subject"] = "DBA + Jitter augmentation & LOPO classification"

        page_step_a_pipeline(pdf)
        page_step_a_data(pdf)
        page_step_a_dba_example(pdf)
        page_step_b_lopo(pdf)
        page_step_b_inline_aug(pdf)
        page_step_b_rules(pdf)
        page_comparison(pdf)

    print(f"PDF généré : {OUT}")


if __name__ == "__main__":
    main()
