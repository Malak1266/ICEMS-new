"""Shared constants for publication figures (Hybrid extremes protocol)."""
from __future__ import annotations

from pathlib import Path

# ── Protocol anchors (must stay consistent with train_hybrid_extremes) ───────
DEFAULT_RUN_DIR = Path("results/hybrid_extremes")
DEFAULT_PKL = Path("data/continuous_per_trial.pkl")
DEFAULT_CSV = Path("data/Exvivo_trial_Participants(Sheet1).csv")
DEFAULT_OUT = Path("results/publication_figures")

CLASS4_ORDER = ("student", "junior", "senior", "expert")
# "student" = clé interne ms ; libellé affiché = Novice (aligné Option A / papier)
CLASS4_LABELS = {
    "student": "Novice",
    "junior": "Junior",
    "senior": "Senior",
    "expert": "Expert",
}
# Ordinal ranks used for monotonicity (1 = novice pole → 4 = expert pole)
CLASS4_RANK = {"student": 1, "junior": 2, "senior": 3, "expert": 4}

# Continuous targets used to map regression scores → discrete classes
CLASS4_TARGETS = (-1.00, -0.33, 0.33, 1.00)

SUBLEVEL_TO_CLASS4 = {
    "ms": "student",
    "pgy1": "junior", "pgy2": "junior", "pgy3": "junior",
    "pgy4": "junior", "pgy5": "junior",
    "pgy6": "senior", "fellow": "senior",
    "staff": "expert",
}

# Early / Middle / Late partitions of normalised gesture time
PHASE_BOUNDS = ((0.0, 0.33), (0.33, 0.66), (0.66, 1.0))
PHASE_LABELS = ("Early [0–33%]", "Middle [33–66%]", "Late [66–100%]")

# Muted Nature / IEEE palette — identical across all three figures
PALETTE = {
    "student": "#C75B5B",
    "junior": "#C9A227",
    "senior": "#6B7C93",
    "expert": "#7A8F6A",
}

FIGURE_WIDTH = 7.2
FIGURE_DPI = 300
EXPORT_FORMATS = ("pdf", "svg", "png")

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "medium",
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
    "savefig.dpi": FIGURE_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
}
