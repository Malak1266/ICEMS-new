"""Vérifie l'hypothèse du confond de durée entre Students et Staffs."""
import pickle
import numpy as np
from scipy import stats

with open("data/continuous_per_trial.pkl", "rb") as f:
    dataset = pickle.load(f)

durations_by_class = {0: [], 8: []}
for rec in dataset.values():
    if rec["y9"] in (0, 8):
        durations_by_class[rec["y9"]].append(rec["T"])

for c, name in [(0, "Student"), (8, "Staff")]:
    d = np.array(durations_by_class[c])
    print(f"{name:8s} (n={len(d):2d}) : mean={d.mean():7.1f}  median={np.median(d):7.1f}  std={d.std():7.1f}")

t, p = stats.mannwhitneyu(durations_by_class[0], durations_by_class[8])
print(f"\nMann-Whitney U test : statistic={t:.2f}, p-value={p:.4f}")
print(f"  p < 0.05  →  différence significative  →  ton hypothèse est VALIDÉE")
print(f"  p >= 0.05 →  pas de différence significative  →  hypothèse non validée")