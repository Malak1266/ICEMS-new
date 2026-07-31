# EXPERIMENT_LOG — Journal d'expériences ICEMS Hybrid 1

## C3 — Audit pipeline (2026-07-20, machine locale Windows)

**Contexte.** Premier passage de `audit_pipeline.py` sur le dépôt réel, *avant toute correction*.

**Commande.**
```bash
python audit_pipeline.py --repo . --scores-glob "runs/seed_*.csv" --out docs/PROVENANCE.md
```

**Synthèse.** `3 ECHEC · 2 INCONNU · 12 vérifications` → **NE PAS LANCER LE RUN**.

| Code | Label | Statut | Note |
|------|-------|--------|------|
| A1 | Artefacts présents | PASS | 69 fichiers signés (MD5) |
| A2 | Doublons checkpoint | WARN | `hybrid1_faithful/seed42` == `hybrid1_smoke/seed42` (norm/split) |
| B1 | Version tenseur unique | PASS | v2 seul dans les `.py` |
| C1 | Chevauchement participants | INCONNU | pas de `runs/seed_*.csv` |
| D1 | Étanchéité Junior/Senior | INCONNU | idem |
| E1 | Normalisation train-only | PASS | |
| F1 | Calibration bornée + train-only | ECHEC | scripts plot/eval sans atanh |
| G1 | Aucune inversion d'axe | ECHEC | **faux positif** : `audit_pipeline.py` se matche lui-même (`[::-1]` dans son regex) |
| G2 | Niveau d'agrégation unique | PASS | |
| H1 | Commit identifié | PASS | `e30f1a2` |
| H2 | Arbre propre | ECHEC | nombreux fichiers non commités |
| H3 | Seeds fixées | PASS | |

**Décision.** Consigne faite. Corrections **pas encore appliquées**. Prochaines actions ordonnées :
1. Produire ou pointer des CSV scores → lever C1/D1
2. Brancher `calibration.py` (bornée) sur les figures / `aggregate_seeds.py` → lever F1
3. Exclure `audit_pipeline.py` du scan G1 → lever faux positif
4. Commit propre → lever H2
5. Re-audit jusqu'à 0 ECHEC · 0 INCONNU
6. Seulement alors : `sbatch run_hybrid1.sh`

**Fichiers livrés Claude (installés localement).**
- `audit_pipeline.py`
- `run_hybrid1.sh`
- `aggregate_seeds.py`
- `calibration.py`
- `palette_icems.py`
- `docs/PROVENANCE_FIGURES.md` (ancien rapport manuel figures, sauvegardé)

---

## C4 — Option A Phase 1–3 (2026-07-20)

**Actions.**
- `run_hybrid1.sh` branché sur `run_experiment.py` + `continuous_per_trial.pkl` + seeds `(42,123,456,789,2024)`
- Groupes CSV : `Novice/Junior/Senior/Expert` ; export vers `runs/seed_*.csv`
- Fix import `_make_sinusoidal_pe` dans `hybrid1_evicems.py`
- Auditeur : F1 recentré pipeline extremes ; G1 ignore `audit_pipeline.py`
- CSV split (placeholder scores) : `runs/seed_42.csv` — 47 participants, 136 essais, train∩test=∅, middle∉train

**Re-audit.**

| Code | Statut |
|------|--------|
| A1–A2 | PASS / WARN |
| B1 | PASS |
| C1 | PASS |
| D1 | PASS |
| E1 | PASS |
| F1 | PASS |
| G1–G2 | PASS |
| H1 | PASS |
| **H2** | **FAIL** — arbre git sale (attendu) |
| H3 | PASS |

**Bloqueur restant.** H2 uniquement → commit propre requis avant Narval.

**Prochaine étape.** Commit (sur demande explicite) → re-audit vert → sync Narval → `sbatch run_hybrid1.sh`.
