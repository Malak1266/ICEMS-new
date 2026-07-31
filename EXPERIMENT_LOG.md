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

---

## C5 — Run A : `use_hoel=False` / MSE plate (pré-enregistré 2026-07-22, AVANT sortie)

**Hypothèse unique testée.** HOEL (tier Staff ×4, asym α Expert=5 vs Novice=1) est l’écart D3 non documenté dans le papier ; le retirer (MSE plate) doit descendre les Novices vers −0.80 sans casser l’ordre ni les Experts.

**Job actif.** `66213237` — `USE_HOEL=0`, array 0–4, `--time=00:45:00`, `--mem=16G`, `--gres=gpu:1`.  
*(Job `66212330` annulé : ancien sbatch 3h/64G/A100, file ~2 jours.)*  
Confirmé au submit : `USE_HOEL=0`. Confirmé au code : `run_experiment.py --no-hoel` → `ExpTrainConfig.use_hoel=False` → `nn.MSELoss()` ; export `REPO_ROOT/runs/seed_{SEED}.csv` (+ copie sous `experiments/.../runs/`).

**Baseline HOEL figée** (`panelA_raw`, `--calib raw`, 5 seeds) :

| Groupe | n | moyenne |
|--------|---|---------|
| Novice | 14 | −0.529 |
| Junior | 14 | −0.055 |
| Senior | 11 | +0.153 |
| Expert | 8 | +0.756 |

**Critères de succès (écrits AVANT lecture des résultats Run A) :**

1. Novice ∈ **[−0.92 ; −0.68]**
2. Ordre monotone : Novice < Junior < Senior < Expert
3. Expert **> +0.60**
4. Contraste Senior−Junior non significatif (comme publié / baseline)

**Scénarios (lecture autorisée uniquement) :**

| # | Observation | Décision |
|---|-------------|----------|
| 1 | Novice dans [−0.92 ; −0.68] | D3 = HOEL ; reproduction fidèle bouclée |
| 2 | Novice descend mais ~−0.6 | HOEL explique une partie ; résidu déclaré (n=14, 1 modèle vs 10) — **pas de 3ᵉ correction** |
| 3 | Expert < +0.60 ou ordre cassé | MSE a sur-corrigé ; HOEL redevient choix défendable, pas faute |

**Agrégation à la sortie :**
```bash
python aggregate_seeds.py --glob "runs/seed_*.csv" --out runA_nohoel --calib raw
# → coller runA_nohoel_rapport.txt ici
```

**Garde-fou CSV :** `grep -L "0.0,0.0,0.0" runs/seed_*.csv` + `head -3` sur chaque seed (refuser placeholders).

**Hors scope de ce run.** Butterworth, init, époques, validité prédictive (B), taux détection (C).

---

## C6 — PRE-REG A2 vs A0 (Attention Temporal Pooling) — figé AVANT scoring du milieu

**Date de verrouillage :** 2026-07-24 (avant tout scoring Junior/Senior des runs attn).  
**Objectif publi :** SPIE MI27 — extension méthodologique, pas gain annoncé sans mesure.

### Agrégation essai→participant (décision de cohérence)

| Source | Agrégation | Notes |
|--------|------------|-------|
| Papier / eApp 2 / `eval_hybrid1_perpair.py` / `aggregate_seeds.py` | **moyenne** | comparabilité R² papier |
| `eval_hybrid1.py` (Model A faithful, artefacts locaux) | **médiane** | robustesse tracking |
| Chiffres « corrigés » cités (−0.881…+0.857) | **non retrouvés** dans `results/hybrid1_faithful` | agrégat local médiane ≈ Nov −0.781 / Jun −0.295 / Sen −0.064 / Exp +0.465 |

**Décision SPIE :** analyse **principale = moyenne** (comparabilité papier).  
**Sensibilité (annexe) = médiane** — les deux rapportées ; A0 et A2 toujours sous la *même* agrégation.  
Pas de mélange silencieux avec d’anciens chiffres mémoire sans re-scoring.

### PRE-REG A2 vs A0 (décidé avant tout scoring du milieu)

```
Primaire   : pente OLS + ρ_middle (Junior+Senior), scores bruts, participant-level
Direction  : A2 > A0 (unilatéral, justifié a priori : l'attention concentre le signal)
Headline   : consistance seed-à-seed (n_favorables/n_seeds + Δ médian [min,max])
             — PAS un p unique ; avec n=5, p bilatéral Wilcoxon ne peut pas < 0.0625
Inférences  : (1) seed-variance = sign-test / Wilcoxon unilatéral
             (2) sampling-variance = IC bootstrap participant (BCa pour R²)
             — deux phrases distinctes, jamais confondues
Confirmé   : majorité seeds favorables (cible 10/10 ou ≥9/10 ; minimum historique 5/5)
             ET AUC/acc extrêmes non dégradées
             ET A6 (time-shuffle) s'effondre vers A0
Partiel    : pente↑ mais Junior–Senior toujours non séparés
             → RÉSULTAT (frontière structurelle), pas échec
Nul        : A2≈A0 → rapporté tel quel ; A6 reste une contribution
Sélection  : early stop, heads, tau, SWA → val EXTRÊMES uniquement
Milieu     : scoré UNE fois, à la fin. Pas de re-run pour garder les bons seeds.
Seeds      : principal {42,123,456,789,2024} ; extension SPIE → 10+ seeds
             sans cherry-pick
SWA/heads  : seulement SI mécanisme A2+A6 confirmé ; conditions étiquetées séparément
Confond    : contrôle α↔activité instrumentale obligatoire avant claim « expertise »
A5         : contrôle mineur (GAP déjà masked_mean) — poids narratif sur A2+A6
```

**Interdiction :** ne pas relire ce bloc après avoir vu les scores milieu pour changer les seuils.

---

## C7 — Canonisation A0 (2026-07-24) — BLOQUANT avant A2

### Traçage des chiffres mémoire (−0.881 / −0.303 / −0.028 / +0.857)

| Hypothèse | Statut |
|-----------|--------|
| `results/hybrid1_faithful` (Model A multivarié, 5 seeds) | **Non** — agrégat médiane ≈ −0.781 / −0.295 / −0.064 / **+0.465** (Expert Δ≈0.39) |
| Re-agrégation moyenne vs médiane sur ces runs | **Insuffisant** pour expliquer Δ Expert 0.39 (3 essais/participant) |
| `results/hybrid1_perpair*` | **Absent** du dépôt local (aucun artefact synchronisé) |
| Cibles papier EV-ICEMS (`PAPER_TARGET`) | −0.80 / −0.09 / +0.25 / +0.75 — **≠** −0.881…+0.857 |
| Grep repo + transcripts | Les −0.881/+0.857 n’apparaissent que comme citation externe (message co-auteur) — **aucune lignée fichier** |

**Verdict traçage :** chiffres **orphelins** dans ce workspace. Origine la plus plausible = run **Narval per-pair (Version B)** non rapatrié, ou agrégation two-stage d’un run hors dépôt. **Non régénérables ici.**

### Dette supplémentaire : checkpoints vs code actuel

Les `model_A_best.pt` de `hybrid1_faithful` utilisent le préfixe `backbone.*` (ancienne structure).  
Le `Hybrid1EVICEMS` actuel attend des clés plates (`lstm.*`, `pos_embedding`, …).  
→ `load_state_dict` **échoue** sans remap. Les `panelE_report.json` / `predictions_*.pkl` restent valides comme *sortie figée de l’ancien code*, mais **ne constituent pas** une baseline rejouable bit-pour-bit avec le code SPIE actuel.

### Décision (A0 admissible)

1. **Abandonner** −0.881…+0.857 comme baseline A0 pour toute comparaison A2 (sauf si le run Narval per-pair est rapatrié *et* rejoué).
2. **A0 canonique SPIE** = Model A multivarié, `pool_type=gap`, `trial_tensor_v2.pkl`, `--agg mean`, architecture **courante**, seeds pré-enregistrées, re-entraîné + re-scoré sous le code gelé.
3. Mémoire / soutenance : attributer explicitement les anciens chiffres au run non synchronisé, ou les remplacer par l’A0 canonique une fois produit.

### Seeds pré-enregistrées (extension à 10 — avant scoring milieu)

```
CORE (historique)     : 42, 123, 456, 789, 2024
EXTENSION (annoncée)  : 1337, 2718, 3141, 9001, 5555
ENSEMBLE CANONIQUE A0/A2/A6 : les 10 ci-dessus
```

Wilcoxon **bilatéral** atteignable sous 10 seeds. Headline reste : k/10 favorables + Δ/seed, pas un p isolé.  
Pas de cherry-pick : liste figée **avant** tout scoring Junior/Senior des runs attn.

---

## C8 — A0 CANONIQUE SPIE (définition fermée)

**Tag git :** `a0-canonical-spie`  
**git SHA :** `48d4d8ec4d08e571a04f1e28ccf93d3aa081015e`  
**Date de gel :** 2026-07-24 (pre-A2 baseline)

| Champ | Valeur figée |
|-------|----------------|
| Architecture | Hybrid1 Model A (Version A, multivarié) |
| `d_in` / `n_features` | **10** |
| Pooling | `pool_type=gap` (masked mean) |
| Agrégation essai→participant | **mean** (principal) ; median = annexe `--agg-both` |
| Données | `data/trial_tensor_v2.pkl` |
| Protocole | extrêmes train/val/test ; milieu scoré une fois en fin |
| Seeds | `42 123 456 789 2024 1337 2718 3141 9001 5555` |
| Déterminisme | `set_full_determinism` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| Scripts | `test_repro_seed42.sh` (gate) → `run_a0_canonical.sh` (array 0–9) |
| Sortie gelée | `results/a0_canonical/seed{SEED}/` — **ne pas retoucher** |

**Gate obligatoire :** `bash test_repro_seed42.sh` → slope / r2 / rho_middle tous `|Δ|<1e-3` entre run_A et run_B (seed 42).  
Si FAIL → ne pas lancer les 10 seeds. Si `use_deterministic_algorithms(True)` plante sur A100 → repli best-effort (seeds + cudnn.deterministic, sans mode strict).

**Ordre :** tag → repro OK → `sbatch run_a0_canonical.sh` → geler dossier → seulement ensuite A2 (`attn`) / A6 (`attn` + `--pool-time-shuffle`).
