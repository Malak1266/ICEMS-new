# Provenance des figures temporelles

> Document généré le 2026-07-20 — **aucune modification de code** (Tâche 1).
> Commit git : les scripts cités sont **non versionnés** dans l’historique git actuel
> (`git log` ne retourne aucune entrée pour `scripts/plot_hybrid1_temporal.py` ni
> `evaluation_publication/`). HEAD du dépôt au moment de l’audit : `e30f1a2`.

---

## 1. `temporal_expert_v2_smooth.png`

> **Incohérence de nommage** : le script produit `temporal_expertise_v2_smooth.png`
> (avec **-ise**), pas `temporal_expert_v2_smooth.png`. Voir § Incohérences.

| Champ | Valeur |
|-------|--------|
| **Script générateur** | `scripts/plot_hybrid1_temporal.py` |
| **Commit git** | *Non versionné* — fichier untracked, pas de SHA disponible |
| **Point d’entrée** | `python scripts/plot_hybrid1_temporal.py --run … --pkl data/trial_tensor_v2.pkl --out … --tag v2_smooth --smooth-sigma 5` |
| **Tenseur consommé** | `data/trial_tensor_v2.pkl` (docstring l. 15 ; `--pkl` requis hors `--reuse-cache`) |
| **Modèle consommé** | 10 checkpoints per-pair Hybrid1 (`results/hybrid1_perpair_v2_patched_ab/pair{k}_*/model_A_best.pt`) — **non** le run `hybrid_extremes` |
| **Niveau d’agrégation** | **Participant** (médiane des essais par participant, puis médiane + IQR par groupe `group4`) |
| **Temps normalisé** | **Par essai** : `resample_to_grid()` interpole les scores frame-level sur `linspace(0, 1, 100)` relativement à la longueur de *chaque* essai ; puis agrégation inter-essais au niveau participant |
| **Post-traitement score** | Calibration affine `a·score + b` (novice/expert) **avant** agrégation ; lissage Gaussien (`--smooth-sigma`, défaut 5) **après** agrégation groupe |
| **Label « Novice »** | Oui — clé interne `novice`, affiché « Novice » (`GROUP_LABELS`, l. 56) |

---

## 2. `temporal_expertise_facets_v2_smooth.png`

| Champ | Valeur |
|-------|--------|
| **Script générateur** | `scripts/plot_hybrid1_temporal.py` — fonction `plot_separate()` |
| **Commit git** | *Non versionné* (même fichier que §1) |
| **Point d’entrée** | Identique à §1 ; fichier écrit par `fig.savefig(out / f"temporal_expertise_facets_{tag}.{ext}")` (l. 226) |
| **Tenseur consommé** | `data/trial_tensor_v2.pkl` |
| **Niveau d’agrégation** | **Participant** (identique §1) ; panneaux 2×2 par `group4` |
| **Temps normalisé** | **Par essai** (identique §1) |
| **Post-traitement** | Calibration affine + lissage Gaussien (identique §1) |

Produit en même exécution que §1 ; sorties supplémentaires : `temporal_{g}_v2_smooth.png` (un panneau par groupe).

---

## 3. `fig1_temporal_stability.png`

| Champ | Valeur |
|-------|--------|
| **Script générateur** | `evaluation_publication/plots.py` → `plot_temporal_stability()` |
| **Orchestrateur** | `evaluation_publication/run_publication.py` → `generate_all_figures()` ; sbatch : `slurm/run_publication_figures.sbatch` |
| **Commit git** | *Non versionné* — répertoire `evaluation_publication/` untracked |
| **Tenseur consommé** | **`data/continuous_per_trial.pkl`** — **pas** `trial_tensor_v1.pkl` ni `trial_tensor_v2.pkl` (`evaluation_publication/config.py` l. 8, `DEFAULT_PKL`) |
| **Modèle consommé** | Checkpoint unique `results/hybrid_extremes/model_best.pt` (protocole train-extrêmes) |
| **Scores frame-level** | Extraits via `src/eval/extract_frame_scores.py` → `frame_predictions.pkl` |
| **Niveau d’agrégation** | **Essai** : pour chaque essai, moyenne des scores frame dans chaque phase (Early / Middle / Late) ; boxplots par classe sur ces moyennes par essai |
| **Temps normalisé** | **Par essai** : `time_norm = linspace(0, 1, n_used)` dans `extract_frame_scores_for_trial()` (`src/eval/extract_frame_scores.py` l. 304) ; `n_used = min(T_raw, seq_len)` — essais tronqués au premier segment `seq_len` (défaut 4000 frames, crop `start`) |
| **Post-traitement score** | **Aucune calibration affine** — scores frame bruts (Tanh, ∈ [-1, 1]) |
| **Label « Student » = « Novice » ?** | **Oui, sémantiquement** : `ms` → clé `"student"` → libellé `"Student"` (`evaluation_publication/config.py` l. 12–17, `SUBLEVEL_TO_CLASS4` l. 25–30). Correspond au pôle novice (`CLASS4_RANK["student"] = 1`, cible −1.00). Le pipeline Hybrid1 (`data_hybrid1.py`) nomme ce même niveau **`novice`** — divergence terminologique inter-pipelines |

---

## 4. Incohérences recensées

| # | Nature | Détail |
|---|--------|--------|
| I1 | **Nom de fichier** | Demande `temporal_expert_v2_smooth.png` ; le script écrit `temporal_expertise_v2_smooth.png` |
| I2 | **Source de données** | Figures §1–2 : `trial_tensor_v2.pkl` ; figure §3 : `continuous_per_trial.pkl` — pipelines distincts |
| I3 | **Architecture / modèle** | §1–2 : 10 modèles per-pair Hybrid1 (composite moyenné) ; §3 : un Hybrid LSTM-Transformer global (extrêmes) |
| I4 | **Agrégation** | §1–2 : participant (médiane essais → médiane groupe) ; §3 : essai (moyenne frame par phase) |
| I5 | **Calibration** | §1–2 : calibration affine post-hoc (`calibrate_affine`, plot calibré) ; §3 : scores bruts Tanh |
| I6 | **Vocabulaire classe basse** | Hybrid1 / plot temporel : **Novice** ; publication fig1 : **Student** — même sous-niveau `ms` |
| I7 | **Troncature temporelle** | §3 tronque les essais > `seq_len` (4000 frames) ; §1–2 score l’intégralité via chunks concaténés |
| I8 | **Traçabilité git** | Aucun des scripts générateurs n’est commité ; impossibilité d’attacher un SHA reproductible |
| I9 | **Audit vs état actuel** | `mae_protocol_audit.md` (§4.2) indiquait `trial_tensor_v2.pkl` **absent** ; le fichier et `build_trial_tensor_v2.py` existent désormais dans l’arbre de travail |
| I10 | **Défaut data_hybrid1** | Corrigé (Tâche 2) : `resolve_hybrid1_paths()` retombe désormais sur `v2` par défaut |

---

## 5. Chaîne de dépendances (résumé)

```
§1–2  trial_tensor_v2.pkl
        → plot_hybrid1_temporal.py
        → 10× Hybrid1ModelA (per-pair)
        → calibration affine + smooth
        → temporal_expertise_v2_smooth.png
        → temporal_expertise_facets_v2_smooth.png

§3    continuous_per_trial.pkl
        → train_hybrid_extremes / model_best.pt
        → extract_frame_scores.py → frame_predictions.pkl
        → evaluation_publication/run_publication.py
        → plots.plot_temporal_stability()
        → fig1_temporal_stability.png
```
