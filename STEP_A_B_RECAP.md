# ICEMS — Récapitulatif Step A & Step B

Document de synthèse : ce qui a été implémenté, la structure des données labellisées, les résultats obtenus et les prochaines étapes.

---

## 1. Contexte

**ICEMS** évalue automatiquement l'expertise chirurgicale à partir de séries temporelles de mouvement (tracking d'instruments en chirurgie simulée).

**Objectif des deux scripts :**

1. **Step A** — Générer des séquences synthétiques (augmentation) via un GRU conditionnel.
2. **Step B** — Prédire un score d'expertise continu avec évaluation **LOPO** (Leave-One-Participant-Out).

**Fichiers créés :**

| Fichier | Description |
|---------|-------------|
| `src/step_A_data_generation.py` | Génération GRU conditionnelle |
| `src/step_B_classification.py` | Scorer causal + LOPO + graphiques |
| `data/augmented_trials.pkl` | Dataset enrichi (136 → 456 trials) |
| `results/level1/*.png` | Visualisations Step B |

**Dépendance ajoutée :** `torch` dans `requirements.txt`.

---

## 2. Données labellisées originales

### Fichiers sources

| Fichier | Contenu |
|---------|---------|
| `data/filtered_data.json` | Données brutes (136 trials, métriques par instrument) |
| `data/continuous_per_trial.pkl` | Dataset ML prêt à l'emploi (**136 trials**) |
| `data/Exvivo_trial_Participants(Sheet1).csv` | Métadonnées des 47 participants |

### Format d'un trial (`continuous_per_trial.pkl`)

```python
("ID_participant", "TrialN") → {
    "X":      np.ndarray (T, 10),   # séquence variable en longueur
    "y9":     int,                  # classe fine 0..8
    "y_reg":  float,                # score régression [-1, +1]
    "level":  str,                  # libellé texte
    "T":      int,                  # nombre de frames
    "fs":     float,                # ~10 Hz
}
```

### Les 10 canaux de `X`

| Index | Canal |
|-------|--------|
| 0–2 | Bipolar : velocity, acceleration, jerk |
| 3–5 | Scissors : velocity, acceleration, jerk |
| 6–8 | Cavitron : velocity, acceleration, jerk |
| 9 | `valid_ratio` — proxy de validité du tracking (1 = frame OK, 0 = occlusion) |

Un trial dure typiquement **3 000 à 14 000 frames** (~5 à 20 min à 10 Hz).

### Les 9 classes fines (`y9`) — labels officiels du pkl

| y9 | Classe | y_reg | Trials |
|----|--------|-------|--------|
| 0 | Medical student | -1.00 | 42 |
| 1 | Resident PGY1 | -0.75 | 14 |
| 2 | Resident PGY2 | -0.50 | 9 |
| 3 | Resident PGY3 | -0.25 | 6 |
| 4 | Resident PGY4 | +0.00 | 3 |
| 5 | Resident PGY5 | +0.25 | 9 |
| 6 | Resident PGY6 | +0.50 | 11 |
| 7 | Fellow | +0.86 | 20 |
| 8 | Staff | +1.00 | 22 |

**47 participants** au total.

### Labels bruts avant normalisation (`filtered_data.json`)

La colonne `level` contient **14 libellés** distincts. Les sous-types Fellow sont fusionnés en `Fellow` dans le pkl :

- Fellow Pediatrics (6), Fellow Epilepsy (3), Fellow Oncology (3), Fellow Spine (3), Fellow/Spine (3), Fellow functional (2) → **20 trials Fellow**

### Colonne `Expertise` (4 classes à la collecte)

Présente dans le JSON et le CSV — regroupement **déjà existant** à la collecte :

| Expertise | Trials | Équivalent |
|-----------|--------|------------|
| Student | 42 | Medical student |
| Junior | 41 | PGY1 – PGY5 |
| Senior | 31 | PGY6 + Fellow |
| Expert | 22 | Staff |

Step A et Step B utilisent ce regroupement en 4 classes (`y4`) avec scores cibles **-1.00 / -0.33 / +0.33 / +1.00**.

```python
Y9_TO_Y4 = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 3}
```

---

## 3. Step A — Génération de données (`step_A_data_generation.py`)

### Objectif

Apprendre la **dynamique temporelle** des gestes par classe, puis générer **80 séquences synthétiques par classe** (320 au total).

### Architecture : `ConditionalGRUGenerator`

```
Classe y4 → Embedding(4, 8)
Frame X_t (10) + embedding → GRU(2 layers, hidden=64, dropout=0.1)
                          → Linear(64→32) → GELU → Linear(32→10)
                          → prédiction frame t+1
```

- **Entraînement :** teacher forcing, loss MSE frame suivante, Adam lr=1e-3, 50 epochs.
- **Génération :** graine = 10 premières frames d'un vrai trial ; puis autoregressif + bruit σ=0.05.

### Sortie : `data/augmented_trials.pkl`

Même structure que l'original + champs `y4`, `y4_reg`, `synthetic=True` pour les trials générés.

Clés synthétiques : `(synth_y4{C}_{NNN}, TrialM)`.

### Adaptations performance

| Constante | Valeur | Raison |
|-----------|--------|--------|
| `TRAIN_CROP_LEN` | 512 | Trials trop longs pour GRU sur séquence entière (CPU) |
| `GEN_MAX_LEN` | 2000 | Plafond génération (moyennes réelles 3.4k–5.6k frames) |

### Résultats Step A

| Classe | Avant | Après | Ajout |
|--------|-------|-------|-------|
| Student | 42 | 122 | +80 |
| Junior | 41 | 121 | +80 |
| Senior | 31 | 111 | +80 |
| Expert | 22 | 102 | +80 |
| **Total** | **136** | **456** | **+320** |

**Validation vitesse moyenne** (canal 0 `bipolar.velocity`) :

| Classe | Écart réel vs synth | Statut |
|--------|---------------------|--------|
| Student | 2.1% | OK |
| Junior | 17.4% | OK |
| Senior | 24.0% | ⚠️ (> 20%) |
| Expert | 48.5% | ⚠️ (> 20%) |

> **Limite connue :** Step A normalise pas les features (contrairement à Step B). Le générateur converge vers une dynamique « moyenne » — les synthétiques Expert ne reproduisent pas bien les vitesses réelles.

### Usage

```bash
python src/step_A_data_generation.py
python src/step_A_data_generation.py --input data/continuous_per_trial.pkl --out data/augmented_trials.pkl
```

---

## 4. Step B — Classification LOPO (`step_B_classification.py`)

### Objectif

Prédire un **score continu** d'expertise par trial, avec évaluation sur **47 participants** (LOPO) : à chaque fold, un chirurgien entier est exclu du train.

**Pourquoi LOPO (et pas train extrêmes / val milieu) :**

- Le modèle voit **tous les niveaux** pendant l'entraînement.
- On teste la généralisation à un **nouveau participant inconnu**.
- Plus réaliste pour un déploiement clinique.

### Architecture : `CausalGRUScorer`

```
X (10) + masque validité (1) → Linear(11→64) → GELU
                              → GRU(2 layers, 64, dropout=0.3)
                              → Linear → GELU → Dropout → Linear(1) → Tanh
                              → score_t par frame ∈ [-1, +1]
```

### Loss

```
loss = MSE(score_agrégé, y4_reg) + 0.1 × pénalité_ordre
```

- **Agrégation :** moyenne pondérée par le masque de validité (entraînement) ; **médiane** des frames valides (inférence trial).
- **Pénalité ordinale :** punition si un Junior reçoit un score plus haut qu'un Senior dans le même batch.

### MC Dropout (incertitude)

À l'inférence : `model.train()` (dropout actif), 30 passes forward → moyenne ± écart-type par frame.

### Protocole LOPO

Pour chaque participant (47 folds) :

1. Retirer **tous** ses trials du train.
2. Entraîner sur les 46 autres + **trials synthétiques** (toujours en train).
3. Prédire sur le participant tenu out.

Optimiseur : AdamW lr=1e-3, weight_decay=1e-4, early stopping patience=8.

### Métriques affichées

1. Pearson r (et p-value)
2. Spearman r (et p-value)
3. Accuracy 4 classes (score → classe la plus proche parmi {-1, -0.33, 0.33, 1})

### Graphiques (`results/level1/`)

| Fichier | Contenu |
|---------|---------|
| `score_vs_time.png` | Score vs temps normalisé [0,1], couleur par classe, bande ±σ MC Dropout |
| `confusion_matrix.png` | Matrice 4×4, accuracy dans le titre |
| `scatter.png` | Score prédit vs réel, droite y=x, Pearson dans le titre |

### Résultats Step B (test rapide exécuté)

```bash
python src/step_B_classification.py --data data/augmented_trials.pkl --epochs 5 --max_folds 3 --mc-passes 5
```

| Métrique | Valeur | Note |
|----------|--------|------|
| Pearson r | +0.767 | 9 trials, 3 participants |
| Spearman r | +0.633 | idem |
| Accuracy 4 classes | 33.3% (3/9) | 5 epochs seulement |

> **Run production non exécuté :** `--epochs 40` sur 47 folds (plusieurs heures CPU).

### Usage

```bash
python src/step_B_classification.py --data data/augmented_trials.pkl --out results/level1 --epochs 40
python src/step_B_classification.py --data data/augmented_trials.pkl --epochs 5 --max_folds 3
```

---

## 5. Pipeline global

```
filtered_data.json
       ↓  build_continuous_dataset.py (existant)
continuous_per_trial.pkl  (136 trials, 9 classes y9)
       ↓  step_A_data_generation.py
augmented_trials.pkl      (456 trials = 136 réels + 320 synth.)
       ↓  step_B_classification.py
results/level1/           (métriques LOPO + 3 graphiques)
```

---

## 6. Différences avec l'ancien code du repo

| Aspect | Ancien (`kfold_cv`, `train_long`) | Nouveau (Step A / B) |
|--------|-----------------------------------|----------------------|
| Framework | TensorFlow / Keras | PyTorch |
| Unité | Fenêtres 300 frames | Séquence trial (crop) |
| Score | 1 score / fenêtre | 1 score / **frame** |
| Split | K-fold, souvent classes 0+8 | LOPO 47 participants, 4 classes |
| Augmentation | Jitter, time-warp | GRU génératif conditionnel |

---

## 7. Prochaines étapes recommandées

### Priorité 1 — Améliorer Step A

- [ ] Normaliser les 9 canaux cinématiques avant entraînement du GRU (comme Step B).
- [ ] Re-valider vitesse moyenne (< 20 % sur les 4 classes).
- [ ] Option : entraîner Step B **sans** synthétiques tant que la validation échoue.

### Priorité 2 — Run Step B production

```bash
python src/step_B_classification.py \
  --data data/augmented_trials.pkl \
  --out results/level1_full \
  --epochs 40
```

### Priorité 3 — Ablations

- [ ] Step B sur `continuous_per_trial.pkl` seul (sans synth).
- [ ] Step B sans pénalité ordinale.
- [ ] Comparaison avec baseline `kfold_cv.py`.

### Priorité 4 — Code

- [ ] Exporter prédictions LOPO en CSV/JSON par fold.
- [ ] Sauvegarder les poids du modèle par fold.
- [ ] Limiter les courbes `score_vs_time` à N trials/classe pour lisibilité.

---

## 8. Commandes rapides

```bash
# Depuis le dossier ICEMS-main

# Reconstruire le pkl original (si besoin)
python src/build_continuous_dataset.py

# Génération synthétique
python src/step_A_data_generation.py

# Classification LOPO (production)
python src/step_B_classification.py --data data/augmented_trials.pkl --out results/level1 --epochs 40

# Test rapide
python src/step_B_classification.py --data data/augmented_trials.pkl --epochs 5 --max_folds 3 --mc-passes 5
```

---

*Document généré le 2026-06-04 — synthèse du travail Step A / Step B et de la structure des données labellisées.*
