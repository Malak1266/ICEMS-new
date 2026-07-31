# PROJECT_PROGRESSION.md

> **Document de référence absolue** pour le nettoyage de la codebase ICEMS.
> Rédigé après la réunion de revue avec le professeur — toute évolution du code
> doit désormais s'aligner sur les directives consignées ici.
>
> **Auteur :** Malak1266
> **Date de rédaction :** 2026-05-22
> **Statut :** Référence active — à respecter pour toutes les Pull Requests à venir.

---

## Table des matières

1. [PARTIE 1 — Statut de l'avancement *avant* la réunion](#partie-1--statut-de-lavancement-avant-la-réunion)
2. [PARTIE 2 — Compte-rendu et remarques du professeur](#partie-2--compte-rendu-et-remarques-du-professeur-pendant-la-réunion)
3. [PARTIE 3 — Feuille de route opérationnelle (post-réunion)](#partie-3--feuille-de-route-opérationnelle-post-réunion)
4. [PARTIE 4 — Architecture du scoring temporel continu (Tâches G + H)](#partie-4--architecture-du-scoring-temporel-continu-tâches-g--h)

---

## PARTIE 1 — Statut de l'avancement (AVANT la réunion)

### 1.1 Architecture de base

| Composant | Valeur |
|---|---|
| Modèle | **Masked Autoencoder (MAE)** pré-entraîné |
| Taille de fenêtre | **32 frames** (fenêtres fixes) |
| Fréquence d'échantillonnage | **10 Hz** |
| Durée d'une fenêtre | **3.2 s** |
| Têtes d'attention | **4 têtes** |
| Dimension par tête | **16** |
| Dimension d'embedding totale | $4 \times 16 = 64$ |

### 1.2 Dataset initial — Matrice à 6 features

| # | Feature | Description physique | Statut post-réunion |
|---|---|---|---|
| 1 | `spread` | Entraxe / écartement entre sphères | **À SUPPRIMER** (constante physique) |
| 2 | `axis_angle` | Inclinaison du corps rigide / repère caméra | **À SUPPRIMER** (non invariante au repère + invalide si N_SPHERES=1) |
| 3 | `velocity` | Vélocité 3D (dérivée 1ʳᵉ de la position) | **À CONSERVER** |
| 4 | `accel` | Accélération (dérivée 2ᵉ) | **À CONSERVER** |
| 5 | `jerk` | Jerk (dérivée 3ᵉ) | **À CONSERVER** |
| 6 | `valid_ratio` | Ratio de validité des frames dans la fenêtre | **À CONSERVER** |

### 1.3 Hypothèse de capture

- **`N_SPHERES = 1`** retenu pour maximiser la détection.
  - **Effet positif observé :** `pct_valid` passe de **6 % → 45 %** par rapport à des configurations 3/4/5 sphères.
  - **Effet négatif (non identifié à l'époque) :** perte irréversible de la pose 6DOF (cf. §2.2).
- **Seuil minimal de validité :** `MIN_VALID = 10 %` (toute fenêtre en dessous est rejetée).

### 1.4 Protocole de validation

- **Méthode :** Leave-One-Out Cross-Validation (**LOOCV**).
- **Cohorte :** **47 participants**, chacun réalisant **3 trials** (essais).
- **Métrique cible :** coefficient de corrélation de **Pearson**.
- **Benchmark historique visé :** $r = 0{,}555$ — *cf. §2.1 : ce benchmark est invalidé.*

### 1.5 Problème identifié avant la réunion : le **Biais Central**

- **Symptôme :** le LSTM de régression prédit des scores **resserrés entre 1.2 et 2.7**.
- **Conséquences :**
  - Aucun *Student* pur n'est prédit à 0.
  - Aucun *Expert* n'est prédit à 3.
  - Les extrêmes sont systématiquement écrasés vers la moyenne.
- **Cause architecturale suspectée :** la prédiction par trial était calculée comme la **moyenne arithmétique** des prédictions de toutes les fenêtres de 32 frames.

---

## PARTIE 2 — Compte-rendu et remarques du professeur (PENDANT la réunion)

### 2.1 L'imposture du Benchmark Historique

- **Constat :** la valeur $r = 0{,}555$ attribuée à l'ancienne étudiante **est FAUSSE**.
- **Vérification effectuée :** cette valeur **n'apparaît nulle part** dans le code scientifique ou dans le papier du projet.
- **Origine identifiée :** artefact issu d'un exemple de la documentation de la bibliothèque **Pandas** (chemin `pandas/site-packages`).
- **Verdict :** le professeur **avait raison de douter**. Le benchmark imaginaire est **définitivement abandonné**.
- **Action :** aucune future expérimentation ne doit chercher à se comparer à cette valeur.

### 2.2 Physique du capteur (Atracsys STK300) et invariance des features

#### 2.2.1 Suppression du `spread`

- Les sphères sont **vissées rigidement** sur des instruments en **fibre de carbone**.
- La distance inter-sphères est donc une **constante physique** (variance nulle).
- **Une feature constante n'apporte aucune information** au modèle.
- **Action :** `spread` doit être **supprimée du pipeline**.

#### 2.2.2 Problématique de l'`axis_angle`

- `axis_angle` est mesuré **par rapport au repère de la caméra**.
- Si la caméra est déplacée pendant la manipulation ou la collecte, l'axis_angle **change artificiellement**.
- Cette feature **n'est PAS invariante aux changements de repère**.
- **Comparaison :**
  | Feature | Invariance au repère caméra ? |
  |---|---|
  | `spread` | Oui (mais constante → inutile) |
  | `axis_angle` | **NON** |
  | `velocity` | **Oui** (s'annule par dérivation) |
  | `accel` | **Oui** |
  | `jerk` | **Oui** |
- **Action :** `axis_angle` doit être **supprimée**.

#### 2.2.3 La perte de la pose 6DOF (corollaire de `N_SPHERES = 1`)

- Avec **1 seule sphère**, l'Atracsys ne peut **pas** calculer l'orientation 3D.
- Le calcul d'une pose 6DOF requiert **au minimum 3 sphères non colinéaires**.
- L'appareil ne renvoie donc que la **position pure $(x, y, z)$** de la sphère — **pas** le centre géométrique d'un corps rigide.
- **Conséquence directe :** l'`axis_angle` actuellement calculé est **faux ou nul** — confirmation supplémentaire qu'il faut le retirer.

#### 2.2.4 Géométrie des instruments — *clarification clinique*

- **Pas de classification ni de segmentation par type d'instrument.**
- Objectif clinique réel : **alléger l'outil au maximum** (fibre de carbone) avec un **set minimal de sphères (idéalement 3)** pour obtenir la pose du corps rigide **sans perturber le chirurgien**.
- **Mots d'ordre :** capturer des mouvements variés **peu importe l'instrument**.
- **Faux positifs (fantômes IR) :** à filtrer via
  1. le **logiciel** de l'Atracsys (volume de travail 3D), et/ou
  2. notre **filtre de coordonnées maximum**.

### 2.3 Réévaluation de l'expérience MAE (slides 10 & 12)

#### 2.3.1 Sens de la validation loss du MAE

- La validation loss (∈ $[0, 1]$) correspond à l'**erreur de reconstruction calculée EXCLUSIVEMENT sur les valeurs masquées** de la séquence — **PAS** sur la séquence complète.
- Interprétation rigoureuse : une **loss de 0.5** signifie que **la reconstruction des signaux temporels masqués est médiocre**.
- **Implication :** toute future communication de résultats MAE doit explicitement rappeler cette définition.

#### 2.3.2 Données Ex Vivo vs Données Cliniques

- Le MAE a donné de **meilleurs résultats sur les données ex vivo**.
- **Raison :** bien que les ex vivo ne soient **pas labellisées en expertise**, les trajectoires y sont **plus continues** (moins d'occlusions, gestes plus déterministes).
- **Reformulation du but du MAE (Slide 12) :**
  > Le MAE n'a **PAS** vocation à extraire des embeddings discriminants de niveau d'expertise.
  >
  > Il sert de **modèle de fondation** (fond de domaine de tracking) pour **combler les données manquantes dues aux occlusions** lors des prédictions futures.

### 2.4 Mutation de l'architecture et correction du Biais Central

#### 2.4.1 Origine du Biais Central

- La **moyenne** des fenêtres **détruit le signal d'expertise**.
- En chirurgie réelle, un **expert** traverse des **« temps morts » statiques** où son profil cinématique ressemble à celui d'un novice.
- Moyenner ces fenêtres **baisse artificiellement** la note globale de l'expert.

#### 2.4.2 Corrections immédiates

1. La **médiane** doit remplacer la **moyenne** dans l'agrégation par trial.
2. Un **filtrage strict** doit éliminer toute fenêtre dont le pourcentage d'occlusion (`1 - valid_ratio`) est trop élevé.

#### 2.4.3 Mort des fenêtres fixes de 32 frames

- Le découpage en blocs arbitraires **casse la mémoire à long terme**.
- **Nouvelle directive :** passer à un **traitement temporel continu**.
  - À chaque instant $t$, le modèle prend en compte **tout l'historique** des frames précédentes,
  - **OU** une **mémoire glissante** d'une durée minimale de **1 minute** (pour oublier le passé obsolète).
  - Prédiction **en continu** du score d'expertise à l'instant $t$.
  - **Mise à jour par pas de $N$ frames** (à paramétrer expérimentalement).

### 2.5 Nouveau protocole d'entraînement et visualisation

#### 2.5.1 Stratégie d'entraînement aux extrêmes

- Entraînement **uniquement** sur les deux classes extrêmes :
  - **Classe 0 — Students** (très nombreux),
  - **Classe 8 — Staffs / Experts**.
- Toutes les classes intermédiaires (**PGY1 à PGY6, Fellows**) sont **exclues de l'entraînement**.

#### 2.5.2 Validation aveugle

- Les classes intermédiaires (PGY1 → PGY6, Fellows) servent **exclusivement d'ensemble de validation**.
- La **droite linéaire LOOCV** et le **$R^2$** sont calculés **UNIQUEMENT sur ces classes du milieu**.
- **Ne pas sur-optimiser le $R^2$** : il indique simplement comment les données épousent la dynamique du modèle.

#### 2.5.3 Normalisation de l'affichage temporel

| Axe | Plage normalisée | Sémantique |
|---|---|---|
| **$X$** | $[0, 1]$ | $0$ = début du trial, $1$ = fin du trial |
| **$Y$** | $[-1, +1]$ | Score d'expertise en continu |

- **Trajectoires attendues :**
  - **Experts :** débutent à $0$ et **convergent vers $+1$**.
  - **Novices :** **divergent vers $-1$**.
  - **Gestes dangereux isolés :** chute brutale du score à un instant $t$ — doit être **visuellement détectable** sur la courbe.

#### 2.5.4 Étude d'ablation

- Une **étude d'ablation systématique** est requise : **avec MAE** vs **sans MAE**, pour quantifier l'apport réel du pré-entraînement.

---

## PARTIE 3 — Feuille de route opérationnelle (post-réunion)

Les actions ci-dessous découlent directement de la PARTIE 2. Toute PR doit
référencer la tâche correspondante.

| Tâche | Description | Statut |
|---|---|---|
| **A** | Abandonner officiellement le benchmark $r = 0{,}555$ (mise à jour de tous les commentaires / docstrings / slides). | **Fait (audit local 2026-05-22)** — aucun script Python du repo local ne contient cette constante. À répercuter sur les slides + scripts Narval (`scripts/mae_pretrain.py`, LSTM régression) à leur prochaine mise à jour. |
| **B** | Réduire la matrice de features de **6 → 4** : conserver `velocity`, `accel`, `jerk`, `valid_ratio` ; supprimer `spread` et `axis_angle`. | **Étage 1 fait (2026-05-22)** — slicing aval dans `build_mae_dataset_enrichi.py`. Étage 2 (`tracking_hungarian.py`) en attente. |
| **C** | Adapter toutes les dimensions d'entrée du MAE et du LSTM à la nouvelle profondeur de 4 features. | **Pipeline local fait (2026-05-22)** — adaptation des scripts Narval (`mae_pretrain.py`, LSTM régression) à effectuer après récupération depuis le cluster. |
| **D** | Modifier le split Train / Test : entraîner **uniquement sur Classe 0 (Students) + Classe 8 (Staffs)** ; valider uniquement sur PGY1–PGY6 + Fellows. | **Fait (2026-05-22)** — `split_extremes_vs_blind()` dans `continuous_scorer.py`. Validé end-to-end sur 55 trials train + 69 trials val aveugle. |
| **E** | Remplacer l'agrégation **moyenne → médiane** pour les scores par trial. | **Fait (2026-05-22)** — `aggregate_trial_score(scores, mode="median")` dans `continuous_scorer.py`. Adapté au paradigme continu (médiane sur la time-series score(t)). |
| **F** | Filtrage strict des fenêtres à forte occlusion ($1 - \text{valid\_ratio}$ élevé). | **Fait (2026-05-22)** — `filter_occluded()` dans `continuous_scorer.py`. 12/136 trials rejetés au seuil `min_valid_ratio=0.30`. |
| **G** | Migration du traitement par **fenêtres fixes 32 frames** vers un **traitement temporel continu** avec mémoire glissante ≥ 1 min, mise à jour par pas de $N$ frames. | **Fait + entraîné (2026-05-22)** — modèle LSTM causal dans `src/continuous_scorer.py` ; auto-test synthétique OK ; entraînement réel sur 124 trials (Pearson aveugle +0.26 en 3 epochs sous-optimisés). |
| **H** | Normalisation de l'affichage : $X \in [0, 1]$, $Y \in [-1, +1]$ ; visualisation des gestes dangereux isolés. | **Fait (2026-05-22)** — fonctions `plot_score_evolution()` et `plot_blind_predictions()` dans `continuous_scorer.py`. |
| **I** | Étude d'ablation systématique **avec / sans MAE**. | **Infrastructure prête** — flag `use_mae_encoder` dans `ScorerConfig`. Reste à brancher l'encoder MAE pré-entraîné (mae_run3 sur Narval). |

---

## PARTIE 4 — Architecture du scoring temporel continu (Tâches G + H)

> Cette partie formalise la **rupture architecturale** décidée en réunion (cf. §2.4.3) :
> abandon des fenêtres fixes de 32 frames pour le scoring d'expertise au profit
> d'un **traitement temporel continu**.

### 4.1 Pourquoi les fenêtres fixes de 32 frames sont indéfendables

Quatre défauts incompatibles avec la mission scientifique du projet :

#### 4.1.1 Découpage arbitraire

Si un geste chirurgical critique se déroule entre les frames `f30` et `f35` :
- Fenêtre 1 (f1→f32) ne capture que `f30-f32`,
- Fenêtre 2 (f17→f48) capture le geste mais le mélange à 16 frames de contexte non pertinent,
- Aucune fenêtre ne voit **le geste complet et isolé**.

Le découpage `(WIN_LEN, STRIDE)` n'a aucune signification physiologique : il est lié à
l'architecture du Transformer, pas à la chirurgie.

#### 4.1.2 Perte de contexte long terme

Le MAE traite chaque fenêtre **isolément**. Quand il analyse la fenêtre 2, il n'a
**aucune connaissance** de ce qu'a fait le chirurgien dans la fenêtre 1.
→ Pas de mémoire à long terme. Or l'expertise se manifeste **précisément** dans la
manière dont les gestes s'enchaînent sur des dizaines de secondes.

#### 4.1.3 Dépendance à la fréquence d'acquisition

| Fréquence Atracsys | Durée de 32 frames |
|---|---|
| 10 Hz  | **3,2 s** |
| 25 Hz  | 1,3 s |
| 100 Hz | 0,32 s |
| 300 Hz | **0,11 s** |

Le même `WIN_LEN = 32` représente des durées **30× différentes** selon la
configuration matérielle. Une fenêtre de 0,1 s n'a aucun sens chirurgical, et il est
contre-scientifique d'avoir un modèle dont le sens dépend du réglage de la caméra.

#### 4.1.4 Temps morts → biais central

Un chirurgien expert peut rester **immobile pendant 5 s** (réflexion, consultation
d'image, vérification mentale). Découpé en fenêtres de 32 frames, ce silence
produit ~1-2 fenêtres de "non-mouvement". Ces fenêtres :
1. N'ont aucun signal cinématique discriminant,
2. Sont prédites par défaut comme **`expertise ≈ 2`** (centre de la distribution),
3. **Tirent la moyenne par trial vers le centre** quand on agrège.

C'est la **cause directe et identifiée** du biais 1.2-2.7 observé avant la réunion.

### 4.2 La nouvelle architecture — scoring temporel continu

#### 4.2.1 Définition formelle

Soit une séquence de features cinématiques pour un trial :

$$
\mathbf{X} \in \mathbb{R}^{T \times 4}, \quad \mathbf{X}_t = (v_t, a_t, j_t, r_t)
$$

avec $v_t$ = vélocité, $a_t$ = accélération, $j_t$ = jerk, $r_t$ = valid_ratio.

**Objectif :** apprendre une fonction $\phi$ telle que pour tout instant $t$ :

$$
\phi : \mathbf{X}_{0:t} \mapsto s_t \in [-1, +1]
$$

où $s_t$ est le score d'expertise à l'instant $t$, calculé à partir de
**toute l'histoire jusqu'à $t$ inclus**.

**Contraintes :**
- $\phi$ doit être **causal** : $s_t$ ne dépend que de $\mathbf{X}_{0:t}$, jamais du futur.
- $\phi$ doit être **streaming-compatible** : on doit pouvoir mettre à jour $s_t$
  incrémentalement à mesure que de nouvelles frames arrivent (pas de re-calcul depuis $t=0$).

#### 4.2.2 Choix d'architecture : LSTM causal récurrent

Trois architectures candidates ont été considérées :

| Option | Mémoire | Inférence streaming | Coût mémoire | Verdict |
|---|---|---|---|---|
| **LSTM unidirectionnel causal** | État caché compressé, théoriquement infinie | $O(1)$ par frame (incrémental) | Faible | ✅ **retenu** |
| Transformer causal (autoregressive) | Attention sur tout l'historique | $O(t)$ par frame | Quadratique en $t$ | Trop lourd pour les longs trials |
| Sliding window 1 min (600 frames) | 600 dernières frames | $O(600)$ par frame | Constant | Compromis acceptable, à garder en variante d'ablation |

→ **Architecture principale : LSTM unidirectionnel.** Le ou les états cachés `(h_t, c_t)`
encodent toute l'histoire passée dans un vecteur de taille fixe (typiquement 128 dims).
L'inférence est en $O(1)$ par frame — idéal pour le streaming.

Variante d'ablation (Tâche I) : LSTM avec sliding memory de 1 min, pour quantifier
l'apport de la mémoire longue.

#### 4.2.3 Topologie complète

```
Input : X[0:t] de shape (t, 4)
   │
   ↓
┌──────────────────────────────────┐
│  Optionnel : MAE encoder (gelé)  │  → enrichit chaque frame avec le contexte
│  shape (t, 4) → (t, 64)          │     local appris en self-supervised.
└──────────────────────────────────┘     Désactivable via flag --no-mae (Tâche I).
   │
   ↓
┌──────────────────────────────────┐
│  LSTM(128, return_sequences=True)│  → produit un état caché par frame
│  shape (t, 64) → (t, 128)        │
└──────────────────────────────────┘
   │
   ↓
┌──────────────────────────────────┐
│  Dense(64, relu)                 │
│  Dense(1, tanh)                  │  → score ∈ [-1, +1]
│  shape (t, 128) → (t, 1)         │
└──────────────────────────────────┘
   │
   ↓
Output : s[0:t] de shape (t, 1)
```

**Pourquoi `tanh` en sortie ?** Force le score dans $[-1, +1]$ par construction
(cf. Tâche H : axe $Y$ normalisé entre $-1$ et $+1$). Pas besoin de post-processing.

#### 4.2.4 Hyperparamètres clés

| Hyperparamètre | Valeur par défaut | Justification |
|---|---|---|
| `LSTM_UNITS` | 128 | Compromis capacité / surapprentissage sur ~94 trials labellisés |
| `DENSE_HIDDEN` | 64 | Tête de régression légère |
| `STRIDE_INFERENCE` | 5 frames | À 10 Hz → un score toutes les 0,5 s ; visualisation fluide sans coût excessif |
| `MEMORY_MODE` | `"full"` | "full" = tout l'historique (LSTM standard) ; "sliding_60s" = mémoire glissante |
| `LEARNING_RATE` | 1e-3 | Standard Adam |
| `LOSS` | `Huber(delta=0.5)` | Plus robuste que MSE aux outliers (gestes dangereux isolés) |

### 4.3 Spécification des cibles d'entraînement (Tâche D appliquée à la régression)

Pour chaque trial de classes extrêmes :

| Classe | Label brut (string) | Cible régression $y$ |
|---|---|---|
| 0 | "Medical student" | $-1{,}0$ |
| 8 | "Staff" | $+1{,}0$ |

Pour les classes intermédiaires (validation aveugle, pas d'entraînement) :

| Classe | Label brut | Cible projetée |
|---|---|---|
| 1 | "Resident PGY1" | $-0{,}75$ |
| 2 | "Resident PGY2" | $-0{,}50$ |
| 3 | "Resident PGY3" | $-0{,}25$ |
| 4 | "Resident PGY4" | $\phantom{+}0{,}00$ |
| 5 | "Resident PGY5" | $+0{,}25$ |
| 6 | "Resident PGY6" | $+0{,}50$ |
| 7 | "Fellow" | $+0{,}86$ |

**Formule** : $y_{\text{reg}} = \dfrac{i}{4} - 1$ pour $i \in \{0, \dots, 8\}$ (9 valeurs équidistantes
dans $[-1, +1]$), avec un **override** à $+0{,}86$ pour la classe Fellow afin de
préserver une marge avec les Staffs ($+1$).

Implémentation : `CLASS_TO_REG` dans `src/build_continuous_dataset.py`.

### 4.4 Visualisation (Tâche H)

Pour chaque trial, un graphe est produit :

| Axe | Plage | Sémantique |
|---|---|---|
| $X$ | $[0, 1]$ normalisé | $0$ = début du trial, $1$ = fin du trial |
| $Y$ | $[-1, +1]$ | Score d'expertise instantané $s_t$ |

**Lecture clinique attendue :**
- **Expert (Staff)** : courbe partant de $\sim 0$ (incertitude initiale) et **convergeant
  vers $+1$** à mesure que le LSTM accumule des preuves de gestes experts.
- **Novice (Student)** : courbe **divergeant vers $-1$**.
- **Geste dangereux isolé** : chute brutale et locale du score (artefact visible
  sur la courbe — point de discussion clinique).

### 4.5 Format des données d'entrée — séquences trial-level

Le pipeline MAE actuel (`build_mae_dataset_enrichi.py`) produit des fenêtres fixes
$(N, 32, 4)$. Ce format est **conservé pour le pré-entraînement MAE** (architecture
qui requiert des séquences de longueur fixe).

Pour le scoring continu, un **nouveau format** est requis :

```python
# continuous_per_trial.pkl  (à produire à partir des données labellisées)
{
    (participant_id, trial_id): {
        "X":      np.ndarray,   # shape (T_trial, 4) — séquence variable
        "y9":     int,          # classe 0..8
        "y_reg":  float,        # cible régression dans [-1, +1]
        "level":  str,          # ex : "Medical student", "Staff"
        "T":      int,          # nombre de frames
        "fs":     float,        # fréquence d'échantillonnage (Hz) — pour la normalisation X∈[0,1]
    },
    ...
}
```

→ La **construction** de ce dataset depuis les vraies données cliniques sera faite
côté Narval, à partir de `final_from_full_data.pkl` (via un nouveau script
`scripts/build_continuous_dataset.py` à créer après rapatriement).

### 4.6 Conservation et nouveau rôle du MAE

Le MAE pré-entraîné conserve sa raison d'être, mais **change de rôle** (cf. §2.3.2) :

| Rôle d'origine (avant réunion) | Rôle nouveau (post-réunion) |
|---|---|
| Extraire des embeddings discriminants d'expertise | **Modèle de fondation pour imputer les frames occluses** |
| Sortie utilisée directement pour scorer | Sortie utilisée comme features enrichies en entrée du LSTM continu |

Concrètement, dans le scoring continu :
1. La séquence brute $(T, 4)$ est éventuellement passée par l'**encoder MAE gelé** pour
   produire des features enrichies $(T, 64)$ qui interpolent intelligemment les
   trous d'occlusion.
2. Ces features enrichies alimentent le LSTM causal.
3. Le scoring final est réalisé par le LSTM, pas par le MAE.

Une étude d'ablation (Tâche I) comparera **avec MAE** vs **sans MAE** pour
quantifier l'apport réel de l'imputation.

---

## ANNEXE — Journal des modifications du pipeline local (2026-05-22)

### Pipeline scoring continu — état complet après la session du 2026-05-22

| Composant | Fichier | État |
|---|---|---|
| Constructeur du dataset continu | `src/build_continuous_dataset.py` | **Créé** — convertit `data/filtered_data.json` (1360 entrées brutes) en `data/continuous_per_trial.pkl` (136 trials, 10 features) |
| Modèle de scoring + entraînement | `src/continuous_scorer.py` | **Créé** — LSTM causal + boucle Tâche D + métriques aveugles + visualisations (Tâches G, H, D, E, F) |
| Dataset local | `data/continuous_per_trial.pkl` | **Produit** (26.3 MB, 136 trials, fs ≈ 10 Hz) |
| Premier modèle entraîné | `results_continuous/scorer.keras` | **Produit** (3 epochs, sous-optimisé, Pearson aveugle +0.26) |

### Pipeline MAE local — état après application de A + B Étage 1 + C

| Composant | Fichier | État |
|---|---|---|
| Producteur des features brutes | `src/tracking_hungarian.py` | **Inchangé** — produit toujours `features_6ch.npy` shape `(T, 6)` |
| Constructeur du dataset MAE | `src/build_mae_dataset_enrichi.py` | **Modifié** — slicing `KEEP_IDX = [0, 1, 2, 5]` ; sortie `X_pretrain_4ch.npy` shape `(N, 32, 4)` |
| Pré-entraînement MAE | `scripts/mae_pretrain.py` (Narval) | **Hors repo** — modifications à appliquer (cf. checklist ci-dessous) |
| LSTM régression aval | `scripts/lstm_regression.py` (Narval) | **Hors repo** — modifications à appliquer (cf. checklist ci-dessous) |

### Checklist pour les scripts Narval

Lorsque les scripts Narval seront rapatriés depuis `~/icems/scripts/`, appliquer
les modifications suivantes pour terminer la Tâche C :

#### Pour `scripts/mae_pretrain.py`

1. Charger `X_pretrain_4ch.npy` (au lieu de `X_pretrain_v1_enrichi.npy` ou `final_from_full_data.pkl`).
2. Vérifier que l'argument CLI `--in_dim` (ou `--n_features`) accepte bien la nouvelle valeur **4**.
3. Mettre à jour `run_mae.sh` avec :
   ```bash
   --data /home/malek1/icems/data/X_pretrain_4ch.npy \
   --output /home/malek1/icems/results/mae_run3 \
   --in_dim 4
   ```
4. Le `--mask 0.40`, `--embed 64`, `--heads 4`, `--blocks 3`, `--ff 128` restent valides
   (l'embed `Dense(64)` s'adapte automatiquement à l'entrée 4D).
5. **Important** : les checkpoints existants `mae_run1` / `mae_run2` (entraînés sur 6 canaux)
   sont **incompatibles** et doivent être marqués comme obsolètes.

#### Pour le LSTM régression aval

1. Adapter `n_features = 4` (au lieu de 6).
2. Charger les `norm_params_4ch.npy` au lieu des `norm_params_enrichi.npy`.
3. Préparer le terrain pour la **Tâche D** (split `{0, 8}` train / `{1..7}` val) en
   ajoutant un masque conditionnel sur `y9`.
4. Préparer le terrain pour les **Tâches E et G** (médiane vs moyenne, mémoire glissante).

#### Audit final post-Narval

- [ ] Toute mention de `r = 0{,}555` purgée des docstrings et des commentaires.
- [ ] Toute mention de `n_features = 6` ou `in_dim = 6` remplacée par `4`.
- [ ] Aucun chemin de fichier ne pointe encore vers `X_pretrain_v1_enrichi.npy`,
      `norm_params_enrichi.npy` ou `meta_enrichi.csv`.
- [ ] Le script de pré-entraînement crash explicitement (assertion) si on lui passe
      un dataset shape `(N, *, 6)`.

---

*Fin du document. Toute modification ultérieure de cette feuille de route doit
être validée explicitement par le superviseur.*
