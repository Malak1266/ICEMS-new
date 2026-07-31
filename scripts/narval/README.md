# ICEMS sur Narval (Alliance Canada)

Guide pour lancer le LOPO Step B avec `--aug-target` sur le cluster Narval.

## Prérequis

- Compte Alliance Canada actif
- Allocation Slurm valide (`def-xxxxx` ou `rrg-xxxxx`)
- Accès SSH : `ssh votre_cc@narval.alliancecan.ca`
- [Configuration MFA / clés SSH](https://docs.alliancecan.ca/wiki/SSH_Keys)

## 1. Transférer le projet sur Narval

Depuis votre machine locale (PowerShell / WSL / Git Bash) :

```bash
# Adapter le chemin local
scp -r "C:/Users/malek/Downloads/ICEMS-main (6)/ICEMS-main" \
    votre_cc@narval.alliancecan.ca:~/ICEMS-main
```

Ou avec `rsync` (plus rapide pour les mises à jour) :

```bash
rsync -avz --exclude .venv --exclude __pycache__ \
    "/c/Users/malek/Downloads/ICEMS-main (6)/ICEMS-main/" \
    votre_cc@narval.alliancecan.ca:~/ICEMS-main/
```

**Fichier indispensable :** `data/continuous_per_trial.pkl` (~quelques Mo).

## 2. Première connexion et environnement

```bash
ssh votre_cc@narval.alliancecan.ca
cd ~/ICEMS-main

# Vérifier votre compte Slurm
account
# ou
sshare -U

bash scripts/narval/setup_env.sh
source .venv/bin/activate
```

## 3. Configurer le job Slurm

Éditez `scripts/narval/run_scaled_lopo.sbatch` :

```bash
nano scripts/narval/run_scaled_lopo.sbatch
```

Remplacez la ligne :

```
#SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT
```

par votre allocation, par ex. `#SBATCH --account=def-votreprof`.

## 4. Stratégie de runs (recommandée)

| Phase | Commande | Durée estimée | Objectif |
|-------|----------|---------------|----------|
| **A — Smoke** | voir ci-dessous | ~30–60 min | Vérifier que le pipeline tourne |
| **B — Intermédiaire** | `AUG_TARGET=100`, 47 folds | plusieurs h | Premier vrai résultat |
| **C — Production** | `AUG_TARGET=200`, 47 folds | très long (24h+) | ~800 trials/fold |

### Phase A — Smoke test (1 fold)

```bash
mkdir -p logs
sbatch --export=ALL,AUG_TARGET=45,MAX_FOLDS=1,EPOCHS=5,MC_PASSES=5,OUT_DIR=results/smoke_narval \
    scripts/narval/run_scaled_lopo.sbatch
```

### Phase B — Run intermédiaire (recommandé en premier)

```bash
sbatch --export=ALL,AUG_TARGET=100,EPOCHS=100,OUT_DIR=results/run_aug100 \
    scripts/narval/run_scaled_lopo.sbatch
```

### Phase C — Run production scaled

```bash
sbatch --time=48:00:00 --export=ALL,AUG_TARGET=200,EPOCHS=100,OUT_DIR=results/run_aug200 \
    scripts/narval/run_scaled_lopo.sbatch
```

> **Note :** `--aug-target 200` génère ~670 DBA DTW par fold × 47 folds. Prévoyez 24–48 h et surveillez les logs.

## 5. Suivre le job

```bash
squeue -u $USER                    # jobs en cours
tail -f logs/icems_scaled-JOBID.out   # log en direct
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS
```

Annuler un job :

```bash
scancel JOBID
```

## 6. Récupérer les résultats

Sur Narval, les fichiers sont dans `results/run_augXXX/` :

- `lopo_predictions.pkl` — prédictions détaillées
- `confusion_matrix.png`
- `scatter.png`
- `score_vs_time.png`

Télécharger vers votre PC :

```bash
scp -r votre_cc@narval.alliancecan.ca:~/ICEMS-main/results/run_aug100 ./results_from_narval/
```

Régénérer un graphique sans relancer le LOPO (sur Narval ou en local) :

```bash
python src/step_B_classification.py \
    --plot-only \
    --out results/run_aug100
```

## 7. Lire les métriques dans le log

À la fin de `logs/icems_scaled-*.out`, cherchez :

```
Métriques LOPO (participants tenus out)
  Pearson  r = ...
  Spearman r = ...
  Accuracy 4 classes = ...%
```

## Dépannage

| Problème | Solution |
|----------|----------|
| `Invalid account` | Corriger `#SBATCH --account=` |
| `data/continuous_per_trial.pkl` manquant | `scp` le fichier ou `python src/build_continuous_dataset.py` |
| Job tué (OOM) | Augmenter `#SBATCH --mem=128G` |
| Trop lent | Réduire `AUG_TARGET` (100 → 50) ou `MAX_FOLDS` pour test |
| CUDA non dispo | `module load cuda/12.2` + PyTorch CUDA dans `setup_env.sh` |

## Run 5 baseline (sans aug-target)

Pour comparer avec Run 5 original :

```bash
sbatch --export=ALL,OUT_DIR=results/run5_baseline \
    --wrap='cd $HOME/ICEMS-main && source .venv/bin/activate && python src/step_B_classification.py --data data/continuous_per_trial.pkl --out results/run5_baseline --epochs 100'
```

Ou soumettre sans `--aug-target` (équilibrage majoritaire du fold).
