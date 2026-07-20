# Guide Narval — malek1 @ narval.alliancecan.ca

Run **GPU A100 sur Narval**, pas sur CPU local.

**Utilisateur :** `malek1`  
**Terminal local :** PowerShell (Windows) — pas Git Bash.

---

## Étape 1 — Transférer le projet (PowerShell Windows)

Ouvre **PowerShell** et exécute :

```powershell
# Dossier local du projet
$LOCAL = "C:\Users\malek\Downloads\ICEMS-main (6)\ICEMS-main"
$REMOTE = "malek1@narval.alliancecan.ca:~/ICEMS-main"

# Créer le dossier distant (première fois)
ssh malek1@narval.alliancecan.ca "mkdir -p ~/ICEMS-main"

# Envoyer les dossiers essentiels (sans .venv)
scp -r "$LOCAL\src"                          "${REMOTE}/"
scp -r "$LOCAL\data"                         "${REMOTE}/"
scp -r "$LOCAL\experiments"                  "${REMOTE}/"
scp    "$LOCAL\requirements.txt"             "${REMOTE}/" 2>$null
```

Si `requirements.txt` n'existe pas, ignore l'erreur de la dernière ligne.

**Première installation complète** (tout le repo, plus long) :

```powershell
scp -r "$LOCAL\*" malek1@narval.alliancecan.ca:~/ICEMS-main/
```

---

## Étape 2 — Connexion SSH (PowerShell)

```powershell
ssh malek1@narval.alliancecan.ca
```

Tu es maintenant **sur Narval** (shell Linux). Les commandes ci-dessous s'exécutent là-bas.

```bash
cd ~/ICEMS-main

module load StdEnv/2023 python/3.11 cuda/12.2

# Première fois seulement
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy pandas scipy scikit-learn matplotlib seaborn

# Vérifier GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Attendu : `True NVIDIA A100-SXM4-80GB` (ou similaire).

---

## Étape 3 — Smoke test GPU (sur Narval)

Toujours connecté en SSH sur Narval :

```bash
cd ~/ICEMS-main
SMOKE=1 bash experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/slurm/submit_narval.sh
```

Le script affiche un **JOBID** (ex. `12345678`).

---

## Étape 4 — Suivre le job depuis PowerShell (sans rester connecté)

Remplace `12345678` par ton JOBID :

```powershell
# Jobs en cours
ssh malek1@narval.alliancecan.ca "squeue -u malek1"

# Dernières lignes du log (task 0 = seed 42)
ssh malek1@narval.alliancecan.ca "tail -30 ~/ICEMS-main/experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/logs/parallel-12345678_0.out"

# Suivre en direct (Ctrl+C pour quitter)
ssh malek1@narval.alliancecan.ca "tail -f ~/ICEMS-main/experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/logs/parallel-12345678_0.out"
```

Dans le log, vérifie : `cuda=True` et un nom de GPU A100.

Résultat smoke attendu :

```
~/ICEMS-main/experiments/.../results/smoke/seed_42/metrics.json
```

---

## Étape 5 — Run complet (5 seeds GPU en parallèle)

Sur Narval (SSH) :

```bash
cd ~/ICEMS-main
bash experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/slurm/submit_narval.sh
```

| Task | Seed | GPU |
|------|------|-----|
| 0 | 42 | A100 ×1 |
| 1 | 123 | A100 ×1 |
| 2 | 456 | A100 ×1 |
| 3 | 789 | A100 ×1 |
| 4 | 2024 | A100 ×1 |

---

## Étape 6 — Agréger les résultats (sur Narval)

Quand `squeue` ne montre plus de jobs :

```bash
cd ~/ICEMS-main
source .venv/bin/activate
python experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/aggregate_results.py
```

---

## Étape 7 — Télécharger les résultats (PowerShell Windows)

```powershell
$DEST = "C:\Users\malek\Downloads\ICEMS-narval-results"
New-Item -ItemType Directory -Force -Path $DEST

scp -r malek1@narval.alliancecan.ca:~/ICEMS-main/experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/results/ $DEST
```

---

## Commandes utiles

**PowerShell (depuis Windows) :**

```powershell
ssh malek1@narval.alliancecan.ca "squeue -u malek1"
ssh malek1@narval.alliancecan.ca "sacct -j 12345678 --format=JobID,State,Elapsed"
ssh malek1@narval.alliancecan.ca "scancel 12345678"
```

**Sur Narval (après ssh) :**

```bash
squeue -u malek1
sacct -j 12345678 --format=JobID,JobName%20,State,Elapsed,MaxRSS
scancel 12345678
```

---

## Dépannage

| Problème | Solution |
|----------|----------|
| `ssh: command not found` | Activer OpenSSH : Paramètres → Applications → Fonctionnalités optionnelles → Client OpenSSH |
| `Permission denied` | Vérifier mot de passe / clé SSH Alliance Canada |
| `QOSMaxJobsPerUserLimit` | `sbatch --array=0-1 experiments/.../slurm/run_parallel.sbatch` (2 seeds) |
| `.venv introuvable` | Refaire étape 2 sur Narval |
| `cuda=False` | `module load cuda/12.2` + réinstaller torch sur Narval |
| Job TIMEOUT | Augmenter `#SBATCH --time=06:00:00` dans `run_parallel.sbatch` |

---

## Récap express

```powershell
# 1. Windows — envoyer le code
scp -r "C:\Users\malek\Downloads\ICEMS-main (6)\ICEMS-main\experiments" malek1@narval.alliancecan.ca:~/ICEMS-main/

# 2. Windows — se connecter
ssh malek1@narval.alliancecan.ca
```

```bash
# 3. Narval — smoke puis run complet
cd ~/ICEMS-main && source .venv/bin/activate
SMOKE=1 bash experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/slurm/submit_narval.sh
bash experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/slurm/submit_narval.sh
python experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/aggregate_results.py
```

```powershell
# 4. Windows — récupérer les résultats
scp -r malek1@narval.alliancecan.ca:~/ICEMS-main/experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/results/ C:\Users\malek\Downloads\ICEMS-narval-results\
```
