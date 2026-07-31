# Avancement du pipeline ICEMS

Date: 2026-05-05
Utilisateur: Malak1266

## Objectif
Préparer un nouveau dataset à partir du fichier JSON brut, l’uploader sur Narval, puis lancer un run MAE sur GPU.

## Ce qui a été fait

### 1) Repérage du dépôt GitHub
Dépôt identifié:
- `D3MIA/ICEMS`

Fichiers repérés comme utiles:
- `generate_data.py`
- `ICEMS.py`
- `scripts/run_mae.sh`
- `scripts/mae_pretrain.py`

### 2) Conversion du JSON brut en pickle
Fichier source local:
- `C:\Users\malek\Downloads\full_data (1).json`

Commande utilisée:
```powershell
python "C:\Users\malek\Downloads\ICEMS-main (3)\ICEMS-main\generate_data.py" --full_json "C:\Users\malek\Downloads\full_data (1).json" --out_pkl "C:\Users\malek\OneDrive\Desktop\pipeline_output\final_from_full_data.pkl"
```

Résultat:
- `final_from_full_data.pkl` créé avec succès
- `final_from_full_data_meta.json` créé avec succès
- 680 entrées générées

### 3) Organisation locale
Les fichiers ont été copiés dans:
- `C:\Users\malek\OneDrive\Desktop\pipeline_output\run2`

Fichiers présents:
- `final_from_full_data.pkl`
- `final_from_full_data_meta.json`

### 4) Transfert vers Narval
Les fichiers ont été transférés vers:
- `~/icems/data/`

Fichiers visibles sur Narval:
- `final_from_full_data.pkl`
- `final_from_full_data_meta.json`
- anciens fichiers toujours présents:
  - `final_from_full_B.pkl`
  - `final_from_full_B_meta.json`
  - `X_pretrain.npy`

### 5) Vérification des scripts de run
Fichiers sur Narval:
- `~/icems/scripts/mae_pretrain.py`
- `~/icems/scripts/run_mae.sh`

Contenu initial de `run_mae.sh`:
```bash
python /home/malek1/icems/scripts/mae_pretrain.py \
    --data   /home/malek1/icems/data/X_pretrain.npy \
    --output /home/malek1/icems/results/mae_run1 \
    --epochs 200 --batch 64 --lr 1e-3 --mask 0.40 \
    --embed 64 --heads 4 --blocks 3 --ff 128
```

### 6) Modification du chemin dataset dans `run_mae.sh`
Le script a été modifié pour pointer vers le nouveau dataset:
- `--data   /home/malek1/icems/data/final_from_full_data.pkl`
- `--output /home/malek1/icems/results/mae_run2`

### 7) Soumission du job Slurm
Job soumis sur Narval:
- `60441735`

État observé dans `squeue`:
- `PD` (Pending)
- raison: `Priority`

## État actuel
Le job est soumis mais en attente de ressources GPU. Aucun log n’est encore disponible tant que le job ne passe pas en exécution.

## Prochaines étapes
1. Surveiller l’état du job avec:
```bash
squeue -u $USER
```
2. Dès que le job passe en `R`, suivre le log avec:
```bash
tail -f ~/icems/logs/mae_60441735.out
```
3. Si le job disparaît de `squeue`, diagnostiquer avec:
```bash
sacct -j 60441735 --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS
```
4. Vérifier dans les logs que le modèle charge bien:
- `final_from_full_data.pkl`
- le dossier de sortie `mae_run2`

## Notes
- L’erreur `can't open file '...generate_data.py'` a été résolue en utilisant le chemin complet du script GitHub local.
- L’erreur `FileNotFoundError: full_data__1_.json` a été résolue en utilisant le vrai nom/chemin du fichier: `full_data (1).json`.
- L’erreur `tail: cannot open ... No such file or directory` est normale tant que le job Slurm n’a pas encore démarré.
