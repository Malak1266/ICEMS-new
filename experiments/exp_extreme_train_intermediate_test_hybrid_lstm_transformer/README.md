# Expérience : Train extrêmes → Test intermédiaires (Hybrid LSTM-Transformer)

Pipeline isolé — **aucun fichier du repo principal n'est modifié**.

## Protocole

| Split | Niveaux | Rôle |
|-------|---------|------|
| **TRAIN** | Medical Student (`ms`) + Expert (`staff`) | Entraînement + augmentation |
| **TEST** | PGY1–PGY5 (junior) + PGY6 + Fellow (senior) | Évaluation généralisation |

- Données : `data/continuous_per_trial.pkl` (trajectoires ATRACSYS, 10 canaux → 13 enrichis)
- Modèle : `HybridLSTMTransformer` (import depuis `src/models/`)
- Loss : HOEL (mêmes hyperparamètres que `train_hybrid_lopo.py`)
- Augmentation (train uniquement) : jitter, time-warp, magnitude scaling, masking partiel

## Lancement Narval (GPU A100) — RECOMMANDÉ

Guide pas à pas : **[NARVAL.md](NARVAL.md)**

```bash
ssh malek1@narval.alliancecan.ca
cd ~/ICEMS-main
source .venv/bin/activate

# Smoke GPU (1 seed, ~15 min)
SMOKE=1 bash experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/slurm/submit_narval.sh

# Run complet (5 seeds GPU en parallèle)
bash experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/slurm/submit_narval.sh

# Agrégation après completion
python experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/aggregate_results.py
```

Depuis **PowerShell Windows** (transfert + suivi) : voir **[NARVAL.md](NARVAL.md)**.

## Lancement local (debug CPU uniquement — ne pas utiliser pour les résultats finaux)

```powershell
cd ICEMS-main
$env:PYTHONPATH = "src"
python experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/run_parallel.py --smoke --workers 2
```

## Sorties

```
experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/results/
  seed_42/
    model.pt
    metrics.json          # MAE, MSE, Spearman + par sous-groupe
    predictions.pkl
    error_distribution.png
    scatter_intermediate.png
  aggregate.json          # agrégat multi-seeds
```
