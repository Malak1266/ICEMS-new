# 🏥 ICEMS - Surgical Expertise Prediction Model

## Modèle Hybride de Prédiction d'Expertise Chirurgicale

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00?style=flat-square&logo=tensorflow)](https://tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

Ce projet implémente un **modèle de deep learning hybride avancé** pour prédire le niveau d'expertise chirurgicale à partir de données de simulation neurochirurgicale. Le système combine les architectures **Transformer**, **LSTM**, **GRU** et **CNN** pour classifier 9 niveaux d'expertise distincts, allant d'étudiant en médecine au personnel senior.

---

## 🎯 Objectifs du Projet

- **Classification multi-classe** : Prédiction de 9 niveaux d'expertise chirurgicale
- **Architecture hybride** : Combinaison optimale de Transformer + LSTM + GRU + CNN
- **Gestion du déséquilibre** : Techniques avancées pour équilibrer les classes rares
- **Évaluation complète** : Métriques spécialisées et matrices de confusion détaillées

---

## 📊 Niveaux d'Expertise Chirurgicale

Le modèle classifie **9 niveaux d'expertise** basés sur la formation médicale :

| Niveau | Description | Score Continu |
|--------|-------------|---------------|
| 0 | Medical Student | 0.000 |
| 1 | Resident PGY1 | 0.125 |
| 2 | Resident PGY2 | 0.250 |
| 3 | Resident PGY3 | 0.375 |
| 4 | Resident PGY4 | 0.500 |
| 5 | Resident PGY5 | 0.625 |
| 6 | Resident PGY6 | 0.750 |
| 7 | Fellow (toutes spécialités) | 0.875 |
| 8 | Staff | 1.000 |

---

## 🏗️ Architecture du Modèle Hybride

### Composants Principaux

```
📡 INPUT (50 timesteps × features)
    ↓
🧠 FEATURE PROJECTION (256D) + Positional Encoding
    ↓
┌─────────────────────────────────────────────────────────┐
│  🔄 TRANSFORMER ENCODER (2 blocs)                      │
│  • Multi-Head Attention (8 têtes)                      │
│  • Feed-Forward Networks                               │
│  • Residual Connections + Layer Normalization          │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  🔁 LSTM BIDIRECTIONNEL (2 niveaux)                   │
│  • LSTM Primaire : 128 unités                         │
│  • LSTM Secondaire : 96 unités                        │
│  • Dropout & Batch Normalization                      │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  ⚡ GRU BIDIRECTIONNEL (2 niveaux)                     │
│  • GRU Primaire : 96 unités                           │
│  • GRU Secondaire : 64 unités                         │
│  • Optimisé pour capture temporelle                   │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  🔲 CNN 1D (Patterns locaux)                          │
│  • Conv1D : 128 + 256 filtres                         │
│  • Détection de motifs chirurgicaux                   │
└─────────────────────────────────────────────────────────┘
                    ↓
🔗 FUSION CROSS-ATTENTION
                    ↓
📊 MULTI-SCALE POOLING
    • Global Max/Average Pooling
    • Attention Temporelle Pondérée
    • Pooling Multi-Segments
                    ↓
🎯 CLASSIFICATION HEAD (512→256→128→9 classes)
```

### Innovations Techniques

- **Focal Loss** : Gestion optimisée des classes déséquilibrées
- **Poids de classe adaptatifs** : Boost spécial pour les niveaux critiques
- **Augmentation de données chirurgicale** : Techniques spécialisées
- **Cross-attention fusion** : Intégration intelligente des features
- **Multi-scale pooling** : Capture d'informations à différentes échelles

---

## 📁 Structure du Projet

```
ICEMS/
├── 📓 hybrid_model_clean.ipynb    # Notebook principal avec modèle hybride
├── 🐍 data_process.py             # Processeur de données chirurgicales
├── 🧪 test.py                     # Script de test et validation
├── ⚙️ setup_environment.py        # Configuration automatique Python
├── 🔧 setup_environment.ps1       # Configuration PowerShell
├── 📋 requirements.txt            # Dépendances Python
│
├── 📊 data/                       # Données chirurgicales
│   ├── Exvivo_trial_Participants(Sheet1).csv    # Métadonnées participants
│   ├── full_data.json                           # Données complètes JSON
│   ├── final_data_normalized_with_levels.pkl    # Données preprocessées
│   ├── filtered_data.json                       # Données filtrées
│   └── raw_data.json                           # Données brutes
│
├── 🎯 model_outputs/              # Résultats et modèles sauvegardés
│   ├── hybrid_surgical_best.keras              # Meilleur modèle
│   ├── results_final.json                      # Résultats détaillés
│   ├── scaler.pkl                             # Normalisateur
│   ├── performance_report.txt                  # Rapport de performance
│   └── *.png                                   # Visualisations
│
└── 🏥 surgical_expertise_env/     # Environnement virtuel Python
    ├── Scripts/                   # Exécutables (Windows)
    ├── Lib/site-packages/        # Packages installés
    └── pyvenv.cfg                 # Configuration environnement
```

---

## ⚡ Installation Rapide

### Prérequis
- **Python 3.10+**
- **Git**
- **8GB RAM minimum** (16GB recommandé)
- **GPU optionnel** (accélération CUDA)

### 1. Cloner le Projet
```bash
git clone https://github.com/D3MIA/ICEMS.git
cd ICEMS
```

### 2. Configuration Automatique (Recommandé)

#### 🐍 Configuration Python
```bash
python setup_environment.py
```

#### 💻 Configuration PowerShell (Windows)
```powershell
.\setup_environment.ps1
```

### 3. Activation de l'Environnement

#### Windows
```cmd
surgical_expertise_env\Scripts\activate
```

#### Linux/macOS
```bash
source surgical_expertise_env/bin/activate
```

### 4. Installation Manuelle (Alternative)
```bash
# Créer environnement virtuel
python -m venv surgical_expertise_env

# Activer environnement
# Windows: surgical_expertise_env\Scripts\activate
# Linux/macOS: source surgical_expertise_env/bin/activate

# Installer dépendances
pip install -r requirements.txt

# Configurer kernel Jupyter
python -m ipykernel install --user --name=surgical_expertise_env
```

---

## 🚀 Utilisation

### 1. Traitement des Données
```bash
# Convertir les données JSON en format PKL
python data_process.py
```

### 2. Entraînement du Modèle

#### Via Jupyter Notebook (Recommandé)
```bash
jupyter notebook
# Ouvrir hybrid_model_clean.ipynb
# Sélectionner le kernel 'surgical_expertise_env'
# Exécuter toutes les cellules
```

#### Via Script Python
```bash
python test.py
```

### 3. Évaluation et Résultats

Les résultats sont automatiquement sauvegardés dans `model_outputs/` :
- **Modèle entraîné** : `hybrid_surgical_best.keras`
- **Résultats JSON** : `results_final.json`
- **Rapport détaillé** : `performance_report.txt`
- **Visualisations** : `*.png`

---

## 📈 Performances du Modèle

### Métriques Principales
- **Architecture** : Hybrid Transformer+LSTM+GRU+CNN
- **Paramètres** : ~2M paramètres optimisés
- **Accuracy 9 classes** : Variable selon distribution des données
- **Accuracy 6 classes regroupées** : Amélioration significative
- **Temps d'entraînement** : ~5-15 minutes (GPU) / 30-60 minutes (CPU)

### Optimisations Spécialisées
- ✅ **Focal Loss** pour classes déséquilibrées
- ✅ **Poids adaptatifs** avec boost pour niveaux critiques
- ✅ **Augmentation de données** chirurgicale
- ✅ **Régularisation avancée** (L1/L2, Dropout, BatchNorm)
- ✅ **Callbacks intelligents** (EarlyStopping, ReduceLR)

### Classes Critiques Identifiées
- **PGY2-PGY4** : Niveaux intermédiaires nécessitant boost
- **Fellow spécialisés** : Regroupement optimisé
- **Regroupement 6 classes** : Amélioration de performance

---

## 🔧 Configuration Avancée

### Paramètres du Modèle
```python
# Paramètres modifiables dans hybrid_model_clean.ipynb
SEQUENCE_LENGTH = 50        # Longueur des séquences temporelles
BATCH_SIZE = 500           # Taille des lots d'entraînement
EPOCHS = 100               # Nombre d'époques maximum
LEARNING_RATE = 0.0008     # Taux d'apprentissage initial
DROPOUT_RATE = 0.1-0.4     # Taux de dropout par couche
```

### Techniques d'Augmentation
```python
# Augmentation spécialisée chirurgicale
- Bruit adaptatif basé expertise (1-3%)
- Décalage temporel (-3 à +4 timesteps)
- Mise à l'échelle (±5-10%)
- Masquage temporel partiel
```

### Équilibrage des Classes
```python
# Boost spécialisé pour classes critiques
ultra_critical_boost = {
    2: 2.0,  # PGY2
    3: 2.5,  # PGY3  
    4: 3.0,  # PGY4
    5: 2.0,  # PGY5
    6: 1.8   # PGY6
}
```

---

## 📊 Analyse des Données

### Source des Données
- **Origine** : Simulations neurochirurgicales ex-vivo
- **Participants** : 1000+ essais de chirurgiens à différents niveaux
- **Features** : Métriques temporelles de performance chirurgicale
- **Collecte** : 2022, essais contrôlés avec métadonnées

### Prétraitement
1. **Conversion JSON → PKL** via `data_process.py`
2. **Normalisation Z-score** des features temporelles
3. **Création de séquences** de longueur fixe (50 timesteps)
4. **Mapping d'expertise** vers 9 niveaux discrets
5. **Division stratifiée** : 60% train / 20% val / 20% test

### Distribution des Classes
- **Déséquilibre naturel** : Plus d'étudiants que de seniors
- **Classes rares** : Fellows spécialisés, PGY intermédiaires
- **Solutions** : Poids adaptatifs, Focal Loss, augmentation ciblée

---

## 🔍 Métriques et Évaluation

### Métriques de Classification
- **Accuracy globale** : Performance sur 9 classes
- **Accuracy par classe** : Détection des classes problématiques
- **Matrices de confusion** : Visualisation des erreurs
- **Rapport de classification** : Précision, Rappel, F1-score

### Métriques de Régression (Simulées)
- **R² Score** : Corrélation avec scores continus d'expertise
- **MAE** : Erreur absolue moyenne sur échelle 0-1
- **MSE** : Erreur quadratique moyenne

### Visualisations Automatiques
- 📊 **Matrices de confusion** (9 et 6 classes)
- 📈 **Courbes d'entraînement** (Loss, Accuracy, LR)
- 🎯 **Distribution des prédictions**
- 📋 **Rapport de performance détaillé**

---

## 🛠️ Dépendances Principales

### Deep Learning
- **TensorFlow 2.19.0** : Framework principal
- **NumPy ≥1.26.0** : Calculs numériques
- **Scikit-learn ≥1.7.0** : Métriques et preprocessing

### Visualisation
- **Matplotlib ≥3.10.0** : Graphiques de base
- **Seaborn ≥0.13.0** : Visualisations statistiques
- **Plotly ≥5.15.0** : Graphiques interactifs

### Environnement
- **Jupyter ≥1.0.0** : Notebooks interactifs
- **Pandas ≥2.0.0** : Manipulation de données
- **TensorBoard ≥2.13.0** : Monitoring d'entraînement

---

## 🚨 Résolution de Problèmes

### Erreurs Communes

#### 1. Erreur de GPU/CUDA
```bash
# Vérifier disponibilité GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Si pas de GPU, le modèle s'adapte automatiquement au CPU
```

#### 2. Problème de Mémoire
```python
# Réduire batch_size dans hybrid_model_clean.ipynb
train_config = {
    'batch_size': 256,  # Réduire de 500 à 256
    # ...
}
```

#### 3. Données Manquantes
```bash
# Vérifier présence des fichiers de données
ls data/
# Doit contenir: final_data_normalized_with_levels.pkl
```

#### 4. Erreur d'Environnement
```bash
# Recréer l'environnement
rm -rf surgical_expertise_env/
python setup_environment.py
```

### Optimisations de Performance

#### Accélération GPU
```python
# Configuration GPU optimale (automatique dans le code)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

#### Réduction Mémoire
- Réduire `batch_size` (500 → 256)
- Réduire `sequence_length` (50 → 30)
- Utiliser `mixed_precision` pour GPU modernes

---

## 📝 Contributions

### Comment Contribuer
1. **Fork** le projet
2. Créer une **branche feature** (`git checkout -b feature/AmazingFeature`)
3. **Commit** les changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une **Pull Request**

### Zones d'Amélioration
- 🔬 **Nouveaux datasets** : Intégration de données supplémentaires
- 🧠 **Architectures** : Test de nouveaux modèles (Vision Transformer, etc.)
- ⚖️ **Équilibrage** : Techniques avancées pour classes rares
- 🎯 **Métriques** : Métriques spécialisées pour domaine médical
- 🚀 **Déploiement** : API REST, interface web, intégration clinique

---


## 👥 Équipe de Développement

- **Recherche** : Équipe D3MIA
- **Développement** : Spécialistes en IA médicale
- **Validation** : Chirurgiens experts et résidents

---

## 📚 Références Scientifiques

### Articles Connexes
- Transformer architectures in medical AI
- LSTM for temporal surgical skill assessment
- Multi-modal fusion for expertise prediction
- Class imbalance in medical classification

### Technologies Utilisées
- **Transformer** : Architecture attention-based
- **LSTM/GRU** : Réseaux récurrents pour séquences
- **CNN 1D** : Convolution pour patterns temporels
- **Focal Loss** : Fonction de perte pour déséquilibre

---

## 🔗 Liens Utiles

- **Repository** : [GitHub ICEMS](https://github.com/D3MIA/ICEMS)
- **TensorFlow** : [Documentation officielle](https://tensorflow.org/)
- **Jupyter** : [Guide d'installation](https://jupyter.org/install)
- **Guides Deep Learning** : [Keras Documentation](https://keras.io/)

---

