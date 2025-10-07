# 🏥 Surgical Instrument Tracking Analysis Tool

## 📖 Description

Outil d'analyse avancé pour évaluer la qualité du tracking des instruments chirurgicaux dans des données de simulation neurochirurgicale. Le script analyse les données de détection d'instruments et génère des rapports complets sur les performances de tracking avec visualisations et identification des cas problématiques.

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python)](https://python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Latest-150458?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Latest-11557c?style=flat-square)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
## 🎯 Fonctionnalités Principales

### ✨ Analyses Effectuées
- **📊 Statistiques descriptives** : Moyennes, médianes, écarts-types pour tous les champs
- **📈 Visualisations** : Histogrammes et distributions pour chaque métrique
- **👥 Analyse par participant** : Performance moyenne par participant
- **🔧 Analyse par instrument** : Performance par type d'instrument (grasper, scissors, drill)
- **🚨 Détection des cas problématiques** : Identification automatique des cas avec fraction < 0.5
- **🏆 Classement des pires cas** : Top des combinaisons participant-trial-instrument les plus problématiques
- **💾 Export de données** : Sauvegarde en CSV et rapports textuels

### 📋 Métriques Analysées
- **`captured_time`** : Temps total de capture de l'instrument (secondes)
- **`inuse_time`** : Temps total d'utilisation de l'instrument (secondes)  
- **`captured_frames`** : Nombre de frames où l'instrument a été détecté
- **`inuse_frames`** : Nombre total de frames où l'instrument était en cours d'utilisation
- **`fraction`** : Ratio de détection (captured_time/inuse_time ou captured_frames/inuse_frames)

---

## 📁 Structure du Projet

```
📦 Surgical Analysis Tool
├── 📄 analyze_missing_data.py      # Script principal d'analyse
├── 📄 requirements.txt             # Dépendances Python
├── 📄 setup_environment.ps1        # Script PowerShell de configuration
├── 📄 setup_environment.py         # Script Python de configuration
├── 📄 test.py                      # Script de test
├── 📁 data/                        # Dossier des données (exclu de Git)
│   ├── 📄 missing_data.json        # Fichier de données principal
│   ├── 📄 *.csv                    # Autres fichiers de données
│   └── 📄 *.pkl                    # Fichiers de données sérialisées
├── 📁 surgical_expertise_env/      # Environnement virtuel Python (exclu de Git)
└── 📁 analyse_missing_data_result/ # Résultats d'analyse (exclu de Git)
    ├── 📄 *.png                    # Graphiques générés
    ├── 📄 *.csv                    # Données exportées
    └── 📄 *.txt                    # Rapports textuels
```

---

## 🚀 Installation et Configuration

### 1️⃣ Prérequis
- **Python 3.8+** installé sur votre système
- **PowerShell** (pour Windows) ou **Terminal** (pour macOS/Linux)
- **Git** (optionnel, pour le clonage du repository)

### 2️⃣ Cloner le Projet
```bash
git clone https://github.com/D3MIA/ICEMS.git
cd ICEMS
```

### 3️⃣ Configuration Automatique (Recommandée)

#### Windows (PowerShell)
```powershell
# Exécuter le script de configuration automatique
.\setup_environment.ps1
```

#### Linux/macOS
```bash
# Exécuter le script Python de configuration
python setup_environment.py
```

### 4️⃣ Configuration Manuelle (Alternative)

#### Étape 1 : Créer l'environnement virtuel
```bash
python -m venv surgical_expertise_env
```

#### Étape 2 : Activer l'environnement
```bash
# Windows
surgical_expertise_env\Scripts\activate

# Linux/macOS  
source surgical_expertise_env/bin/activate
```

#### Étape 3 : Installer les dépendances
```bash
pip install -r requirements.txt
```

---

## 📊 Format des Données d'Entrée

### 🔗 Format JSON Attendu

Le script attend un fichier JSON avec la structure suivante :

```json
{
  "participant": {
    "0": "1020614",
    "1": "1020614", 
    "2": "1020614"
  },
  "trial": {
    "0": "Trial1",
    "1": "Trial1",
    "2": "Trial1"
  },
  "instrument": {
    "0": "scissors",
    "1": "grasper", 
    "2": "drill"
  },
  "captured_time": {
    "0": 24.749,
    "1": 45.231,
    "2": null
  },
  "inuse_time": {
    "0": 67.04,
    "1": 78.45,
    "2": 120.5
  },
  "captured_frames": {
    "0": 232.0,
    "1": 425.0,
    "2": null
  },
  "inuse_frames": {
    "0": 631.0,
    "1": 738.0,
    "2": 1134.0
  },
  "fraction": {
    "0": 0.3691676611,
    "1": 0.5764332247,
    "2": null
  }
}
```

### 📝 Description des Champs

| Champ | Type | Description | Obligatoire |
|-------|------|-------------|-------------|
| `participant` | string | ID unique du participant | ✅ Oui |
| `trial` | string | Nom/numéro du trial (ex: "Trial1", "Trial2") | ✅ Oui |
| `instrument` | string | Type d'instrument ("scissors", "grasper", "drill") | ✅ Oui |
| `captured_time` | float/null | Temps de capture en secondes | ❌ Non |
| `inuse_time` | float/null | Temps d'utilisation en secondes | ❌ Non |
| `captured_frames` | float/null | Nombre de frames capturées | ❌ Non |
| `inuse_frames` | float/null | Nombre de frames d'utilisation | ❌ Non |
| `fraction` | float/null | Ratio de détection (0-1) | ❌ Non |

### ⚠️ Notes Importantes sur les Données
- Les valeurs `null` sont automatiquement converties en `NaN`
- Le script **ne calcule PAS** les fractions manquantes automatiquement
- Les données manquantes sont préservées comme `NaN` pour maintenir l'authenticité
- Format d'export pandas DataFrame vers JSON supporté

---

## 🎮 Utilisation

### 🏃‍♂️ Lancement Rapide

1. **Activer l'environnement virtuel :**
```bash
# Windows
surgical_expertise_env\Scripts\activate

# Linux/macOS
source surgical_expertise_env/bin/activate
```

2. **Exécuter l'analyse :**
```bash
python analyze_missing_data.py --input data/missing_data.json --out_dir ./analyse_missing_data_result
```

### 🔧 Options de la Ligne de Commande

```bash
python analyze_missing_data.py [OPTIONS]

Options:
  --input PATH     Chemin vers le fichier JSON d'entrée 
                   (défaut: data/missing_data.json)
  
  --out_dir PATH   Dossier de sortie pour les résultats
                   (défaut: ./analyse_missing_data_result)
  
  --help          Afficher l'aide
```

### 📋 Exemples d'Utilisation

#### Exemple 1 : Analyse Standard
```bash
python analyze_missing_data.py --input data/missing_data.json --out_dir ./results
```

#### Exemple 2 : Fichier Personnalisé
```bash
python analyze_missing_data.py --input /path/to/my_data.json --out_dir ./custom_analysis
```

#### Exemple 3 : Analyse avec Dossier Absolu
```bash
python analyze_missing_data.py --input "C:\Data\surgical_data.json" --out_dir "C:\Results\Analysis"
```

---

## 📊 Résultats Générés

### 📁 Structure des Résultats

```
📁 analyse_missing_data_result/
├── 📊 missing_data_analysis_report.txt           # Rapport textuel complet
├── 📊 worst_tracking_combinations.txt            # Rapport des pires cas
├── 📄 worst_tracking_combinations.csv           # Export CSV des pires cas
├── 📄 raw_data_export.csv                       # Export des données brutes
├── 📈 fraction_distribution.png                 # Histogramme des fractions
├── 📈 participant_analysis.png                  # Analyse par participant
└── 📈 instrument_analysis.png                   # Analyse par instrument
```

### 📋 Contenu des Rapports

#### 1️⃣ Rapport Textuel Principal
- 📊 **Statistiques globales** : Nombre d'entrées, participants, trials, instruments
- 📈 **Statistiques descriptives** : Moyennes, médianes, écarts-types
- 👥 **Analyse par participant** : Performance moyenne par participant
- 🔧 **Analyse par instrument** : Performance par type d'instrument
- 🚨 **Cas problématiques** : Liste des cas avec fraction < 0.5

#### 2️⃣ Rapport des Pires Cas
- 🏆 **Classement descendant** : Combinaisons participant-trial-instrument les plus problématiques
- 📊 **Métriques détaillées** : Toutes les valeurs pour chaque cas
- 🚨 **Identification critique** : Focus sur les cas avec fraction = 0.0

#### 3️⃣ Visualisations
- **Distribution des fractions** : Histogramme montrant la répartition des performances
- **Analyse par participant** : Graphique en barres des performances moyennes
- **Analyse par instrument** : Comparaison des performances par type d'instrument

### 📊 Métriques de Performance

#### Seuils d'Évaluation
- **🛠️ Excellent** : fraction ≥ 0.8 (tracking très fiable)
- **🟡 Bon** : 0.5 ≤ fraction < 0.8 (tracking acceptable)  
- **🔴 Problématique** : fraction < 0.5 (tracking insuffisant)
- **❌ Critique** : fraction = 0.0 (aucune détection)

---

## 🐛 Dépannage et FAQ

### ❓ Problèmes Courants

#### 1. Erreur "FileNotFoundError"
```
❌ Erreur: FileNotFoundError: [Errno 2] No such file or directory: 'data/missing_data.json'
```
**Solution :** Vérifiez que le fichier existe et que le chemin est correct.

#### 2. Erreur "ModuleNotFoundError" 
```
❌ Erreur: ModuleNotFoundError: No module named 'pandas'
```
**Solution :** Activez l'environnement virtuel et installez les dépendances :
```bash
surgical_expertise_env\Scripts\activate
pip install -r requirements.txt
```

#### 3. Erreur de Format JSON
```
❌ Erreur: JSONDecodeError: Expecting ',' delimiter
```
**Solution :** Vérifiez la structure JSON avec un validateur en ligne.

#### 4. Données Vides
```
⚠️ Aucune donnée valide trouvée après parsing
```
**Solution :** Vérifiez que votre JSON contient les champs `participant`, `trial`, et `instrument`.

### 🔧 Options de Debug

Pour activer les logs détaillés, modifiez la ligne suivante dans le script :
```python
# Ligne ~70 dans analyze_missing_data.py
DEBUG = True  # Changer False en True
```

---

## 🔬 Détails Techniques

### 🏗️ Architecture du Code

```python
# Structure principale du script
AdvancedSurgicalDataProcessor
├── load_missing_data()           # Chargement des données JSON
├── parse_entry()                 # Parsing et validation d'une entrée
├── safe_float() / safe_int()     # Conversion sécurisée des types
├── plot_fraction_distribution()  # Génération des graphiques
├── analyze_participants()        # Analyse par participant
├── analyze_instruments()         # Analyse par instrument
└── generate_worst_cases_report() # Génération du classement
```

### 🔄 Flux de Traitement

1. **📂 Chargement** : Lecture et parsing du JSON
2. **✅ Validation** : Vérification de la structure des données
3. **🔧 Parsing** : Extraction et conversion des valeurs
4. **📊 Analysis** : Calcul des statistiques et métriques
5. **📈 Visualisation** : Génération des graphiques
6. **📝 Rapport** : Création des rapports textuels
7. **💾 Export** : Sauvegarde des résultats

### ⚡ Optimisations

- **🚀 Gestion mémoire** : Traitement par chunks pour gros datasets
- **🛡️ Robustesse** : Gestion d'erreurs complète avec try/catch
- **📊 Performance** : Utilisation de NumPy pour les calculs vectorisés
- **🎨 Visualisations** : Matplotlib optimisé avec styles personnalisés

---

## 📝 Contributions et Support

### 🤝 Comment Contribuer
1. **Fork** le repository
2. **Créez** une branche pour votre fonctionnalité (`git checkout -b feature/ma-fonctionnalite`)
3. **Commitez** vos changements (`git commit -am 'Ajout de ma fonctionnalité'`)
4. **Pushez** vers la branche (`git push origin feature/ma-fonctionnalite`)
5. **Créez** une Pull Request

### 📋 Standards de Code
- **🐍 PEP 8** : Style de code Python
- **📝 Documentation** : Docstrings pour toutes les fonctions
- **🧪 Tests** : Tests unitaires pour les nouvelles fonctionnalités
- **🔧 Type Hints** : Annotations de type recommandées

### 📞 Support

Pour signaler un bug ou demander une fonctionnalité :
1. Vérifiez la section Dépannage ci-dessus
2. Consultez les logs d'erreur complets
3. Préparez un exemple de données qui pose problème
4. Créez une issue sur le repository Git

---

## 📜 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

*🏥 Développé pour l'analyse de simulation neurochirurgicale - Université/Institution*

