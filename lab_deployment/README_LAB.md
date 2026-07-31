# 🧪 ICEMS — Quickstart Lab PC

> Imprime cette page et garde-la à côté du PC du labo.

---

## Installation (une seule fois sur le PC du labo)

```powershell
# 1. Aller dans le dossier
cd C:\ICEMS\lab_deployment

# 2. Créer environnement Python isolé (recommandé)
python -m venv venv
.\venv\Scripts\activate

# 3. Installer dépendances
pip install -r requirements.txt
```

Vérification rapide :

```powershell
python -c "import numpy, pandas, scipy, matplotlib; print('OK toutes deps installees')"
```

---

## Avant chaque session de collecte — Checklist 5 min

```text
[ ] PC alimenté, SpryTrack ouvert
[ ] Atracsys STK300 chaude (5 min sans capture)
[ ] Rigid body 4 sphères monté sur manche
[ ] Fond noir mat à 30-50 cm derrière la zone
[ ] Distance caméra ↔ centre = 80 cm (mesurée au mètre)
[ ] SpryTrack → Settings → Acquisition Rate = 120 Hz
[ ] Test statique 10 s → valid_ratio > 95 % (sinon STOP, debug matériel)
[ ] Dossier de session créé :
    C:\ICEMS\dataset\<YYYY-MM-DD>\generic\
```

---

## Workflow de session (3 commandes à connaître)

### 1. Après chaque rep — Contrôle qualité (GO / REDO)

```powershell
python src\process_rep.py "C:\ICEMS\dataset\2026-MM-DD\generic\translation_slow\R01"
```

→ Affiche `✓ GO` ou `✗ REDO` + raisons. **N'attendre que sur GO** pour passer à la rep suivante.

### 2. À la fin de la session — Traitement complet

```powershell
python src\process_session.py "C:\ICEMS\dataset\2026-MM-DD"
```

→ Convertit toutes les `*_Fiducial.csv` en `fiducials_clean.csv`, lance le tracking hongrois à 120 Hz, génère `features_6ch.npy` partout, écrit `global_report.csv`.

### 3. Bilan qualité de toute la session

```powershell
python src\check_quality.py --all "C:\ICEMS\dataset\2026-MM-DD\generic"
```

→ Liste de toutes les reps en GO / REDO, taux de réussite global, recommandations.

---

## Arborescence des dossiers après une session

```text
C:\ICEMS\dataset\2026-MM-DD\
└── generic\
    ├── translation_slow\
    │   ├── R01\
    │   │   ├── Export_xxx_Fiducial.csv    ← brut SpryTrack
    │   │   ├── fiducials_raw.csv          ← produit
    │   │   ├── fiducials_clean.csv        ← produit
    │   │   └── meta.txt                   ← produit
    │   ├── R02\
    │   └── ...
    ├── translation_med\
    └── ...

C:\ICEMS\pipeline_output_2026-MM-DD\        ← produit en fin de session
└── generic\
    ├── translation_slow\
    │   ├── R01\
    │   │   ├── features_6ch.npy           ← le fichier qui sera utilisé pour le MAE
    │   │   ├── tracks_hungarian.csv
    │   │   ├── R01_3d_comparison.png
    │   │   ├── R01_features.png
    │   │   ├── R01_jumps.png
    │   │   └── stats.json
    │   └── R02\
    │       └── ...
    └── global_report.csv                  ← bilan qualité de toute la session
```

---

## Critères GO automatiques (script process_rep.py)

| Critère | Seuil | Si KO |
|---|---|---|
| `n_frames` | ≥ 100 | Rep trop courte (< 1 s à 120 Hz) |
| `pct_valid` | ≥ 70 % | Occlusion excessive |
| `mean_n_fiducials_kept` | ≥ 3 | Sphères perdues — vérifier orientation |
| `fps_observed` | 96-144 Hz | SpryTrack n'a pas tenu la cadence |

---

## En cas de problème

### valid_ratio chronique < 70 % sur toutes les reps

1. **Vérifier la caméra** : refaire la calibration SpryTrack.
2. **Vérifier les sphères** : nettoyer (alcool sec), refixer si elles bougent.
3. **Vérifier l'éclairage** : éteindre les sources IR (lampes halogènes, lumière du soleil directe).
4. **Vérifier le fond** : retirer tout vêtement clair ou objet brillant dans le champ.

### fps_observed < 100 Hz alors que SpryTrack est configuré à 120

→ La machine n'arrive pas à suivre. Solutions :
- Fermer les autres applications (navigateur, Office, etc.).
- Réduire à **100 Hz** ou **60 Hz** (`Settings → Acquisition Rate`).
- Capturer sur un SSD au lieu d'un HDD.

### Échec du tracking sur une rep ("Aucune frame avec assez de sphères valides")

→ La rep est inutilisable. La supprimer et la refaire avec un setup plus rigide.

---

## Transfert vers le PC principal

À la fin de chaque journée de collecte :

```powershell
# Zipper la session brute + la sortie pipeline
Compress-Archive -Path "C:\ICEMS\dataset\2026-MM-DD","C:\ICEMS\pipeline_output_2026-MM-DD" `
                 -DestinationPath "C:\ICEMS\transfer_2026-MM-DD.zip"
```

Puis copier `transfer_2026-MM-DD.zip` sur clé USB ou via le réseau vers le PC principal.

---

## Contact

En cas de doute, consulter `COLLECTION_PROTOCOL.md` (document détaillé).
