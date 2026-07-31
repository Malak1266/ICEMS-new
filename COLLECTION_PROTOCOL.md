# ICEMS — Protocole de collecte Atracsys v2

> **Statut** : Approche révisée post-réunion 2026-05 (1 rigid body générique, MAE focus).
> **Version** : 1.0 — 2026-05-25
> **Objectif** : 25 000+ fenêtres pour pré-entraîner le MAE sur la dynamique du geste.

---

## TL;DR — Avant chaque session

```text
[ ] Caméra Atracsys STK300 chaude (allumée depuis 5 min)
[ ] Rigid body 4 sphères monté sur manche allongé (≥ 15 cm)
[ ] Fond noir mat installé derrière la zone de capture
[ ] SpryTrack configuré à 120 Hz
[ ] Test de calibration : rigid body statique au centre → valid_ratio > 95 %
[ ] Catalogue de mouvements imprimé / sur tablette
[ ] Dossier de session créé : pipeline_input/<YYYY-MM-DD>/generic/
```

---

## 1. Préparation matérielle

### 1.1 Choix du rigid body

| Disponible | À utiliser ? | Raison |
|---|---|---|
| Corps 3 sphères | ❌ | Minimum géométrique, ne survit pas à 1 occlusion |
| **Corps 4 sphères** | ✅ **OUI — par défaut** | Compromis optimal poids/robustesse |
| Corps 5 sphères | ⚠️ Backup uniquement | Auto-occlusion fréquente |

**Décision** : faire toutes les sessions avec le corps 4 sphères. Si besoin de comparaison, faire 5-10 reps avec le 5 sphères en fin de session pour validation.

### 1.2 Manche allongé (élément clé)

**Pourquoi** : ta main occulte les sphères à 80 cm de la caméra → cause principale du faible `valid_ratio` historique (~20 %).

**Solution** :
- Tube PVC noir ou bois peint en noir mat, **longueur 15-30 cm**.
- Le rigid body se fixe au bout du manche (vis ou collier).
- Ta main tient le manche **au milieu/arrière**, jamais près des sphères.
- Le manche doit être suffisamment rigide pour ne pas vibrer → influence le jerk.

### 1.3 Fond noir mat

**Pourquoi** : les vêtements clairs et les reflets génèrent des fiducials parasites qui dégradent l'association hongroise.

**Solution** :
- Tissu velours noir ou carton mat noir derrière la zone de capture.
- Surface ≥ 1.5 × 1.5 m, à 30-50 cm derrière le centre de la zone.
- Éviter tout objet réfléchissant dans le champ (montre, bague, surface laquée).

### 1.4 Tenue de l'opérateur

- T-shirt ou pull **noir** (ou couleur sombre mate).
- Pas de bracelet, montre brillante, alliance.
- Cheveux attachés si longs (évite les reflets).

---

## 2. Configuration SpryTrack à 120 Hz

> **Cible** : 120 Hz constants, déviation < 5 %.

### 2.1 Étapes dans SpryTrack GUI

1. Ouvrir SpryTrack.
2. **File → Connect** → sélectionner la caméra STK300.
3. **Settings → Acquisition** :
   - Frame Rate Target : `120 Hz`
   - Integration Time : `auto` (laisser SpryTrack ajuster selon l'illumination)
   - LED Power : `high` (meilleure visibilité des sphères passives)
4. **Settings → Tracking** :
   - Tracking Method : `Fiducial Detection`
   - Probability Threshold : `0.7` (plus permissif que les 0.9 actuels, on filtre ensuite côté Python)
5. **Settings → Recording** :
   - Format : `CSV` (export `*_Fiducial.csv`)
   - Output Directory : `~/dataset/<YYYY-MM-DD>/generic/<motion>/<rep>/`

### 2.2 Vérification de la fréquence effective

Après une capture test de 10 s :

```powershell
python -c "
import pandas as pd
df = pd.read_csv('chemin/vers/Export_xxx_Fiducial.csv', sep=';')
t = df['Timestamp'].apply(lambda x: int(x, 16))
n_frames = df['Timestamp'].nunique()
duration_s = (t.max() - t.min()) / 1e9
print(f'fps_observed = {n_frames / duration_s:.1f} Hz')
"
```

**Critère d'acceptation** : `fps_observed` entre **115 et 125 Hz**. Si en-dessous de 100 Hz, la machine n'arrive pas à suivre — réduire à 100 Hz ou 60 Hz.

---

## 3. Catalogue de mouvements

> **Total cible** : ~400 reps × 4-8 s = ~30-50 min de capture utile = ~25 000 fenêtres après extraction.

Chaque mouvement est nommé exactement comme indiqué (utilisé dans l'arborescence des dossiers).

### 3.1 Catégorie A — Cinétique pure (≈ 80 reps)

| Nom (dossier) | Description | Reps | Durée |
|---|---|---|---|
| `translation_slow` | A→B sur 30 cm, vitesse lente régulière | 20 | 5 s |
| `translation_med` | A→B sur 30 cm, vitesse moyenne | 20 | 3 s |
| `translation_fast` | A→B sur 30 cm, le plus vite possible | 20 | 1.5 s |
| `rotation_slow` | Rotation 360° autour axe long, lente | 10 | 5 s |
| `rotation_fast` | Rotation 360° autour axe long, rapide | 10 | 2 s |

### 3.2 Catégorie B — Trajectoires structurées (≈ 60 reps)

| Nom | Description | Reps | Durée |
|---|---|---|---|
| `figure_8` | Forme de 8 horizontale | 15 | 4 s |
| `circle` | Cercle horizontal de 20 cm de diamètre | 15 | 3 s |
| `zigzag` | Aller-retour serpentin (5 oscillations) | 15 | 5 s |
| `helix` | Translation 30 cm + rotation 360° simultanées | 15 | 6 s |

### 3.3 Catégorie C — Gestes chirurgicaux simulés (≈ 100 reps — **prioritaire**)

| Nom | Description | Reps | Durée |
|---|---|---|---|
| `precision_grasp` | Approche lente d'un point précis (~1 mm) | 25 | 3 s |
| `suture_pattern` | Mouvement de suture (4-5 passes) | 20 | 8 s |
| `dissection_sweep` | Balayage horizontal contrôlé | 25 | 4 s |
| `retraction_hold` | Position tenue + micro-ajustements | 15 | 10 s |
| `instrument_swap` | Pose-reprise rapide (simule changement d'outil) | 15 | 3 s |

### 3.4 Catégorie D — Modulation d'expertise (≈ 80 reps — **clé pour le fine-tuning**)

| Nom | Description | Reps | Durée |
|---|---|---|---|
| `expert_fluid` | Trajectoire fluide, sans correction (imite staff) | 20 | 5 s |
| `expert_decisive` | Mouvement unique sans hésitation | 20 | 4 s |
| `novice_hesitant` | Pauses, redémarrages, micro-tremblements volontaires | 20 | 5 s |
| `novice_overshoot` | Dépassement cible puis correction | 20 | 4 s |

### 3.5 Catégorie E — Conditions limites (≈ 40 reps)

| Nom | Description | Reps | Durée |
|---|---|---|---|
| `static_steady` | Immobile, tenu fermement | 10 | 30 s |
| `static_tremor` | Immobile mais petits tremblements volontaires | 10 | 30 s |
| `out_of_range` | Sort/rentre dans le volume de tracking | 10 | 10 s |
| `mixed_freestyle` | Geste libre improvisé (diversité maximale) | 10 | 15 s |

**Total** : ≈ 360 reps en 5 catégories.

---

## 4. Procédure par session

### 4.1 Démarrage (15 min)

1. **Chauffer la caméra** : SpryTrack ouvert, capture vide pendant 5 min.
2. **Calibration** : poser le rigid body **statique au centre** de la zone de capture (à 80 cm de la caméra).
3. **Vérifier valid_ratio à vide** : enregistrer 10 s statique.
   - Attendu : **> 95 % valides**, **4 fiducials détectés**, **jump_max < 2 mm** (sphères qui ne bougent pas).
   - Si valid_ratio < 90 % statique : problème matériel (ajustement caméra, manche, fond).
4. **Définir 3 zones de profondeur** : Z1 = 60 cm, Z2 = 80 cm, Z3 = 100 cm de la caméra. Marquer au sol.

### 4.2 Pendant la collecte (1.5 - 2 h par session)

Pour **chaque mouvement** du catalogue :

```text
1. Annoncer à voix haute le nom + numéro de rep (s'enregistre dans la métadonnée audio si activée)
2. Positionner rigid body à la zone (Z1, Z2 ou Z3)
3. SpryTrack → bouton Record (ou raccourci F5)
4. Exécuter le mouvement pendant la durée prévue
5. SpryTrack → bouton Stop (F5)
6. Vérifier le pct_valid affiché (cf. 4.3)
7. Si pct_valid > 70 % : passer à la rep suivante (pause 2-3 s)
   Si pct_valid < 70 % : re-faire la rep immédiatement
```

### 4.3 Contrôle qualité en direct

Après chaque rep, lancer :

```powershell
python src/check_quality.py "C:\Users\malek\Downloads\dataset\YYYY-MM-DD\generic\translation_slow\R01"
```

Le script affiche :
- `pct_valid` : devrait être > 70 %
- `n_frames_total` : devrait correspondre à ~ duration_s × 120
- `mean_n_fiducials_kept` : devrait être proche de 4
- Verdict GO / NO-GO

### 4.4 Pause / fin de session

- Pause toutes les 45 min (fatigue → tremblements involontaires qui biaisent).
- Boire de l'eau (déshydratation → tremblements amplifiés).
- En fin de session : **bilan global** via `python src/check_quality.py --all "C:\Users\malek\Downloads\dataset\YYYY-MM-DD\generic"`.

---

## 5. Structure des dossiers attendus

```
dataset/
└── 2026-MM-DD/
    └── generic/                          ← un seul "instrument" (post-réunion)
        ├── translation_slow/
        │   ├── R01/
        │   │   ├── Export_xxx_Fiducial.csv     ← brut SpryTrack
        │   │   ├── fiducials_raw.csv           ← post convert_srytrack.py
        │   │   ├── fiducials_clean.csv         ← post convert_srytrack.py
        │   │   └── meta.txt
        │   ├── R02/
        │   └── ...
        ├── translation_med/
        ├── figure_8/
        ├── suture_pattern/
        └── ...
```

---

## 6. Post-traitement (après chaque session)

### 6.1 Conversion SpryTrack → fiducials

```powershell
cd C:\Users\malek\Downloads\ICEMS-main (6)\ICEMS-main
$env:PYTHONIOENCODING="utf-8"
python src\convert_srytrack.py --all "C:\Users\malek\Downloads\dataset\2026-MM-DD"
```

### 6.2 Tracking hongrois + features

```powershell
python src\tracking_hungarian.py --all "C:\Users\malek\Downloads\dataset\2026-MM-DD" --out "pipeline_output_2026-MM-DD" --fs 120
```

**Important** : `--fs 120` est crucial. Sans ça, les calculs cinématiques seront faussés d'un facteur 9× (120/14 ≈ 8.6).

### 6.3 Vérification du `global_report.csv`

Critères de qualité acceptables pour une session :

| Indicateur | Cible | Si en-dessous |
|---|---|---|
| `pct_valid` moyenne | > 80 % | revoir setup matériel |
| `jump_mean_mm` | < 8 mm | vérifier `--fs` et `MAX_JUMP` |
| `jump_p95_mm` | < 20 mm | filtrer les reps aberrantes |
| Nb reps avec pct_valid > 70 % | > 80 % du total | re-faire les mauvaises |

### 6.4 Construction du dataset MAE

Une fois plusieurs sessions accumulées :

```powershell
python src\build_mae_dataset_enrichi.py --base "C:\Users\malek\Downloads\dataset" --out "data/atracsys_mae"
```

→ Produit `X_pretrain_4ch.npy` (ou 6ch si on décide de conserver `spread` + `axis_angle`).

---

## 7. Checklist de validation finale

Avant de considérer la phase de collecte terminée :

- [ ] **Volume** : ≥ 25 000 fenêtres après extraction (cible 50 000)
- [ ] **Diversité motion** : ≥ 15 catégories de mouvements représentées
- [ ] **Diversité profondeur** : Z1, Z2, Z3 chacun représentés
- [ ] **Qualité** : `pct_valid` moyen ≥ 80 % sur tout le corpus
- [ ] **Cohérence temporelle** : `fps_observed` ≈ 120 sur 95 % des reps
- [ ] **Validité physique** : aucune fenêtre avec velocity > 2 m/s (saut probablement artéfactuel)

---

## 8. Risques connus et mitigations

| Risque | Probabilité | Mitigation |
|---|---|---|
| Fatigue → biais de tremblement | Élevée (4h/jour) | Pauses 5 min toutes les 45 min |
| Biais d'un seul opérateur | Élevée (solo) | Variabilité intra-opérateur (jours différents, énergie variable) |
| Sphères qui se décollent | Modérée | Vérification visuelle avant chaque session |
| Reflets parasites (vêtements clairs) | Modérée | Tenue sombre stricte |
| Drift caméra (thermal) | Faible | Chauffer 5 min ; re-calibration à mi-session |
| Saturation disque (CSV volumineux à 120 Hz) | Modérée | Surveiller `df -h` ; chaque rep ~ 200-500 KB |

---

## 9. Journal de session (template)

À remplir avant chaque session, à archiver dans `dataset/YYYY-MM-DD/session_log.md` :

```markdown
# Session YYYY-MM-DD

- **Heure début** : HH:MM
- **Heure fin** : HH:MM
- **Opérateur** : Malek
- **Rigid body** : 4 sphères (modèle X)
- **Fréquence SpryTrack** : 120 Hz
- **Distance caméra** : 80 cm
- **Éclairage** : néon plafond + lampe d'appoint
- **État physique** : reposé / fatigué / café (entoure)

## Mouvements enregistrés
- translation_slow : R01-R20 ✅
- translation_med  : R01-R15 (R10 manquée, refaire)
- ...

## Notes / anomalies
- Vers 14h32 : interruption courte (caméra a perdu signal 2 sec)
- valid_ratio plus bas que d'habitude sur figure_8 → revérifier le bras
```

---

## 10. Évolution du protocole

Ce document est **vivant**. Après chaque session, on note dans `PROJECT_PROGRESSION.md` les ajustements à apporter :
- Catégories de mouvement à ajouter / retirer
- Paramètres SpryTrack à ajuster
- Problèmes matériels rencontrés

**Prochaine révision prévue** : après la première session (sera v1.1 avec corrections terrain).
