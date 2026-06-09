#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse des données manquantes (missing_data) pour évaluer la qualité de détection des instruments.

Les données contiennent 5 champs par participant/trial/instrument :
- Captured_time : temps total où l'instrument est tracké
- Inuse_time : temps total où l'instrument a été utilisé  
- Captured_frames : nombre de frames capturées
- Inuse_frames : nombre de frames où l'instrument a été utilisé
- Fraction : pourcentage où l'instrument est tracké quand utilisé (captured_time / inuse_time)

Une Fraction élevée indique une bonne détection.
Une Fraction faible peut indiquer des données insuffisantes.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

# Configuration des graphiques
plt.style.use('default')
sns.set_palette("husl")

def load_missing_data(file_path: str) -> pd.DataFrame:
    """
    Charge les données missing_data depuis un fichier JSON.
    
    Args:
        file_path: Chemin vers le fichier JSON
        
    Returns:
        DataFrame avec les colonnes: participant, trial, instrument, captured_time, 
        inuse_time, captured_frames, inuse_frames, fraction
    """
    print(f"📁 Chargement des données: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"🔍 Type de données chargées: {type(data)}")
    if isinstance(data, list) and len(data) > 0:
        print(f"🔍 Premier élément: {type(data[0])}")
        print(f"🔍 Contenu du premier élément: {str(data[0])[:200]}...")
    elif isinstance(data, dict):
        print(f"🔍 Clés disponibles: {list(data.keys())}")
    
    # Gérer différents formats de données
    rows = []
    
    if isinstance(data, dict):
        # Debug: voir le type de chaque valeur
        print("🔍 Types des valeurs dans le dictionnaire:")
        for key, value in data.items():
            print(f"   {key}: {type(value)} (longueur: {len(value) if hasattr(value, '__len__') else 'N/A'})")
            if hasattr(value, '__len__') and len(value) > 0:
                print(f"      Premiers éléments: {value[:3] if isinstance(value, (list, tuple)) else 'N/A'}")
        
        # Vérifier si c'est un dictionnaire avec des dictionnaires indexés comme valeurs (format DataFrame)
        if all(isinstance(v, dict) for v in data.values()):
            print("🔍 Format détecté: dictionnaire avec des dictionnaires indexés (format pandas/DataFrame)")
            # Les données sont organisées par colonnes, chaque colonne est un dict avec des index
            keys = list(data.keys())
            lengths = [len(data[k]) for k in keys]
            print(f"🔍 Longueurs par colonne: {dict(zip(keys, lengths))}")
            
            if len(set(lengths)) == 1:  # Tous les dictionnaires ont la même longueur
                num_entries = lengths[0]
                print(f"🔍 Traitement de {num_entries} entrées depuis le format DataFrame")
                
                # Obtenir les index (probablement '0', '1', '2', etc.)
                first_column = list(data.values())[0]
                indices = list(first_column.keys())
                print(f"🔍 Premiers indices: {indices[:5]}...")
                
                for i, idx in enumerate(indices):
                    # Créer une entrée en combinant les valeurs pour cet index
                    entry = {}
                    for col_name in keys:
                        if idx in data[col_name]:
                            entry[col_name] = data[col_name][idx]
                        else:
                            entry[col_name] = None
                    
                    if i < 3:  # Debug pour les 3 premières entrées
                        print(f"🔍 Entrée brute {i} (index {idx}): {entry}")
                    rows.append(parse_entry(entry, debug=(i < 3)))
            else:
                print(f"⚠️ Longueurs des dictionnaires différentes: {dict(zip(keys, lengths))}")
                # Prendre la longueur minimale et traiter
                min_length = min(lengths)
                print(f"🔍 Utilisation de la longueur minimale: {min_length}")
                # Obtenir les index du dictionnaire le plus court
                shortest_col = min(data.values(), key=len)
                indices = list(shortest_col.keys())
                
                for i, idx in enumerate(indices):
                    entry = {}
                    for col_name in keys:
                        if idx in data[col_name]:
                            entry[col_name] = data[col_name][idx]
                        else:
                            entry[col_name] = None
                    
                    if i < 3:  # Debug pour les 3 premières entrées
                        print(f"🔍 Entrée brute {i} (index {idx}): {entry}")
                    rows.append(parse_entry(entry, debug=(i < 3)))
        # Vérifier si c'est un dictionnaire avec des arrays/listes comme valeurs
        elif all(isinstance(v, (list, tuple)) for v in data.values()):
            print("🔍 Format détecté: dictionnaire avec des listes comme valeurs")
            # Les données sont organisées par colonnes
            # Créer une liste d'entrées en combinant les valeurs par index
            keys = list(data.keys())
            lengths = [len(data[k]) for k in keys]
            print(f"🔍 Longueurs par colonne: {dict(zip(keys, lengths))}")
            
            if len(set(lengths)) == 1:  # Toutes les listes ont la même longueur
                num_entries = lengths[0]
                print(f"🔍 Traitement de {num_entries} entrées depuis le format colonnes")
                for i in range(num_entries):
                    entry = {k: data[k][i] for k in keys}
                    if i < 3:  # Debug pour les 3 premières entrées
                        print(f"🔍 Entrée brute {i}: {entry}")
                    rows.append(parse_entry(entry, debug=(i < 3)))
            else:
                print(f"⚠️ Longueurs des listes différentes: {dict(zip(keys, lengths))}")
                # Prendre la longueur minimale
                min_length = min(lengths)
                print(f"🔍 Utilisation de la longueur minimale: {min_length}")
                for i in range(min_length):
                    entry = {k: data[k][i] for k in keys}
                    if i < 3:  # Debug pour les 3 premières entrées
                        print(f"🔍 Entrée brute {i}: {entry}")
                    rows.append(parse_entry(entry, debug=(i < 3)))
        # Si ce n'est pas un dictionnaire de listes, vérifier d'autres formats
        elif all(isinstance(v, (str, int, float)) for v in data.values()):
            print("🔍 Format détecté: dictionnaire simple (une seule entrée)")
            # C'est probablement une seule entrée avec tous les champs
            rows.append(parse_entry(data, debug=True))
        elif 'data' in data:
            print("🔍 Format détecté: dictionnaire avec clé 'data'")
            data = data['data']
        elif len(data.keys()) == 1:
            print("🔍 Format détecté: dictionnaire avec une seule clé")
            # Si une seule clé, prendre sa valeur
            data = list(data.values())[0]
        else:
            print("🔍 Format détecté: dictionnaire avec clés comme entrées")
            # Traiter chaque clé comme une entrée
            for key, value in data.items():
                if isinstance(value, dict):
                    entry = value.copy()
                    if 'participant' not in entry:
                        entry['participant'] = key
                    rows.append(parse_entry(entry, debug=(len(rows) < 3)))
    
    if isinstance(data, list):
        for i, entry in enumerate(data):
            try:
                if isinstance(entry, dict):
                    rows.append(parse_entry(entry, debug=(i < 3)))
                elif isinstance(entry, str):
                    # Essayer de parser la chaîne comme JSON
                    try:
                        parsed_entry = json.loads(entry)
                        rows.append(parse_entry(parsed_entry, debug=(i < 3)))
                    except json.JSONDecodeError:
                        print(f"⚠️ Impossible de parser l'entrée {i}: {entry[:100]}...")
                        continue
                else:
                    print(f"⚠️ Type d'entrée non supporté à l'index {i}: {type(entry)}")
                    continue
            except Exception as e:
                print(f"⚠️ Erreur lors du traitement de l'entrée {i}: {e}")
                continue
    
    if not rows:
        raise ValueError("Aucune donnée valide trouvée dans le fichier JSON")
    
    df = pd.DataFrame(rows)
    print(f"✅ {len(df)} entrées chargées")
    print(f"📊 Participants: {df['participant'].nunique()}, Trials: {df['trial'].nunique()}, Instruments: {df['instrument'].nunique()}")
    
    return df

def parse_entry(entry: dict, debug: bool = False) -> dict:
    """
    Parse une entrée individuelle et extrait les champs requis.
    """
    # Extraire les valeurs avec différentes variantes de noms, gérer None et NaN
    def safe_float(value, default=0.0):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def safe_int(value, default=0):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        try:
            return int(float(value))  # float() d'abord au cas où c'est une string
        except (ValueError, TypeError):
            return default

    captured_time = safe_float(entry.get('captured_time', entry.get('Captured_time', 0)))
    inuse_time = safe_float(entry.get('inuse_time', entry.get('Inuse_time', 0)))
    captured_frames = safe_int(entry.get('captured_frames', entry.get('Captured_frames', 0)))
    inuse_frames = safe_int(entry.get('inuse_frames', entry.get('Inuse_frames', 0)))
    
    # Extraire la fraction sans calcul - mettre NaN si elle n'existe pas
    fraction_raw = entry.get('fraction', entry.get('Fraction', None))
    # Gérer les valeurs None, NaN, ou invalides
    if fraction_raw is None:
        fraction = np.nan
        if debug:
            print(f"⚠️ Fraction non trouvée dans les données - assignation de NaN")
    else:
        try:
            if isinstance(fraction_raw, float) and np.isnan(fraction_raw):
                fraction = np.nan
                if debug:
                    print(f"⚠️ Fraction est NaN dans les données originales")
            else:
                fraction = float(fraction_raw)
                if debug:
                    print(f"✅ Fraction extraite des données: {fraction}")
        except (ValueError, TypeError):
            fraction = np.nan
            if debug:
                print(f"⚠️ Fraction invalide dans les données - assignation de NaN")
    
    if debug:
        print(f"🔍 Valeurs extraites:")
        print(f"   - captured_time: {captured_time}")
        print(f"   - inuse_time: {inuse_time}")
        print(f"   - captured_frames: {captured_frames}")
        print(f"   - inuse_frames: {inuse_frames}")
        print(f"   - fraction finale: {fraction}")
    
    result = {
        'participant': str(entry.get('participant', entry.get('Participant', 'unknown'))),
        'trial': str(entry.get('trial', entry.get('Trial', 'unknown'))),
        'instrument': str(entry.get('instrument', entry.get('Instrument', entry.get('Tool', 'unknown')))),
        'captured_time': captured_time,
        'inuse_time': inuse_time,
        'captured_frames': captured_frames,
        'inuse_frames': inuse_frames,
        'fraction': fraction
    }
    
    if debug:
        print(f"🔍 Entrée parsée finale: {result}")
    
    return result

def analyze_by_participant_trial(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse groupée par participant/trial (moyenne des instruments).
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame agrégé par participant/trial
    """
    print("\n🔍 Analyse par participant/trial...")
    
    # Groupby participant/trial et calculer les moyennes
    grouped = df.groupby(['participant', 'trial']).agg({
        'captured_time': ['mean', 'std', 'min', 'max'],
        'inuse_time': ['mean', 'std', 'min', 'max'],
        'captured_frames': ['mean', 'std', 'min', 'max'],
        'inuse_frames': ['mean', 'std', 'min', 'max'],
        'fraction': ['mean', 'std', 'min', 'max'],
        'instrument': 'count'  # nombre d'instruments par trial
    }).round(4)
    
    # Aplatir les noms de colonnes
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={'instrument_count': 'num_instruments'}).reset_index()
    
    return grouped

def generate_summary_stats(df: pd.DataFrame) -> Dict:
    """
    Génère des statistiques résumées.
    
    Args:
        df: DataFrame des données
        
    Returns:
        Dictionnaire des statistiques
    """
    print("\n📈 Calcul des statistiques résumées...")
    
    stats = {
        'total_entries': len(df),
        'participants': df['participant'].nunique(),
        'trials': df['trial'].nunique(),
        'instruments': df['instrument'].nunique(),
        'fraction_stats': {
            'mean': df['fraction'].mean(),
            'median': df['fraction'].median(),
            'std': df['fraction'].std(),
            'min': df['fraction'].min(),
            'max': df['fraction'].max(),
            'q25': df['fraction'].quantile(0.25),
            'q75': df['fraction'].quantile(0.75)
        },
        'low_fraction_count': (df['fraction'] < 0.5).sum(),
        'high_fraction_count': (df['fraction'] >= 0.8).sum(),
        'zero_fraction_count': (df['fraction'] == 0).sum()
    }
    
    # Pourcentages
    stats['low_fraction_pct'] = (stats['low_fraction_count'] / len(df)) * 100
    stats['high_fraction_pct'] = (stats['high_fraction_count'] / len(df)) * 100
    stats['zero_fraction_pct'] = (stats['zero_fraction_count'] / len(df)) * 100
    
    return stats

def plot_fraction_distribution(df: pd.DataFrame, output_dir: str):
    """
    Graphique de distribution des fractions.
    """
    print("📊 Génération du graphique de distribution des fractions...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Distribution des Fractions de Détection', fontsize=16, fontweight='bold')
    
    # Histogramme global
    axes[0,0].hist(df['fraction'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].axvline(df['fraction'].mean(), color='red', linestyle='--', label=f'Moyenne: {df["fraction"].mean():.3f}')
    axes[0,0].axvline(df['fraction'].median(), color='green', linestyle='--', label=f'Médiane: {df["fraction"].median():.3f}')
    axes[0,0].set_xlabel('Fraction')
    axes[0,0].set_ylabel('Fréquence')
    axes[0,0].set_title('Distribution Globale des Fractions')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Box plot par instrument
    if df['instrument'].nunique() > 1:
        df.boxplot(column='fraction', by='instrument', ax=axes[0,1])
        axes[0,1].set_title('Distribution des Fractions par Instrument')
        axes[0,1].set_xlabel('Instrument')
        axes[0,1].set_ylabel('Fraction')
    else:
        axes[0,1].text(0.5, 0.5, 'Un seul instrument\ndans les données', 
                       ha='center', va='center', transform=axes[0,1].transAxes)
        axes[0,1].set_title('Distribution par Instrument')
    
    # Scatter plot: captured_time vs inuse_time
    scatter = axes[1,0].scatter(df['inuse_time'], df['captured_time'], 
                               c=df['fraction'], cmap='viridis', alpha=0.6)
    axes[1,0].plot([0, df['inuse_time'].max()], [0, df['inuse_time'].max()], 
                   'r--', alpha=0.5, label='Ligne parfaite (fraction=1)')
    axes[1,0].set_xlabel('Temps d\'utilisation (s)')
    axes[1,0].set_ylabel('Temps capturé (s)')
    axes[1,0].set_title('Temps Capturé vs Temps d\'Utilisation')
    axes[1,0].legend()
    plt.colorbar(scatter, ax=axes[1,0], label='Fraction')
    
    # Scatter plot: captured_frames vs inuse_frames
    scatter2 = axes[1,1].scatter(df['inuse_frames'], df['captured_frames'], 
                                c=df['fraction'], cmap='viridis', alpha=0.6)
    axes[1,1].plot([0, df['inuse_frames'].max()], [0, df['inuse_frames'].max()], 
                   'r--', alpha=0.5, label='Ligne parfaite (fraction=1)')
    axes[1,1].set_xlabel('Frames d\'utilisation')
    axes[1,1].set_ylabel('Frames capturées')
    axes[1,1].set_title('Frames Capturées vs Frames d\'Utilisation')
    axes[1,1].legend()
    plt.colorbar(scatter2, ax=axes[1,1], label='Fraction')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fraction_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_participant_analysis(df: pd.DataFrame, output_dir: str):
    """
    Analyse par participant.
    """
    print("📊 Génération de l'analyse par participant...")
    
    # Calculer les moyennes par participant
    participant_stats = df.groupby('participant').agg({
        'fraction': ['mean', 'std', 'count'],
        'captured_time': 'mean',
        'inuse_time': 'mean'
    }).round(4)
    
    participant_stats.columns = ['fraction_mean', 'fraction_std', 'num_entries', 
                                'captured_time_mean', 'inuse_time_mean']
    participant_stats = participant_stats.reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Analyse par Participant', fontsize=16, fontweight='bold')
    
    # Fraction moyenne par participant
    axes[0,0].bar(range(len(participant_stats)), participant_stats['fraction_mean'], 
                  color='lightcoral', alpha=0.7)
    axes[0,0].axhline(y=0.5, color='red', linestyle='--', label='Seuil critique (0.5)')
    axes[0,0].axhline(y=0.8, color='green', linestyle='--', label='Seuil bon (0.8)')
    axes[0,0].set_xlabel('Participant (index)')
    axes[0,0].set_ylabel('Fraction Moyenne')
    axes[0,0].set_title('Fraction Moyenne par Participant')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Distribution des fractions moyennes
    axes[0,1].hist(participant_stats['fraction_mean'], bins=20, alpha=0.7, 
                   color='lightblue', edgecolor='black')
    axes[0,1].set_xlabel('Fraction Moyenne')
    axes[0,1].set_ylabel('Nombre de Participants')
    axes[0,1].set_title('Distribution des Fractions Moyennes')
    axes[0,1].grid(True, alpha=0.3)
    
    # Nombre d'entrées par participant
    axes[1,0].bar(range(len(participant_stats)), participant_stats['num_entries'], 
                  color='lightgreen', alpha=0.7)
    axes[1,0].set_xlabel('Participant (index)')
    axes[1,0].set_ylabel('Nombre d\'Entrées')
    axes[1,0].set_title('Nombre d\'Entrées par Participant')
    axes[1,0].grid(True, alpha=0.3)
    
    # Corrélation entre temps d'utilisation et fraction
    axes[1,1].scatter(participant_stats['inuse_time_mean'], participant_stats['fraction_mean'], 
                      alpha=0.7, color='purple')
    axes[1,1].set_xlabel('Temps d\'Utilisation Moyen (s)')
    axes[1,1].set_ylabel('Fraction Moyenne')
    axes[1,1].set_title('Temps d\'Utilisation vs Fraction')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'participant_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return participant_stats

def plot_instrument_analysis(df: pd.DataFrame, output_dir: str):
    """
    Analyse par instrument.
    """
    print("📊 Génération de l'analyse par instrument...")
    
    # Calculer les moyennes par instrument
    instrument_stats = df.groupby('instrument').agg({
        'fraction': ['mean', 'std', 'count'],
        'captured_time': 'mean',
        'inuse_time': 'mean'
    }).round(4)
    
    instrument_stats.columns = ['fraction_mean', 'fraction_std', 'num_entries', 
                               'captured_time_mean', 'inuse_time_mean']
    instrument_stats = instrument_stats.reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Analyse par Instrument', fontsize=16, fontweight='bold')
    
    # Fraction moyenne par instrument
    x_pos = range(len(instrument_stats))
    axes[0,0].bar(x_pos, instrument_stats['fraction_mean'], 
                  color='orange', alpha=0.7)
    axes[0,0].axhline(y=0.5, color='red', linestyle='--', label='Seuil critique (0.5)')
    axes[0,0].axhline(y=0.8, color='green', linestyle='--', label='Seuil bon (0.8)')
    axes[0,0].set_xlabel('Instrument')
    axes[0,0].set_ylabel('Fraction Moyenne')
    axes[0,0].set_title('Fraction Moyenne par Instrument')
    axes[0,0].set_xticks(x_pos)
    axes[0,0].set_xticklabels(instrument_stats['instrument'], rotation=45, ha='right')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Box plot détaillé par instrument
    df.boxplot(column='fraction', by='instrument', ax=axes[0,1])
    axes[0,1].set_title('Distribution Détaillée par Instrument')
    axes[0,1].set_xlabel('Instrument')
    axes[0,1].set_ylabel('Fraction')
    
    # Nombre d'entrées par instrument
    axes[1,0].bar(x_pos, instrument_stats['num_entries'], 
                  color='cyan', alpha=0.7)
    axes[1,0].set_xlabel('Instrument')
    axes[1,0].set_ylabel('Nombre d\'Entrées')
    axes[1,0].set_title('Nombre d\'Entrées par Instrument')
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels(instrument_stats['instrument'], rotation=45, ha='right')
    axes[1,0].grid(True, alpha=0.3)
    
    # Temps moyen par instrument
    width = 0.35
    axes[1,1].bar([x - width/2 for x in x_pos], instrument_stats['captured_time_mean'], 
                  width, label='Temps Capturé', alpha=0.7, color='blue')
    axes[1,1].bar([x + width/2 for x in x_pos], instrument_stats['inuse_time_mean'], 
                  width, label='Temps d\'Utilisation', alpha=0.7, color='red')
    axes[1,1].set_xlabel('Instrument')
    axes[1,1].set_ylabel('Temps (s)')
    axes[1,1].set_title('Temps Moyen par Instrument')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(instrument_stats['instrument'], rotation=45, ha='right')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'instrument_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return instrument_stats

def identify_problematic_cases(df: pd.DataFrame, threshold_low: float = 0.5) -> pd.DataFrame:
    """
    Identifie les cas problématiques (fraction faible).
    
    Args:
        df: DataFrame des données
        threshold_low: Seuil en dessous duquel considérer comme problématique
        
    Returns:
        DataFrame des cas problématiques
    """
    print(f"\n🚨 Identification des cas problématiques (fraction < {threshold_low})...")
    
    problematic = df[df['fraction'] < threshold_low].copy()
    problematic = problematic.sort_values('fraction')
    
    print(f"   Trouvé {len(problematic)} cas problématiques sur {len(df)} ({len(problematic)/len(df)*100:.1f}%)")
    
    return problematic

def generate_report(df: pd.DataFrame, stats: Dict, participant_stats: pd.DataFrame, 
                   instrument_stats: pd.DataFrame, problematic: pd.DataFrame, 
                   output_dir: str):
    """
    Génère un rapport textuel complet.
    """
    print("📝 Génération du rapport textuel...")
    
    report_path = os.path.join(output_dir, 'missing_data_analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RAPPORT D'ANALYSE DES DONNÉES MANQUANTES\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. STATISTIQUES GÉNÉRALES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Nombre total d'entrées: {stats['total_entries']}\n")
        f.write(f"Participants uniques: {stats['participants']}\n")
        f.write(f"Trials uniques: {stats['trials']}\n")
        f.write(f"Instruments uniques: {stats['instruments']}\n\n")
        
        f.write("2. STATISTIQUES DES FRACTIONS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Moyenne: {stats['fraction_stats']['mean']:.4f}\n")
        f.write(f"Médiane: {stats['fraction_stats']['median']:.4f}\n")
        f.write(f"Écart-type: {stats['fraction_stats']['std']:.4f}\n")
        f.write(f"Minimum: {stats['fraction_stats']['min']:.4f}\n")
        f.write(f"Maximum: {stats['fraction_stats']['max']:.4f}\n")
        f.write(f"Q1 (25%): {stats['fraction_stats']['q25']:.4f}\n")
        f.write(f"Q3 (75%): {stats['fraction_stats']['q75']:.4f}\n\n")
        
        f.write("3. CLASSIFICATION DES CAS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Cas problématiques (fraction < 0.5): {stats['low_fraction_count']} ({stats['low_fraction_pct']:.1f}%)\n")
        f.write(f"Bons cas (fraction ≥ 0.8): {stats['high_fraction_count']} ({stats['high_fraction_pct']:.1f}%)\n")
        f.write(f"Cas avec fraction = 0: {stats['zero_fraction_count']} ({stats['zero_fraction_pct']:.1f}%)\n\n")
        
        f.write("4. ANALYSE PAR PARTICIPANT\n")
        f.write("-" * 40 + "\n")
        f.write(f"Participant avec la meilleure fraction moyenne: {participant_stats.loc[participant_stats['fraction_mean'].idxmax(), 'participant']} ({participant_stats['fraction_mean'].max():.4f})\n")
        f.write(f"Participant avec la pire fraction moyenne: {participant_stats.loc[participant_stats['fraction_mean'].idxmin(), 'participant']} ({participant_stats['fraction_mean'].min():.4f})\n")
        f.write(f"Participants avec fraction moyenne < 0.5: {(participant_stats['fraction_mean'] < 0.5).sum()}\n\n")
        
        f.write("5. ANALYSE PAR INSTRUMENT\n")
        f.write("-" * 40 + "\n")
        f.write(f"Instrument avec la meilleure fraction moyenne: {instrument_stats.loc[instrument_stats['fraction_mean'].idxmax(), 'instrument']} ({instrument_stats['fraction_mean'].max():.4f})\n")
        f.write(f"Instrument avec la pire fraction moyenne: {instrument_stats.loc[instrument_stats['fraction_mean'].idxmin(), 'instrument']} ({instrument_stats['fraction_mean'].min():.4f})\n\n")
        
        f.write("6. TOP 10 DES CAS PROBLÉMATIQUES\n")
        f.write("-" * 40 + "\n")
        f.write("Participant\tTrial\tInstrument\tFraction\tTemps_Utilisé\tTemps_Capturé\n")
        for _, row in problematic.head(10).iterrows():
            f.write(f"{row['participant']}\t{row['trial']}\t{row['instrument']}\t{row['fraction']:.4f}\t{row['inuse_time']:.2f}\t{row['captured_time']:.2f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("RECOMMANDATIONS:\n")
        f.write("- Cas avec fraction < 0.5: Vérifier la qualité des données\n")
        f.write("- Cas avec fraction = 0: Données probablement inutilisables\n")
        f.write("- Cas avec fraction > 0.8: Données de bonne qualité\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ Rapport sauvegardé: {report_path}")

def main():
    """
    Fonction principale d'analyse.
    """
    parser = argparse.ArgumentParser(description='Analyse des données manquantes')
    parser.add_argument('--input', '-i', required=True, help='Fichier JSON des données manquantes')
    parser.add_argument('--out_dir', '-o', default='./missing_data_analysis', help='Dossier de sortie')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Seuil pour cas problématiques')
    
    args = parser.parse_args()
    
    # Créer le dossier de sortie
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Dossier de sortie: {output_dir}")
    
    # Charger les données
    df = load_missing_data(args.input)
    
    # Calculer les statistiques
    stats = generate_summary_stats(df)
    
    # Analyses groupées
    participant_trial_stats = analyze_by_participant_trial(df)
    
    # Graphiques
    plot_fraction_distribution(df, str(output_dir))
    participant_stats = plot_participant_analysis(df, str(output_dir))
    instrument_stats = plot_instrument_analysis(df, str(output_dir))
    
    # Identifier les cas problématiques
    problematic = identify_problematic_cases(df, args.threshold)
    
    # Générer le rapport
    generate_report(df, stats, participant_stats, instrument_stats, problematic, str(output_dir))
    
    # Sauvegarder les DataFrames
    df.to_csv(output_dir / 'raw_data.csv', index=False)
    participant_trial_stats.to_csv(output_dir / 'participant_trial_stats.csv', index=False)
    participant_stats.to_csv(output_dir / 'participant_stats.csv', index=False)
    instrument_stats.to_csv(output_dir / 'instrument_stats.csv', index=False)
    problematic.to_csv(output_dir / 'problematic_cases.csv', index=False)
    
    # Générer le rapport des pires combinaisons
    generate_worst_cases_report(df, str(output_dir))
    
    print(f"\n✅ Analyse terminée! Résultats dans: {output_dir}")
    print(f"📊 Résumé rapide:")
    print(f"   - Fraction moyenne: {stats['fraction_stats']['mean']:.3f}")
    print(f"   - Cas problématiques: {stats['low_fraction_count']} ({stats['low_fraction_pct']:.1f}%)")
    print(f"   - Bons cas: {stats['high_fraction_count']} ({stats['high_fraction_pct']:.1f}%)")

def generate_worst_cases_report(df: pd.DataFrame, output_dir: str):
    """
    Génère un rapport détaillé des pires cas de tracking classés par ordre décroissant.
    """
    output_path = Path(output_dir)
    
    # Trier par fraction croissante (pires scores en premier)
    df_sorted = df.sort_values('fraction', ascending=True).copy()
    
    # Créer une colonne combinée pour l'identification
    df_sorted['combination'] = df_sorted.apply(
        lambda row: f"{row['participant']}_{row['trial']}_{row['instrument']}", 
        axis=1
    )
    
    # Fichier de rapport principal
    report_file = output_path / "worst_tracking_combinations.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RAPPORT DES PIRES COMBINAISONS DE TRACKING\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"📊 Analyse basée sur {len(df_sorted)} combinaisons participant-trial-instrument\n")
        f.write(f"📉 Classement par ordre décroissant de qualité (fraction)\n\n")
        
        # Statistiques générales
        f.write("📈 STATISTIQUES GÉNÉRALES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Score moyen: {df_sorted['fraction'].mean():.4f}\n")
        f.write(f"Score médian: {df_sorted['fraction'].median():.4f}\n")
        f.write(f"Pire score: {df_sorted['fraction'].min():.4f}\n")
        f.write(f"Meilleur score: {df_sorted['fraction'].max():.4f}\n")
        f.write(f"Écart-type: {df_sorted['fraction'].std():.4f}\n\n")
        
        # Cas problématiques (fraction < 0.5)
        problematic = df_sorted[df_sorted['fraction'] < 0.5]
        f.write(f"🚨 CAS PROBLÉMATIQUES (fraction < 0.5): {len(problematic)} cas\n")
        f.write("-" * 50 + "\n\n")
        
        # Top 20 des pires cas
        f.write("🔴 TOP 20 DES PIRES COMBINAISONS:\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'Rang':<4} {'Participant':<12} {'Trial':<8} {'Instrument':<12} {'Fraction':<10} {'Détails'}\n")
        f.write("-" * 80 + "\n")
        
        for i, (_, row) in enumerate(df_sorted.head(20).iterrows(), 1):
            details = f"Cap:{row['captured_time']:.1f}s/Inuse:{row['inuse_time']:.1f}s"
            f.write(f"{i:<4} {row['participant']:<12} {row['trial']:<8} {row['instrument']:<12} "
                   f"{row['fraction']:<10.4f} {details}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        
        # Analyse par catégories
        f.write("\n📊 ANALYSE PAR CATÉGORIES:\n")
        f.write("=" * 40 + "\n\n")
        
        # Par instrument (moyennes)
        f.write("🔧 PERFORMANCE MOYENNE PAR INSTRUMENT:\n")
        f.write("-" * 40 + "\n")
        instrument_stats = df_sorted.groupby('instrument')['fraction'].agg(['mean', 'std', 'count']).sort_values('mean')
        for instrument, stats in instrument_stats.iterrows():
            f.write(f"{instrument:<12}: {stats['mean']:.4f} ± {stats['std']:.4f} ({stats['count']} cas)\n")
        
        f.write("\n👤 PERFORMANCE MOYENNE PAR PARTICIPANT (10 pires):\n")
        f.write("-" * 50 + "\n")
        participant_stats = df_sorted.groupby('participant')['fraction'].agg(['mean', 'std', 'count']).sort_values('mean')
        for participant, stats in participant_stats.head(10).iterrows():
            f.write(f"Participant {participant}: {stats['mean']:.4f} ± {stats['std']:.4f} ({stats['count']} cas)\n")
        
        f.write("\n🧪 PERFORMANCE MOYENNE PAR TRIAL:\n")
        f.write("-" * 30 + "\n")
        trial_stats = df_sorted.groupby('trial')['fraction'].agg(['mean', 'std', 'count']).sort_values('mean')
        for trial, stats in trial_stats.iterrows():
            f.write(f"{trial}: {stats['mean']:.4f} ± {stats['std']:.4f} ({stats['count']} cas)\n")
        
        # Liste complète des cas problématiques
        if len(problematic) > 0:
            f.write(f"\n🚨 LISTE COMPLÈTE DES CAS PROBLÉMATIQUES ({len(problematic)} cas):\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'Participant':<12} {'Trial':<8} {'Instrument':<12} {'Fraction':<10} {'Temps Cap/Inuse'}\n")
            f.write("-" * 60 + "\n")
            
            for _, row in problematic.iterrows():
                f.write(f"{row['participant']:<12} {row['trial']:<8} {row['instrument']:<12} "
                       f"{row['fraction']:<10.4f} {row['captured_time']:.1f}s/{row['inuse_time']:.1f}s\n")
    
    # Fichier CSV des pires cas
    csv_file = output_path / "worst_tracking_combinations.csv"
    df_export = df_sorted[['participant', 'trial', 'instrument', 'fraction', 
                          'captured_time', 'inuse_time', 'captured_frames', 'inuse_frames']].copy()
    df_export.to_csv(csv_file, index=False)
    
    print(f"📝 Rapport des pires cas généré: {report_file}")
    print(f"📊 CSV exporté: {csv_file}")
    
    # Afficher un résumé rapide des pires cas
    print(f"\n🔴 TOP 5 DES PIRES CAS:")
    for i, (_, row) in enumerate(df_sorted.head(5).iterrows(), 1):
        print(f"   {i}. {row['participant']}_{row['trial']}_{row['instrument']}: {row['fraction']:.4f}")

if __name__ == "__main__":
    main()
