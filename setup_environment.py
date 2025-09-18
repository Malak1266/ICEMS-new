"""
Script de configuration de l'environnement de développement
pour le projet de prédiction d'expertise chirurgicale
"""

import subprocess
import sys
import os
from pathlib import Path

def create_virtual_environment():
    """Créer un environnement virtuel pour le projet"""
    print("🔧 Création de l'environnement virtuel...")
    
    # Nom de l'environnement virtuel
    venv_name = "surgical_expertise_env"
    
    try:
        # Créer l'environnement virtuel
        subprocess.run([sys.executable, "-m", "venv", venv_name], check=True)
        print(f"✅ Environnement virtuel '{venv_name}' créé avec succès!")
        
        # Chemin vers l'environnement virtuel
        if os.name == 'nt':  # Windows
            python_path = os.path.join(venv_name, "Scripts", "python.exe")
            pip_path = os.path.join(venv_name, "Scripts", "pip.exe")
        else:  # Linux/macOS
            python_path = os.path.join(venv_name, "bin", "python")
            pip_path = os.path.join(venv_name, "bin", "pip")
        
        return python_path, pip_path, venv_name
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de la création de l'environnement virtuel: {e}")
        return None, None, None

def install_requirements(pip_path):
    """Installer les dépendances depuis requirements.txt"""
    print("📦 Installation des dépendances...")
    
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"❌ Fichier {requirements_file} non trouvé!")
        return False
    
    try:
        # Mettre à jour pip
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        print("✅ pip mis à jour!")
        
        # Installer les dépendances
        subprocess.run([pip_path, "install", "-r", requirements_file], check=True)
        print("✅ Toutes les dépendances ont été installées!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'installation des dépendances: {e}")
        return False

def create_jupyter_kernel(python_path, venv_name):
    """Créer un kernel Jupyter pour l'environnement virtuel"""
    print("📓 Configuration du kernel Jupyter...")
    
    try:
        # Installer ipykernel dans l'environnement virtuel
        subprocess.run([python_path, "-m", "pip", "install", "ipykernel"], check=True)
        
        # Créer le kernel Jupyter
        subprocess.run([
            python_path, "-m", "ipykernel", "install", 
            "--user", "--name", venv_name, 
            "--display-name", f"Python ({venv_name})"
        ], check=True)
        
        print(f"✅ Kernel Jupyter '{venv_name}' créé!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de la création du kernel Jupyter: {e}")
        return False

def create_project_structure():
    """Créer la structure de dossiers du projet"""
    print("📁 Création de la structure du projet...")
    
    directories = [
        "data/processed_data_json",
        "models",
        "logs",
        "results",
        "notebooks",
        "scripts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Dossier créé: {directory}")

def main():
    """Fonction principale de configuration"""
    print("🚀 Configuration de l'environnement de développement")
    print("=" * 60)
    
    # Créer la structure du projet
    create_project_structure()
    
    # Créer l'environnement virtuel
    python_path, pip_path, venv_name = create_virtual_environment()
    
    if python_path is None:
        print("❌ Impossible de continuer sans environnement virtuel")
        return
    
    # Installer les dépendances
    if install_requirements(pip_path):
        print("✅ Dépendances installées avec succès!")
    else:
        print("❌ Échec de l'installation des dépendances")
        return
    
    # Créer le kernel Jupyter
    if create_jupyter_kernel(python_path, venv_name):
        print("✅ Kernel Jupyter configuré!")
    
    print("\n" + "=" * 60)
    print("🎉 Configuration terminée!")
    print(f"📝 Pour activer l'environnement:")
    
    if os.name == 'nt':  # Windows
        print(f"   {venv_name}\\Scripts\\activate")
    else:  # Linux/macOS
        print(f"   source {venv_name}/bin/activate")
    
    print(f"📓 Pour lancer Jupyter:")
    print(f"   jupyter notebook")
    print(f"   (Sélectionnez le kernel '{venv_name}' dans vos notebooks)")

if __name__ == "__main__":
    main()
