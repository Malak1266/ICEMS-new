# Script PowerShell pour configurer l'environnement de développement
# Projet: Prédiction d'expertise chirurgicale

Write-Host "🚀 Configuration de l'environnement de développement" -ForegroundColor Green
Write-Host "$('=' * 60)"

# Vérifier si Python est installé
Write-Host "🔍 Vérification de Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version
    Write-Host "✅ Python trouvé: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python n'est pas installé ou n'est pas dans le PATH" -ForegroundColor Red
    exit 1
}

# Nom de l'environnement virtuel
$venvName = "surgical_expertise_env"

# Créer l'environnement virtuel
Write-Host "🔧 Création de l'environnement virtuel '$venvName'..." -ForegroundColor Yellow
python -m venv $venvName

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Environnement virtuel créé avec succès!" -ForegroundColor Green
} else {
    Write-Host "❌ Erreur lors de la création de l'environnement virtuel" -ForegroundColor Red
    exit 1
}

# Activer l'environnement virtuel
Write-Host "🔄 Activation de l'environnement virtuel..." -ForegroundColor Yellow
& "$venvName\Scripts\Activate.ps1"

# Mettre à jour pip
Write-Host "📦 Mise à jour de pip..." -ForegroundColor Yellow
& "$venvName\Scripts\python.exe" -m pip install --upgrade pip

# Installer les dépendances
Write-Host "📦 Installation des dépendances depuis requirements.txt..." -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    & "$venvName\Scripts\pip.exe" install -r requirements.txt
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Toutes les dépendances ont été installées!" -ForegroundColor Green
    } else {
        Write-Host "❌ Erreur lors de l'installation des dépendances" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "❌ Fichier requirements.txt non trouvé!" -ForegroundColor Red
    exit 1
}

# Installer ipykernel pour Jupyter
Write-Host "📓 Installation d'ipykernel..." -ForegroundColor Yellow
& "$venvName\Scripts\python.exe" -m pip install ipykernel

# Créer le kernel Jupyter
Write-Host "📓 Création du kernel Jupyter..." -ForegroundColor Yellow
& "$venvName\Scripts\python.exe" -m ipykernel install --user --name $venvName --display-name "Python ($venvName)"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Kernel Jupyter '$venvName' créé!" -ForegroundColor Green
} else {
    Write-Host "❌ Erreur lors de la création du kernel Jupyter" -ForegroundColor Red
}

# Créer la structure de dossiers
Write-Host "📁 Création de la structure du projet..." -ForegroundColor Yellow
$directories = @(
    "data\processed_data_json",
    "models",
    "logs", 
    "results",
    "notebooks",
    "scripts"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✅ Dossier créé: $dir" -ForegroundColor Green
    } else {
        Write-Host "📁 Dossier existe déjà: $dir" -ForegroundColor Cyan
    }
}

Write-Host "`n$('=' * 60)"
Write-Host "🎉 Configuration terminée avec succès!" -ForegroundColor Green
Write-Host "📝 Pour activer l'environnement à l'avenir:" -ForegroundColor Yellow
Write-Host "   $venvName\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "📓 Pour lancer Jupyter Notebook:" -ForegroundColor Yellow
Write-Host "   jupyter notebook" -ForegroundColor Cyan
Write-Host "   (Sélectionnez le kernel '$venvName' dans vos notebooks)" -ForegroundColor Cyan
Write-Host "🐍 Pour utiliser Python dans cet environnement:" -ForegroundColor Yellow
Write-Host "   $venvName\Scripts\python.exe" -ForegroundColor Cyan
