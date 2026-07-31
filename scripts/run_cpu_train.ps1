# =============================================================================
# ICEMS — Entrainement multi-task (y4 + y9) sur CPU (Windows)
#
# Usage :
#   .\scripts\run_cpu_train.ps1 -Mode smoke
#   .\scripts\run_cpu_train.ps1 -Mode smoke -NoAug
#   .\scripts\run_cpu_train.ps1 -Mode full -MaxFolds 10 -NoAug
#   .\scripts\run_cpu_train.ps1 -Mode full
# =============================================================================
param(
    [ValidateSet("smoke", "full")]
    [string]$Mode = "smoke",

    [int]$MaxFolds = 0,
    [int]$Epochs = 40,
    [int]$Patience = 12,
    [int]$BatchSize = 8,
    [double]$Alpha = 0.4,
    [switch]$NoAug,
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "ERREUR: .venv introuvable. Lance d'abord:" -ForegroundColor Red
    Write-Host "  python -m venv .venv"
    Write-Host "  pip install torch numpy scipy scikit-learn pandas tslearn tqdm"
    exit 1
}

. .\.venv\Scripts\Activate.ps1
$env:PYTHONUNBUFFERED = "1"

foreach ($f in @("data\continuous_per_trial.pkl", "data\filtered_data.json", "src\train_clf_multitask.py")) {
    if (-not (Test-Path $f)) {
        Write-Host "ERREUR: fichier manquant -> $f" -ForegroundColor Red
        exit 1
    }
}

python -c "import torch; print('Device CPU (cuda=' + str(torch.cuda.is_available()) + ')')"

if ($OutDir -eq "") {
    $suffix = if ($NoAug) { "noaug" } else { "aug" }
    $OutDir = "results/clf_multitask/cpu_${Mode}_${suffix}"
}

$argsList = @(
    "-u", "src/train_clf_multitask.py",
    "--out-dir", $OutDir,
    "--epochs", "$Epochs",
    "--patience", "$Patience",
    "--batch-size", "$BatchSize",
    "--alpha", "$Alpha"
)

if ($Mode -eq "smoke") {
    $argsList += "--smoke-test"
} else {
    $argsList += "--full"
}

if ($MaxFolds -gt 0) {
    $argsList += @("--max-folds", "$MaxFolds")
}

if ($NoAug) {
    $argsList += "--no-aug"
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " ICEMS CPU training" -ForegroundColor Cyan
Write-Host " Mode     : $Mode"
Write-Host " OutDir   : $OutDir"
Write-Host " Epochs   : $Epochs"
Write-Host " Batch    : $BatchSize"
Write-Host " NoAug    : $NoAug"
Write-Host " MaxFolds : $(if ($MaxFolds -gt 0) { $MaxFolds } else { 'all (47)' })"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

python @argsList

Write-Host ""
Write-Host "Termine. Resultats dans: $OutDir" -ForegroundColor Green
Write-Host "  - predictions.csv"
Write-Host "  - metrics.json"
