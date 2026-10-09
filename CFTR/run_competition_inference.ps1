param(
    [Parameter(Mandatory=$true)][string]$TestCsv,
    [string]$OutputCsv = ""
)

$ErrorActionPreference = "Stop"
$PanelRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$FinalDir = Join-Path $PanelRoot "final"
$Python = Join-Path $PanelRoot ".venv\Scripts\python.exe"

if (-not $OutputCsv) {
    $OutputCsv = Join-Path $PanelRoot "output\CFTR_predictions.csv"
}

$OutputDir = Split-Path -Parent $OutputCsv
if ($OutputDir -and -not (Test-Path -LiteralPath $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "CFTR Python ortami bulunamadi: $Python. Once GENOVA_FINAL\setup_envs.ps1 calistirin."
}
if (-not (Test-Path -LiteralPath $TestCsv)) {
    throw "Test dosyasi bulunamadi: $TestCsv"
}

$Predict = Join-Path $FinalDir "predict.py"
$Verify = Join-Path $FinalDir "verify_submission.py"
$Audit = [IO.Path]::ChangeExtension($OutputCsv, ".audit.csv")
$Drift = [IO.Path]::ChangeExtension($OutputCsv, ".drift.json")

& $Python $Predict --input $TestCsv --output $OutputCsv --mode main --audit $Audit --drift-report $Drift
& $Python $Verify --test $TestCsv --submission $OutputCsv

Write-Host "Ana model kimligi: CFTR_319_ID3_CATBOOST_RF_EQUAL_SOFT_VOTING_5SEED"
Write-Host "Tahmin: $OutputCsv"
Write-Host "Denetim: $Audit"
Write-Host "Drift: $Drift"
