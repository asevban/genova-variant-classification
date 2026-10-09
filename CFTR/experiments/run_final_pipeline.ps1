$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$python = Join-Path $projectRoot ".venv-tabpfn\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python ortamı bulunamadı: $python"
}

Write-Host "1/6 — 319 özellikli veri seti oluşturuluyor"
node build_scenario1d_319_dataset.js

Write-Host "2/6 — ID3 nested CV"
$env:ID3_ONLY_319 = "1"
Remove-Item Env:ID3_MISSING_INDICATORS -ErrorAction SilentlyContinue
Remove-Item Env:ID3_RUN_SUFFIX -ErrorAction SilentlyContinue
node id3_benign_weighted_322_vs_320.js

Write-Host "3/6 — CatBoost nested CV"
$env:CATBOOST_ONLY_319 = "1"
Remove-Item Env:MISSING_INDICATORS -ErrorAction SilentlyContinue
& $python catboost_benign_weighted_finalists_nested_cv.py

Write-Host "4/6 — Random Forest nested CV"
& $python evaluate_randomforest_319_nested.py 319

Write-Host "5/6 — AdaBoost yedek bileşen tahminleri"
& $python sklearn_scenario1c_320_nested_cv.py 319

Write-Host "6/6 — Ana ve yedek ensemble karşılaştırması"
& $python evaluate_319_randomforest_hybrids.py

Remove-Item Env:ID3_ONLY_319 -ErrorAction SilentlyContinue
Remove-Item Env:CATBOOST_ONLY_319 -ErrorAction SilentlyContinue

Write-Host "Final pipeline tamamlandı. Sonuç: results\randomforest_319_hybrid_scores.csv"
