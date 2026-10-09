param(
    [string]$InputPath = "",
    [string]$TeamId = "885171",
    [string]$ApplicationId = "4807248"
)

$ErrorActionPreference = "Stop"
$PanelRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Panel = Split-Path -Leaf $PanelRoot

if (-not $InputPath) {
    $InputPath = Join-Path $PanelRoot "input\$Panel.csv"
}

$Python = Join-Path $PanelRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = Join-Path $PanelRoot ".venv\bin\python"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "$Panel Python ortami bulunamadi: $PanelRoot\.venv"
}

& $Python (Join-Path $PanelRoot "run_panel_final.py") --panel $Panel --input $InputPath --team-id $TeamId --application-id $ApplicationId
exit $LASTEXITCODE
