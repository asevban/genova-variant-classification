$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Script = Join-Path $Root "merge_panel_jsons.py"

function Find-Python {
    $localCandidates = @(
        (Join-Path $Root "..\PAH\.venv\Scripts\python.exe"),
        (Join-Path $Root "..\CFTR\.venv\Scripts\python.exe"),
        (Join-Path $Root "..\MASTER\.venv\Scripts\python.exe"),
        (Join-Path $Root "..\KANSER\.venv\Scripts\python.exe"),
        (Join-Path $Root "..\GENOVA_FINAL\PAH\.venv\Scripts\python.exe"),
        (Join-Path $Root "..\GENOVA_FINAL\CFTR\.venv\Scripts\python.exe"),
        (Join-Path $Root "..\GENOVA_FINAL\MASTER\.venv\Scripts\python.exe"),
        (Join-Path $Root "..\GENOVA_FINAL\KANSER\.venv\Scripts\python.exe")
    )

    foreach ($candidate in $localCandidates) {
        $resolved = Resolve-Path -LiteralPath $candidate -ErrorAction SilentlyContinue
        if ($resolved) {
            try {
                & $resolved.Path --version *> $null
                if ($LASTEXITCODE -eq 0) {
                    return [pscustomobject]@{ Exe = $resolved.Path; Args = @() }
                }
            } catch {
            }
        }
    }

    $candidates = @(
        "py -3.11",
        "py -3.12",
        "py -3.10",
        "py -3.9",
        "py -3.8",
        "python",
        "python3"
    )

    foreach ($candidate in $candidates) {
        $parts = $candidate -split " "
        $exe = $parts[0]
        $arguments = @()
        if ($parts.Count -gt 1) {
            $arguments = $parts[1..($parts.Count - 1)]
        }
        try {
            & $exe @arguments --version *> $null
            if ($LASTEXITCODE -eq 0) {
                return [pscustomobject]@{ Exe = $exe; Args = $arguments }
            }
        } catch {
        }
    }
    throw "Python bulunamadi. Lutfen Python 3.8+ kurulu oldugunu kontrol edin."
}

$PythonCommand = Find-Python
$PythonExe = $PythonCommand.Exe
$PythonArgs = @($PythonCommand.Args)
& $PythonExe @PythonArgs $Script --input-dir "input" --output "output\GENOVA_FINAL_SUBMISSION.json" --format "internal"

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
