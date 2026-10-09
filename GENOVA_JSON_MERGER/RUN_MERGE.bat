@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_MERGE.ps1"
if %ERRORLEVEL% neq 0 (
  echo.
  echo RUN_MERGE.ps1 basarisiz oldu. Python dogrudan deneniyor...
  py -3 merge_panel_jsons.py --input-dir input --output output\GENOVA_FINAL_SUBMISSION.json --format internal
  if %ERRORLEVEL% equ 0 goto done
  python merge_panel_jsons.py --input-dir input --output output\GENOVA_FINAL_SUBMISSION.json --format internal
  if %ERRORLEVEL% equ 0 goto done
  python3 merge_panel_jsons.py --input-dir input --output output\GENOVA_FINAL_SUBMISSION.json --format internal
  if %ERRORLEVEL% neq 0 (
    echo.
    echo Python bulunamadi veya merge basarisiz oldu.
    exit /b %ERRORLEVEL%
  )
)

:done
endlocal
