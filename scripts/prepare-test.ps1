param(
    [string]$PythonCommand = "py"
)

$ErrorActionPreference = "Stop"
Clear-Host
Write-Host "=== ISOLATED YOLO26 TEST ENVIRONMENT ==="

$venv = Join-Path $PSScriptRoot "..\.venv"
$requirements = Join-Path $PSScriptRoot "..\YOLO26Surveillance\requirements.txt"

if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    & $PythonCommand -3.12 -m venv $venv
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.12 virtual environment creation failed"
    }
}

$pythonExe = Join-Path $venv "Scripts\python.exe"
& $pythonExe -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) {
    throw "Dependency install failed"
}

& (Join-Path $PSScriptRoot "smoke-test.ps1") -PythonExe $pythonExe
