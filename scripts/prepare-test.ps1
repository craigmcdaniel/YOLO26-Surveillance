param(
    [Parameter(Mandatory=$true)]
    [string]$PythonExe
)

$ErrorActionPreference = "Stop"
Clear-Host
Write-Host "=== ISOLATED YOLO26 TEST ENVIRONMENT ==="

$venv = Join-Path $PSScriptRoot "..\.venv"
$requirements = Join-Path $PSScriptRoot "..\YOLO26Surveillance\requirements.txt"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python 3.12 executable not found: $PythonExe"
}

$version = & $PythonExe -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ($LASTEXITCODE -ne 0 -or $version.Trim() -ne "3.12") {
    throw "Expected Python 3.12; got $version"
}

if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    & $PythonExe -m venv $venv
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.12 virtual environment creation failed"
    }
}

$pythonInVenv = Join-Path $venv "Scripts\python.exe"

& $pythonInVenv -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed"
}

& (Join-Path $PSScriptRoot "smoke-test.ps1") -PythonExe $pythonInVenv
