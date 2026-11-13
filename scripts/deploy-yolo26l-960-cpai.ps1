# Stage isolated YOLO26l-960 CodeProject.AI module alongside production.
clear

$ErrorActionPreference = "Stop"
$Root = "G:\YOLO26-L960"
$Model = Join-Path $Root "yolo26l-surveillance-960.onnx"
$Stage = Join-Path $Root "cpai-deploy\YOLO26LSurveillance960"
$Modules = "C:\Program Files\CodeProject\AI\modules"
$Target = Join-Path $Modules "YOLO26LSurveillance960"
$Expected = "471090E1E1304EF1E48F1801E48D8E91F574E09708171BED803F08EED5E4288A"
$Base = "https://raw.githubusercontent.com/craigmcdaniel/YOLO26-Surveillance/test/deploy/codeproject-ai/YOLO26LSurveillance960"
$Names = @("detect_adapter.py", "yolo26_surveillance.py", "modulesettings.json", "modulesettings.windows.json", "requirements.txt", "install.bat")

Write-Host "=== YOLO26L-960 CODEPROJECT.AI MODULE DEPLOY ==="
Write-Host "Model:  $Model"
Write-Host "Target: $Target"

if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    Write-Host "STOPPED: model missing"
    return
}
if (-not (Test-Path -LiteralPath $Modules -PathType Container)) {
    Write-Host "STOPPED: CodeProject.AI module installation directory missing"
    return
}
if (Test-Path -LiteralPath $Target) {
    Write-Host "STOPPED: candidate module already exists; no overwrites"
    return
}
$Hash = (Get-FileHash -LiteralPath $Model -Algorithm SHA256).Hash
if ($Hash -ne $Expected) {
    Write-Host "STOPPED: unexpected ONNX SHA256 $Hash"
    return
}

try {
    New-Item -Path $Stage -ItemType Directory -Force | Out-Null
    foreach ($Name in $Names) {
        Write-Host "Fetching $Name"
        Invoke-WebRequest -Uri "$Base/$Name" -OutFile (Join-Path $Stage $Name)
    }
    Copy-Item -LiteralPath $Model -Destination (Join-Path $Stage "yolo26l-surveillance-960.onnx") -Force
    $StagedHash = (Get-FileHash -LiteralPath (Join-Path $Stage "yolo26l-surveillance-960.onnx") -Algorithm SHA256).Hash
    if ($StagedHash -ne $Expected) {
        throw "Model changed during staging"
    }
    $Settings = Get-Content -LiteralPath (Join-Path $Stage "modulesettings.json") -Raw | ConvertFrom-Json
    if (-not $Settings.Modules.YOLO26LSurveillance960) {
        throw "Module ID missing"
    }
    $Launch = $Settings.Modules.YOLO26LSurveillance960.LaunchSettings
    $Route = $Settings.Modules.YOLO26LSurveillance960.RouteMaps[0].Route
    if ($Launch.AutoStart -or $Route -ne "vision/yolo26l-surveillance-960") {
        throw "Module activation or route differs from expected test configuration"
    }
    New-Item -Path $Target -ItemType Directory -ErrorAction Stop | Out-Null
    foreach ($Name in $Names) {
        Copy-Item -LiteralPath (Join-Path $Stage $Name) -Destination (Join-Path $Target $Name) -ErrorAction Stop
    }
    $InstalledAdapter = Get-Content -LiteralPath (Join-Path $Target "detect_adapter.py") -Raw
    if (-not $InstalledAdapter.Contains($Model)) {
        throw "Installed adapter does not reference the model on G:"
    }
    Write-Host "=== MODULE FILES INSTALLED ==="
    Write-Host "Directory: $Target"
    Write-Host "Route: $Route"
    Write-Host "ONNX stays on G: $Model"
    Write-Host "AutoStart: false"
    Write-Host "Service not restarted; route not yet active."
    Get-CimInstance Win32_Service | Where-Object { $_.Name -match "CodeProject" } | Select-Object Name, State, PathName | Format-List
}
catch {
    Write-Host "DEPLOYMENT ERROR: $($_.Exception.Message)"
    Write-Host "Production module unchanged. Inspect partial target before retry."
}
