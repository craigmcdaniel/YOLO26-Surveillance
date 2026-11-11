param(
    [Parameter(Mandatory=$true)]
    [string]$PythonExe,
    [string]$ModelPath = "$PSScriptRoot\..\YOLO26Surveillance\yolo26s-filtered-test.onnx"
)

$ErrorActionPreference = "Stop"
Clear-Host
Write-Host "=== YOLO26 SURVEILLANCE ISOLATED SMOKE TEST ==="

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

if (-not (Test-Path $ModelPath)) {
    throw "ONNX model not found: $ModelPath"
}

$env:YOLO26_MODEL_PATH = (Resolve-Path $ModelPath).Path

$code = @'
import os
import onnxruntime as ort
model = os.environ["YOLO26_MODEL_PATH"]
s = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
print("Providers:", s.get_providers())
print("Inputs:", [(v.name, v.shape) for v in s.get_inputs()])
print("Outputs:", [(v.name, v.shape) for v in s.get_outputs()])
assert any(v.name == "surveillance_output" and v.shape[1] == 12 for v in s.get_outputs()), "Expected 12-channel surveillance_output missing"
print("PASS: Model initializes and filtered output is present")
'@

$code | & $PythonExe -
if ($LASTEXITCODE -ne 0) {
    throw "Smoke test failed"
}

Write-Host "=== TEST COMPLETE ==="
