"""Stage 5B: isolated DirectML smoke/performance check, synthetic input only."""
from pathlib import Path
from time import perf_counter
import statistics

import numpy as np
import onnxruntime as ort

model = Path(r"G:\YOLO26-L960\yolo26l-surveillance-960.onnx")
if not model.is_file():
    raise SystemExit(f"Model missing: {model}")

options = ort.SessionOptions()
options.enable_mem_pattern = False
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

print("=== DIRECTML 960 EXECUTION TEST ===", flush=True)
print("Available providers:", ort.get_available_providers(), flush=True)
if "DmlExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("DirectML provider unavailable")

session = ort.InferenceSession(
    str(model),
    sess_options=options,
    providers=["DmlExecutionProvider", "CPUExecutionProvider"],
)
print("Session providers:", session.get_providers(), flush=True)
input_info = session.get_inputs()[0]
if input_info.shape != [1, 3, 960, 960]:
    raise RuntimeError(f"Wrong input shape: {input_info.shape}")
outputs = {item.name: item.shape for item in session.get_outputs()}
if outputs.get("surveillance_output") != [1, 12, 18900]:
    raise RuntimeError(f"Unexpected output shape: {outputs}")

tensor = np.zeros((1, 3, 960, 960), dtype=np.float32)
feed = {input_info.name: tensor}

print("Warming up...", flush=True)
for _ in range(3):
    result = session.run(["surveillance_output"], feed)[0]
if result.shape != (1, 12, 18900) or not np.isfinite(result).all():
    raise RuntimeError("Invalid DirectML session result")

print("Running 10 synthetic-input inferences...", flush=True)
times = []
for _ in range(10):
    before = perf_counter()
    result = session.run(["surveillance_output"], feed)[0]
    times.append((perf_counter() - before) * 1000)

print("OUTPUT:", result.shape, flush=True)
print("Median session.run ms:", round(statistics.median(times), 2), flush=True)
print("Minimum ms:", round(min(times), 2), flush=True)
print("Maximum ms:", round(max(times), 2), flush=True)
print("NOTE: CPU EP fallback may occur; these are whole-session timings.", flush=True)
print("Production unchanged.", flush=True)
