"""Stage 4: CPU execution smoke test of filtered YOLO26l 960.
Never edits source or production models. No image dataset required.
"""
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

model = Path(r"G:\YOLO26-L960\yolo26l-surveillance-960.onnx")
if not model.is_file():
    raise SystemExit(f"Missing model: {model}")

print("=== STAGE 4: ONNX RUNTIME CPU SMOKE TEST ===", flush=True)
print("Available providers:", ort.get_available_providers(), flush=True)

session = ort.InferenceSession(
    str(model),
    providers=["CPUExecutionProvider"],
)
input_node = session.get_inputs()[0]
outputs = {node.name: node.shape for node in session.get_outputs()}
print("Input:", input_node.name, input_node.shape, flush=True)
print("Outputs:", outputs, flush=True)

if input_node.shape != [1, 3, 960, 960]:
    raise RuntimeError("Model input shape mismatch")
if outputs.get("output0") != [1, 84, 18900]:
    raise RuntimeError("Raw output shape mismatch")
if outputs.get("surveillance_output") != [1, 12, 18900]:
    raise RuntimeError("Surveillance output shape mismatch")

frame = np.zeros((1, 3, 960, 960), dtype=np.float32)
start = time.perf_counter()
raw, surveillance = session.run(
    ["output0", "surveillance_output"],
    {input_node.name: frame},
)
duration = (time.perf_counter() - start) * 1000
indices = [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 19, 20]
np.testing.assert_array_equal(surveillance, raw[:, indices, :])
if not np.isfinite(surveillance).all():
    raise RuntimeError("Non-finite output values")

print("EXECUTION: PASS", flush=True)
print("Gather values: exact match to raw selected channels", flush=True)
print("Output finite: PASS", flush=True)
print(f"Single-run CPU time: {duration:.1f} ms (not a benchmark)", flush=True)
print("Production unchanged.", flush=True)
