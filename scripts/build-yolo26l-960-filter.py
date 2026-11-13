"""Stage 3: add the documented surveillance Gather output to YOLO26l 960.

This does not change or overwrite either source model. Run from the
G: isolated environment after verifying the source SHA and raw export.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

root = Path(r"G:\YOLO26-L960")
reference = Path(r"G:\YOLO26-Migration\YOLO26Surveillance\yolo26s-filtered-test.onnx")
raw = root / "yolo26l.onnx"
target = root / "yolo26l-surveillance-960.onnx"
expected_hash = "DBC9326FC46819B02E27C842E9A3132291C5B0D424012610F061EDFC44422E17"

expected_names = [
    "person", "bicycle", "car", "motorcycle",
    "bus", "truck", "cat", "dog",
]
expected_coco_indices = [0, 1, 2, 3, 5, 7, 15, 16]
expected_selection = [0, 1, 2, 3] + [x + 4 for x in expected_coco_indices]

for path in (reference, raw):
    if not path.is_file():
        raise RuntimeError(f"Required input missing: {path}")
if target.exists():
    raise RuntimeError(f"Output already exists; refusing to overwrite: {target}")

actual_hash = hashlib.sha256(reference.read_bytes()).hexdigest().upper()
if actual_hash != expected_hash:
    raise RuntimeError(f"Reference SHA mismatch: {actual_hash}")

reference_graph = onnx.load(str(reference))
ref_outputs = {item.name: item for item in reference_graph.graph.output}
if set(ref_outputs) != {"output0", "surveillance_output"}:
    raise RuntimeError(f"Reference outputs changed: {sorted(ref_outputs)}")

producers = [
    node for node in reference_graph.graph.node
    if "surveillance_output" in node.output
]
if len(producers) != 1 or producers[0].op_type != "Gather":
    raise RuntimeError("Reference surveillance_output is not produced by one Gather")

ref_node = producers[0]
ref_axis = next(
    (attr.i for attr in ref_node.attribute if attr.name == "axis"), 0
)
if ref_node.input[0] != "output0" or ref_axis != 1:
    raise RuntimeError("Unexpected reference Gather source or axis")

ref_initializers = {
    item.name: numpy_helper.to_array(item)
    for item in reference_graph.graph.initializer
}
ref_indices = ref_initializers.get(ref_node.input[1])
if ref_indices is None:
    raise RuntimeError("Reference Gather indices not found as initializer")

actual_indices = [int(x) for x in ref_indices.reshape(-1)]
print("Reference Gather indices:", actual_indices, flush=True)
if actual_indices != expected_selection:
    raise RuntimeError(
        f"Reference class selection differs from documented order: {actual_indices}"
    )

model = onnx.load(str(raw))
inputs = {item.name: item for item in model.graph.input}
outputs = {item.name: item for item in model.graph.output}
if set(inputs) != {"images"} or set(outputs) != {"output0"}:
    raise RuntimeError("Unexpected raw ONNX inputs or outputs")

def dims(value):
    return [
        dim.dim_value
        for dim in value.type.tensor_type.shape.dim
    ]

if dims(inputs["images"]) != [1, 3, 960, 960]:
    raise RuntimeError(f"Wrong input dimensions: {dims(inputs['images'])}")
if dims(outputs["output0"]) != [1, 84, 18900]:
    raise RuntimeError(f"Wrong output dimensions: {dims(outputs['output0'])}")

index_name = "surveillance_class_indices"
output_name = "surveillance_output"
existing_names = {
    name for node in model.graph.node
    for name in list(node.input) + list(node.output)
}
existing_names.update(item.name for item in model.graph.initializer)
if index_name in existing_names or output_name in existing_names:
    raise RuntimeError("Surveillance graph names already exist")

model.graph.initializer.append(
    numpy_helper.from_array(
        np.asarray(actual_indices, dtype=np.int64),
        name=index_name,
    )
)
model.graph.node.append(
    helper.make_node(
        "Gather",
        inputs=["output0", index_name],
        outputs=[output_name],
        name="SurveillanceClassFilter",
        axis=1,
    )
)
model.graph.output.append(
    helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT, [1, 12, 18900]
    )
)

# Keep raw model metadata intact. Add separate surveillance mapping.
for key, value in {
    "surveillance_class_names": json.dumps(expected_names),
    "surveillance_coco_indices": json.dumps(expected_coco_indices),
}.items():
    metadata = model.metadata_props.add()
    metadata.key = key
    metadata.value = value

onnx.checker.check_model(model)
onnx.save(model, str(target))

verified = onnx.load(str(target))
onnx.checker.check_model(verified)
actual_outputs = {item.name: dims(item) for item in verified.graph.output}
expected_outputs = {
    "output0": [1, 84, 18900],
    "surveillance_output": [1, 12, 18900],
}
if actual_outputs != expected_outputs:
    raise RuntimeError(f"Saved outputs differ: {actual_outputs}")

print("=== STAGE 3 GRAPH VALIDATED ===", flush=True)
print("MODEL:", target, flush=True)
for name, shape in actual_outputs.items():
    print("OUTPUT:", name, shape, flush=True)
print("CLASSES:", ", ".join(expected_names), flush=True)
print("SHA256:", hashlib.sha256(target.read_bytes()).hexdigest(), flush=True)
print("Production model unchanged.", flush=True)
