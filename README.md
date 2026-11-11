# YOLO26 Surveillance — experimental beta

Surveillance-focused YOLO26s ONNX output-slicing experiment for Blue Iris and CodeProject.AI. This repository preserves the working test artifact and CodeProject.AI adapter recovered from the development archive dated 2026-11-11.

## Blue Iris quick test

If your existing Blue Iris AI setup supports custom ONNX models, place `YOLO26Surveillance/yolo26s-filtered-test.onnx` in your AI models folder, select it and test. This is not a full CodeProject.AI module installation.

## Experimental output mapping

The test graph exposes `surveillance_output` with shape **[1,12,8400]**: four bounding-box channels and eight class-score channels. The adapter currently maps `person, bicycle, car, motorcycle, bus, truck, cat, dog`. **Bicycle was intentionally retained for a graph-surgery experiment.** The intended final public surveillance model has seven classes: person, car, truck, bus, motorcycle, cat and dog.

The original full graph output remains present in the tested file. Graph-level output selection is confirmed; removal of all unused internal computation is not yet established. The graph-transformation script was not in the provided archive and must be published separately for source-to-model reproducibility.

## Environments already tested

- **INFINITY:** Windows Server 2025, CodeProject.AI 2.9.5, RX 9070 XT with ONNX Runtime DirectML. The YOLO26 Surveillance module appears as installed v0.1.0.
- **CHARON:** Blue Iris on NVIDIA Quadro RTX 4000. The experimental ONNX model was recognized and generated detections through the existing Blue Iris custom AI model interface.

Do not equate Blue Iris's DirectML provider on NVIDIA with CUDA inference. Timings depend on whether they measure pure inference, analysis or full round-trip time. Posted screenshots are observations, not controlled benchmarks.

## CodeProject.AI module

`YOLO26Surveillance/` contains the Python detector, ModuleRunner adapter, manifests, ONNX and dependency information. The isolated beta installer has not been validated against a fresh CodeProject.AI installation.

## Safe CHARON dry run

On a separate test directory, run `./scripts/prepare-test.ps1` with Python 3.12. It creates a local virtual environment, installs requirements, and validates the model using CPU inference. Do **not** replace working CodeProject.AI modules or disturb Blue Iris, Neolink or Mosquitto in production.

## Release checks

- Verify final seven-class channel mapping and Blue Iris labels
- Publish the graph-slicing script
- Verify source weights' redistribution license
- Perform the isolated clean installer test
- Retain the experimental `test` branch; no stable release is asserted

The full test artifact, source and example screenshots are being staged for this branch. Until they are committed, this README is documentation only.
