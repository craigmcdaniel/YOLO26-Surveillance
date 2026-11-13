# YOLO26l 960 surveillance build — recovery log

**Status:** in progress, no YOLO26l ONNX artifact validated or deployed.
**Updated:** 2026-11-13
**Branch:** `test`
**Owner/source of truth:** this repository and its existing `README.md`, `docs/TEST_BRANCH.md`, and `YOLO26Surveillance/yolo26_surveillance.py`.

## Goal

Build and benchmark a separate YOLO26l **960 x 960** FP32 ONNX model for surveillance on INFINITY without changing the working YOLO26s / CodeProject.AI / Blue Iris production pipeline. Preserve the experimental eight-class order initially: `person, bicycle, car, motorcycle, bus, truck, cat, dog`. The future public release intends seven classes (omitting bicycle); do not silently change this during the comparison.

## Provenance and confirmed observations

| Evidence time | Source | Observation | Status |
| --- | --- | --- | --- |
| 2026-11-11 | GitHub `test/README.md` | Experimental artifact advertises a named output `surveillance_output` of `[1,12,8400]` and says the original full graph output is retained. | Documented; verify locally on source ONNX before graph surgery. |
| 2026-11-11 | GitHub `test/docs/TEST_BRANCH.md` | Reference `yolo26s-filtered-test.onnx` SHA-256: `DBC9326FC46819B02E27C842E9A3132291C5B0D424012610F061EDFC44422E17`. Graph transformation source script was absent from the archive. | Documented, not yet reverified against the current local file. |
| 2026-11-13 | INFINITY PowerShell output | Located reference at `G:\YOLO26-Migration\YOLO26Surveillance\yolo26s-filtered-test.onnx`. First enumerated ONNX output shape `[1,84,8400]`. | Observed. This does **not** disprove a second named `surveillance_output`; prior script looked only at output index 0. |
| 2026-11-13 | INFINITY PowerShell output | Isolated build environment `G:\YOLO26-L960\venv` has Python packages: Ultralytics 8.4.212, torch 2.15.0+cpu, ONNX 1.23.2, ONNX Runtime 1.32.0. | Observed. |
| 2026-11-13 | INFINITY PowerShell output | Pip completed successfully, but PowerShell emitted `NativeCommandError` on stderr output; initial monolithic script stopped before writing build script. | Observed; dependency installation need not be repeated. |
| 2026-11-13 | INFINITY PowerShell output | Second monolithic script wrote `G:\YOLO26-L960\build_yolo26l.py` but stopped with `Expected eight classes plus four box channels.` because it checked only `old.graph.output[0]`. | Observed; no YOLO26l export was performed by that run. |

## Root cause of aborted build

The script incorrectly assumed **the first ONNX output** must be the filtered `[1,12,8400]` tensor. The source documentation explicitly states that the original full output remains, so the observed `[1,84,8400]` first output may be expected. Check **all output names and shapes**, especially `surveillance_output`. Do not assume that a reduced output channel count proves upstream neural network pruning.

## Implementation constraints

- **No new build artifacts on C:** (including ProgramData): keep models, Python environment, scripts, caches and tests on `G:\YOLO26-L960` or approved non-system storage.
- Do not overwrite or redeploy working YOLO26s.
- Keep development operations modular: (1) reference audit and SHA check; (2) raw 960 export; (3) graph output filter; (4) shape and runtime validation; (5) DirectML performance / real-camera comparisons; (6) optional CodeProject.AI integration.
- Never guess output names / shapes or class indices. Capture actual exported graph first.
- The current detector `YOLO26Surveillance/yolo26_surveillance.py` hard-codes `INPUT_SIZE=640` and runs `session.run(["surveillance_output"], ...)`. Build a separate 960-capable test adapter; preserve stable source.
- Inferences about DirectML speed, detection quality, or actual removal of internal computation require specific tests. Pure ONNX shape validation is insufficient.
- Prior scripts are **experimental/failed**; do not treat them as a working recipe.

## Next stage: reference inspection (read-only)

1. Compute SHA256 of the local source ONNX and compare against documented hash.
2. Print *all* input/output tensor names and shapes, metadata, and nodes producing `surveillance_output`.
3. Recover class-channel mapping from the documented adapter plus graph evidence. Stop if evidence contradicts the eight-class order.
4. Log results below before beginning independent 960 export.

## Pending checkpoints

- [x] Reference hash and named output verified on INFINITY.
- [x] Actual YOLO26l export format and 960 output shape confirmed.
- [x] New named `surveillance_output` generated without overwriting source.
- [x] ONNX checker and ONNX Runtime CPU smoke test passed.
- [ ] 960 test adapter implemented.
- [ ] DirectML benchmark on RX 9070 XT completed.
- [ ] Detection quality compared on identical camera images.
- [ ] Source-to-model transformation script committed for reproducibility.

## Append new evidence here

Keep event time, source command/artifact, raw observed result, interpretation, correction, and SHA when relevant. Distinguish **documented**, **observed**, **inferred**, and **validated**.

## Checkpoint 1 — verified on INFINITY (2026-11-13)

User ran read-only reference verification and reported:

```text
SHA256: DBC9326FC46819B02E27C842E9A3132291C5B0D424012610F061EDFC44422E17
Matches documented artifact: True
INPUT: images [1, 3, 640, 640]
OUTPUT: output0 [1, 84, 8400]
OUTPUT: surveillance_output [1, 12, 8400]
PRODUCER: Gather / SurveillanceClassFilter
INPUTS: ['output0', 'surveillance_class_indices']
```

**Result:** authoritative artifact hash matches, source has **two outputs**, and the existing filtered output uses Gather of the raw output with a constant index tensor. The previous error was the build script validating only output 0. Reference inspection is COMPLETE; never repeat this failed assumption. Next stage is independently exporting *raw* YOLO26l at 960, preserving production and without graph edits. Ultralytics documentation says `nms=None` exports raw one-to-many outputs; `nms=False` selects the incompatible NMS-free head. Determine the actual output shape from the exported model before adding a Gather.

## Checkpoint 2 — raw YOLO26l 960 ONNX export verified (2026-11-13)

User supplied INFINITY console output:

```text
Environment: Ultralytics 8.4.212, Python 3.12.0, torch 2.15.0+cpu
Checkpoint: G:\YOLO26-L960\yolo26l.pt (50.7 MB)
YOLO26l fused: 188 layers, 24,807,420 parameters, 196.3 GFLOPs
ONNX: opset 18; export succeeded
Model: G:\YOLO26-L960\yolo26l.onnx
File length: 99,735,771 bytes (95.1 MB as reported by exporter)
INPUT: images [1, 3, 960, 960]
OUTPUT: output0 [1, 84, 18900]
ONNX checker: passed
```

**Status: Stage 2 verified.** The raw checkpoint was exported using `nms=None`, `dynamic=False`, `simplify=False`, `batch=1`, `device="cpu"`. This is *not* yet a surveillance-filtered ONNX. No CodeProject.AI deployment. Production YOLO26s unchanged. User prefers prioritizing detection quality at 960 resolution over resource minimization, subject to real-world benchmarks. Stage 3 should add a second `Gather`-produced `surveillance_output`, retain `output0` (matching the reference), preserve eight-class order from GitHub, validate output `[1,12,18900]` before runtime and DirectML benchmarking. Perform work in short independent steps; do not touch C:.

## Checkpoint 3 — surveillance Gather output validated (2026-11-13)

User ran `scripts/build-yolo26l-960-filter.py` on INFINITY and supplied successful output:

```text
Reference Gather indices: [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 19, 20]
MODEL: G:\YOLO26-L960\yolo26l-surveillance-960.onnx
OUTPUT: output0 [1, 84, 18900]
OUTPUT: surveillance_output [1, 12, 18900]
CLASSES: person, bicycle, car, motorcycle, bus, truck, cat, dog
SHA256: 471090e1e1304ef1e48f1801e48d8e91f574e09708171bed803f08eed5e4288a
Exit code: 0
Production model unchanged.
```

**Stage 3 verified:** ONNX structural checker passed, `surveillance_output` added as named Gather output with the exact class selection verified against the reference model. This establishes graph structure, **not** that ONNX Runtime or DirectML can execute it, nor accuracy/speed on images. Next: independently test ONNX Runtime inference and record CPU versus DirectML providers/latency; build an isolated resolution-aware adapter afterward. Preserve source and target models.

## Checkpoint 4 — CPU runtime smoke test passed (2026-11-13)

User ran `scripts/smoke-yolo26l-960.py` in the isolated INFINITY build environment:

```text
Available providers: ['AzureExecutionProvider', 'CPUExecutionProvider']
Input: images [1, 3, 960, 960]
Outputs: {'output0': [1, 84, 18900], 'surveillance_output': [1, 12, 18900]}
EXECUTION: PASS
Gather values: exact match to raw selected channels
Output finite: PASS
Single-run CPU time: 298.4 ms (not a benchmark)
Production unchanged.
Runtime test exit code: 0
```

**Stage 4 verified:** synthetic-input CPU execution passed; the filtered output exactly matches selection of the raw tensor, with no non-finite values. This does **not** establish image detection quality or DirectML performance. The installed `onnxruntime` build presently exposes Azure/CPU only; install/check `onnxruntime-directml` in the isolated `G:\YOLO26-L960\venv` as the next independently testable stage. Do not alter deployed CodeProject.AI or other Python environments. No claim that 298.4 ms represents steady-state CPU throughput.

## Checkpoint 5A — DirectML runtime installed (2026-11-13)

User provided INFINITY PowerShell output confirming successful installation of `onnxruntime-directml==1.24.4` into isolated `G:\YOLO26-L960\venv`, and reported:

```text
ONNX Runtime version: 1.24.4
Available execution providers: ['DmlExecutionProvider', 'CPUExecutionProvider']
=== STAGE 5A COMPLETE ===
```

**Verified:** DirectML provider is available in the isolated runtime. Not yet verified: successful model GPU execution, provider node allocation or GPU latency. Next stage: separate non-destructive DirectML model session creation, warmup and timed runs with `enable_mem_pattern=False`, `execution_mode=ORT_SEQUENTIAL` and optional CPU fallback controls, logging actual session providers. Keep all files on G: and production untouched.

**Future branch idea (deferred):** portable backend support for ROCm and native CUDA as alternatives to DirectML. Do not conflate available runtime providers with tested performance or commit these backends into current production work.
