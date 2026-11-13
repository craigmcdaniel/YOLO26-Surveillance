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
- [x] Initial DirectML concurrent-load runtime test on RX 9070 XT completed (full GPU allocation not verified).
- [x] Initial prediction-count and latency comparison completed on identical camera images (correctness review pending).
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

## Checkpoint 5B — concurrent workload conditions (2026-11-13, before benchmark results)

User reports that while testing YOLO26l-960, **Enshrouded is running at demanding graphics settings** and **another server is actively performing inference**. Treat the forthcoming DirectML latency values as measurements under this concurrent workload, **not** idle-system or isolated-GPU benchmarks. The statement does not establish whether the other inference workload runs on INFINITY's GPU, another host's GPU, or over the network; verify resource contention before attribution. Record workload state alongside latency and compare against a later idle run if useful. Do not alter game, existing model, or production services for this test.

## Checkpoint 5B — DirectML concurrent-load inference passed (2026-11-13)

INFINITY user-reported benchmark results with Enshrouded running at demanding settings and another server active in inference:

```text
Available providers: ['DmlExecutionProvider', 'CPUExecutionProvider']
Session providers: ['DmlExecutionProvider', 'CPUExecutionProvider']
Output: (1, 12, 18900)
3 warm-up runs, 10 measured synthetic-input session.run calls
Median: 27.03 ms
Minimum: 22.70 ms
Maximum: 31.69 ms
Exit code: 0
Production unchanged
```

Task Manager screenshot recorded around the test:
- AMD Radeon RX 9070 XT, 3D utilization 96%; Compute 0 graph showed 0% at screenshot instant (not a definitive engine-assignment trace).
- Dedicated VRAM 10.7 / 16.0 GB; total GPU memory 12.1 / 31.8 GB; shared 1.3 / 15.8 GB.
- GPU temperature 51°C; system RAM 21.6 / 31.6 GB (68%); CPU around 10%.
- Driver version shown 32.0.32015.2008, dated 2026-09-28.
- Enshrouded running under heavy settings; existing inference server also active. Whether its computation shared the same GPU is not established.

**Status:** DirectML provider available and ONNX Runtime session successfully ran the filtered 960 model. These are short, synthetic-input, whole-session times, **not** end-to-end Blue Iris/CodeProject.AI latency, per-op GPU provenance, or real-world detection quality. CPU fallback has not been ruled out; assess provider assignment using profiling if needed. Do not infer a sustained maximum FPS from ten samples. Next stage: separate 960-aware adapter and real camera-image validations. Keep production YOLO26s untouched.

## Stage 6 — paired real-image benchmark prepared (2026-11-13)

User reports approximately 1,000+ original test images under `G:\YOLO26-L960\test-images`, including oddball subjects/file types; user reports no duplicate images. Image contents are not yet independently inventoried. Do not impose expected subjects, labels or per-camera grouping.

Script committed: `scripts/benchmark-yolo26-640-vs-960.py`. It is an isolated, non-destructive, resumable paired-image test. It hashes/verifies both known ONNX models; loads both via DirectML (CPU fallback permitted); letterboxes each original image at the model's native resolution; processes eight surveillance classes at 0.25 confidence with per-class NMS IoU 0.45; writes durable JSONL for every successful image/model evaluation, plus decode/inference errors; refreshes a CSV aggregate periodically; and does not overwrite the reference or modify production. Supports JPG/JPEG/JFIF/PNG/WEBP/BMP/TIFF/GIF (first frame), including nested directories. Performance figures are `session.run()` only, do not include image decode, preprocessing or postprocessing. This is a workload- and dataset-specific comparison; differences are not a labeled accuracy/recall measurement without image ground truth.

**Caveats:** keeping two GPU sessions resident may compete with Enshrouded for available VRAM, and CPU fallback could affect timing. Outcomes are not yet validated; no claims about actual accuracy or stable completion. Results path `G:\YOLO26-L960\results\stage6`. No deduplication/delete operations; no source image alterations.

## Checkpoint 6 — completed paired real-image benchmark (2026-11-13)

User supplied completed console output for `scripts/benchmark-yolo26-640-vs-960.py` on INFINITY. 862 image files were detected and **862 complete matched image/model pairs** were processed in 71.8 seconds. Both DirectML+CPU-fallback sessions reported active. Production unchanged. Benchmark exited with code 0. One non-fatal Python `SyntaxWarning: invalid escape sequence '\\Y'` arose from the leading documentation string; no runtime failure. Structured output files: `G:\YOLO26-L960\results\stage6\results.jsonl` and `summary.csv`.

| Class | YOLO26s-640 | YOLO26l-960 |
| --- | ---: | ---: |
| person | 347 | 407 |
| bicycle | 15 | 10 |
| car | 381 | 375 |
| motorcycle | 1 | 6 |
| bus | 0 | 0 |
| truck | 151 | 191 |
| cat | 12 | 14 |
| dog | 5 | 11 |
| **Total** | **912** | **1014** |

YOLO26s-640: 518 images with >=1 detection, median `session.run` 4.61 ms, mean 5.08 ms.
YOLO26l-960: 509 images with >=1 detection, median `session.run` 24.07 ms, mean 24.29 ms.
YOLO26l had 102 more predictions (+11.18%), 9 fewer images with any prediction, and median runtime approximately 5.22x YOLO26s. These are paired workload measurements at a shared 0.25 score cutoff and classwise NMS IoU 0.45. Counts are **not** proof of recall, precision, correctness, or verified false positives; labels of source images have not been manually established. Benchmark set contains diverse camera snapshots and user-described oddball images, with no duplicate processing reported. The output JSONL can be analyzed to identify disagreements and review corresponding images. **Next:** generate a per-image disagreement report and inspect sample images to distinguish improved true positives from new false positives; then develop/validate a resolution-aware adapter. Preserve production and raw source images.

## Stage 7 — prepare for live CodeProject.AI test (2026-11-13)

User explicitly prioritizes proceeding to a live test during cleanup, rather than more offline image disagreement analysis. The existing 640 production module and its files must remain available for rollback.

The `test` branch now contains a **separate experimental** `experiments/yolo26l-960/yolo26_surveillance.py` derived from the current detector. Differences from original: reads static BCHW model input dimensions from ONNX Runtime, validates named `surveillance_output` shape [1,12,N], and uses model-derived height/width for aspect-preserving resize and letterboxing. Original `YOLO26Surveillance/yolo26_surveillance.py` unchanged. This file has not yet been executed in an isolated real-image test.

Next: stage the experimental adapter under `G:\YOLO26-L960\live-test`, verify DML inference on a real test image, and inventory the **actual running CodeProject.AI service/module locations and launch configuration** before deciding how to switch model paths or register an independent test endpoint. Do not blindly patch live service files; source may reside on C:, but new work stays on G:/H:. When switching, preserve original settings, module and model and ensure rollback. No Frigate changes; preserve auth.enabled=false, recording, and existing production alert routing.

## Stage 7 — first live Blue Iris YOLO26l-960 detection confirmed (2026-11-13)

User supplied live Blue Iris AI result for capture `Camera02.20261113_142837Z.11316.16385-0.dat` (filename timestamp 2026-11-13 14:28:37 UTC):

```json
{
  "api": "custom",
  "found": {
    "success": true,
    "error": null,
    "message": "",
    "predictions": [
      {
        "category": 1,
        "label": "person",
        "confidence": 0.830482721328735,
        "x_min": 150,
        "x_max": 530,
        "y_min": 0,
        "y_max": 469
      }
    ],
    "count": 1,
    "moduleId": "yolo26l-surveillance-960.onnx",
    "executionProvider": "ONNXRunTime: DirectML: NVIDIA Quadro RTX 4000 with Max-Q Design",
    "analysisRoundTripMs": 169
  }
}
```

**Confirmed:** The Blue Iris-side AI interface successfully invoked the model named `yolo26l-surveillance-960.onnx` through its custom API and returned a person detection at 83.05% confidence; execution provider reports DirectML on NVIDIA Quadro RTX 4000 with Max-Q Design. This is a **live integration milestone** independent from INFINITY's earlier RX 9070 XT test.

**Do not conflate:** 169 ms `analysisRoundTripMs` is Blue Iris-reported end-to-end analysis round trip, while INFINITY's 24.07 ms is median `session.run` inference on synthetic/collected test frames. The GPU and timing scope differ. Image itself and ground-truth person presence have not been independently inspected; confirmed classification is the returned model prediction, not yet human-verified accuracy. No change to Frigate, Node-RED or production route was required to report this result. Preserve this capture identifier as first live evidence.

### Live Camera02 configuration clarification (2026-11-13)

User clarified that this first live YOLO26l-960 detection was **straight inference with a Blue Iris `exists` filter**. Interpret the result as direct model inference evaluated against an object-presence condition, **not** proof of a conventional Blue Iris motion-triggered detection and **not** an additional separate AI confirmation pass. The exact filter expression and triggering cadence were not supplied; do not invent them. This clarification supersedes any framing that presumes motion triggers or a double-check pipeline at the Blue Iris side. The actual person detection at confidence 0.8304827213 and 169 ms analysisRoundTrip remains as recorded above.

### Camera02 AI object rule screenshot — actual filter configuration (2026-11-13)

Screenshot provided by user confirms the following settings in Blue Iris's `AI object` dialog: Enabled checked; Label/s `*` (wildcard); minimum confidence **50%**; minimum time **0 seconds**; Sense **Exists** selected (not Moving, Static, or Existence change); Where **Anywhere** selected (not zones or center rectangle); Action **Trigger**; firing mode **Single fire**; memo/label blank. Profile buttons are displayed but individual effective scheduling/activation cannot be reliably inferred from screenshot alone.

Consequence: the reported 83.048% person detection meets the 50% confidence condition. The object-presence rule does not require a detected object to move; this screenshot itself does not independently establish every upstream image acquisition/trigger mode or full trigger cadence. Since labels are unrestricted (`*`), any class returned by the eight-class ONNX model can satisfy this rule. Preserve this as exact live configuration evidence rather than substituting an assumed narrower person filter.

### Live YOLO26s vehicle result — Cam1 (2026-11-13)

During side-by-side live Blue Iris testing, user supplied result for `Cam1.20261113_130001Z.1480626.17-0.dat` (filename timestamp 2026-11-13 13:00:01 UTC). Custom AI result: success true; `moduleId=yolo26s-filtered-test.onnx`; `executionProvider=ONNXRunTime: DirectML: NVIDIA Quadro RTX 4000 with Max-Q Design`; `analysisRoundTripMs=323`. One `car` prediction: category 2, confidence **0.892409920692444**, box x=[538,727], y=[223,298]. This is **the smaller 640 model**, not YOLO26l-960. Earlier 960 result was a `person` from **Camera02** with 169 ms round trip. These examples differ in camera, subject and capture timestamp and therefore cannot establish relative model latency or correctness. The user reports the two YOLO models are live side by side; same-frame evaluation remains to be demonstrated.

### Cross-camera same-car clarification — Cam1 and Willard (2026-11-13)

**User explicitly confirms that the Cam1 YOLO26s car and Willard YOLO26l car are the same physical vehicle.** Preserve the user's identification rather than treating this as an unknown association. The camera images are different viewpoints and therefore **not a same-frame controlled comparison**, despite both capture identifiers containing `20261113_130001Z`.

- Cam1: `Cam1.20261113_130001Z.1480626.17-0.dat`, YOLO26s-640, car, confidence 0.892409920692444 (89.24%), bounding box (538,223)-(727,298), analysisRoundTrip 323 ms.
- Willard: `Willard.20261113_130001Z.5618511.17-0.dat`, YOLO26l-960, car, confidence 0.915965259075165 (91.60%), bounding box (1358,316)-(1834,487), analysisRoundTrip 237 ms.
- Both report ONNX Runtime DirectML on NVIDIA Quadro RTX 4000 with Max-Q Design.

**Confirmed cross-camera detection:** both models reported a car identified by the user as the same vehicle. Differences in camera viewpoint, scale, processing and capture mean the differing confidence and latency do **not** demonstrate one model is superior or faster.

## Stage 8 — CodeProject.AI 960 side-by-side deployment package (2026-11-13)

User requested build/deployment for CodeProject.AI while Blue Iris live A/B continues. The `test` branch now contains a second, independently identified CodeProject.AI module under `deploy/codeproject-ai/YOLO26LSurveillance960/`, including a 960-aware `yolo26_surveillance.py`, the SDK `detect_adapter.py`, distinct module ID `YOLO26LSurveillance960`, unique queue `yolo26l_surveillance_960_queue`, endpoint `vision/yolo26l-surveillance-960`, and copied installation requirements. `AutoStart=false` intentionally avoids unverified automatic activation. The default ONNX model filename is `yolo26l-surveillance-960.onnx`; the existing `YOLO26Surveillance` module and its `vision/yolo26-surveillance` route are unchanged.

`scripts/deploy-yolo26l-960-cpai.ps1` checks the locally built model SHA-256 (`471090e1e1304ef1e48f1801e48d8e91f574e09708171bed803f08eed5e4288a`), downloads versioned deployment files to `G:\YOLO26-L960\cpai-deploy\YOLO26LSurveillance960`, verifies JSON ID, route and disabled AutoStart, copies the unmodified ONNX file, and side-installs the candidate under `H:\CodeProjectAI\modules\YOLO26LSurveillance960` **only if the expected H: module parent exists and the candidate target does not already exist**. No overwrite, process stop/restart, change to the production module, Frigate, Blue Iris, or Node-RED. Does not claim a live endpoint yet. Returns the detected CodeProject.AI Windows service details for next stage.

**Pending before live activation:** confirm the actual CPAI service's installed module discovery root, its Python 3.12 runtime and SDK/dependency availability, verify the separate module is recognized by the server (and configured to start), then send one real image to the new route. Activation and live API response remain unverified. The module components are repository builds; their execution on the installed CPAI runtime has not yet been validated. Rollback: disable/remove the *new* candidate module after confirming existing service is stable; existing production module is never overwritten.

### Actual INFINITY CPAI installation confirmed; deployment correction (2026-11-13)

User-provided PowerShell inventory:
- `C:\Program Files\CodeProject\AI\` contains downloads, logs, modules, runtimes, scripts, SDK, Server, utils.
- Existing live module: `C:\Program Files\CodeProject\AI\modules\YOLO26Surveillance`.
- Windows service `CodeProject.AI Server` is **Running**, binary `C:\Program Files\CodeProject\AI\Server\CodeProject.AI.Server.exe`.
- Previous deployment run **stopped at missing `H:\CodeProjectAI\modules`** before any module install; previous H: location was stale. The user clarified OS drive C: is acceptable for installed applications, but project development/model builds and artifacts remain on G:.

Corrected `scripts/deploy-yolo26l-960-cpai.ps1` on branch `test` now checks actual `C:\Program Files\CodeProject\AI\modules`, installs **only six candidate module code/config/install files** into a new `YOLO26LSurveillance960` directory, and deliberately **does not copy ONNX to C:**. New test `detect_adapter.py` defaults to `G:\YOLO26-L960\yolo26l-surveillance-960.onnx` (override via `YOLO26_SURVEILLANCE_MODEL` if needed). Staging, source model, source verification, archives remain G:. Existing YOLO26s module unchanged. This corrected deployment has **not yet been run or verified**. AutoStart remains false and service restart is not performed; new API route is not yet live. Need first verify deployment output and Python runtime integration before activation.

### Stage 8 deployment succeeded — files installed (2026-11-13)

User ran corrected deployment in elevated INFINITY PowerShell (`PS C:\WINDOWS\system32>`) and supplied output confirming:
```text
=== MODULE FILES INSTALLED ===
Directory: C:\Program Files\CodeProject\AI\modules\YOLO26LSurveillance960
Route: vision/yolo26l-surveillance-960
ONNX stays on G: G:\YOLO26-L960\yolo26l-surveillance-960.onnx
AutoStart: false
Service not restarted; route not yet active.

Name: CodeProject.AI Server
State: Running
PathName: C:\Program Files\CodeProject\AI\Server\CodeProject.AI.Server.exe
```
All six candidate module files were fetched into the G: staging directory and side-installed on C: without modifying production `YOLO26Surveillance`. **Do not conflate successful file deployment with a loaded or runnable service route**. CodeProject.AI docs say module settings are discovered on server startup; new module is `AutoStart=false`. Before restart, compare currently-working module's runtime settings with new candidate and inspect available Python3.12 runtimes/SDK on the machine. Because running production detection may be in use, avoid an unplanned service restart. Once prerequisites are verified, start/discover test module and validate POST /v1/vision/yolo26l-surveillance-960 with an image.

### Stage 8 runtime inventory verified (2026-11-13)

User's live INFINITY read-only PowerShell output:
- Production module `YOLO26Surveillance`: AutoStart True; FilePath `detect_adapter.py`; Runtime `python3.12`; RuntimeLocation `Local`; queue `yolo26_surveillance_queue`; Parallelism 1.
- Candidate module `YOLO26LSurveillance960`: AutoStart False; same FilePath/Runtime/RuntimeLocation; **distinct** queue `yolo26l_surveillance_960_queue`; Parallelism 1.
- CodeProject.AI root runtime `C:\Program Files\CodeProject\AI\runtimes\bin\windows\python312\python.exe` exists; listing also includes Python 3.12 standard-library `Lib\venv\scripts\nt\python.exe`, which is a venv template and not proof of deployed venv. Separate G: development venv Python exists.
- Presence of DirectML, SDK, Pillow, numpy in the **actual application runtime** remains unconfirmed, as does recognition of candidate by running CPAI process. Do not restart production until confirmed. Next read-only check runs import tests on available application Python executable(s) and queries module status API.

### Stage 8 runtime-import check failed; server module inventory confirms 960 not loaded (2026-11-13)

User uploaded original 299-line INFINITY console output for read-only Python import / CodeProject.AI module status checks. Python `C:\Program Files\CodeProject\AI\runtimes\bin\windows\python312\python.exe` was invoked with a PowerShell here-string as `-c` argument; **the diagnostic script itself failed parsing** with `SyntaxError: unterminated string literal`, showing malformed `print(name, OK, getattr(module, __version__, "))`. This is a PowerShell/Python quoting failure in assistant-provided diagnostic, **not evidence of missing Python packages**. Correct via writing an actual `.py` file under G: and executing that path; avoid inline `-c` quoting.

`GET http://localhost:32168/v1/module/list/status` succeeded, `success=true`, `code=200`, hostname `INFINITY`. Registered statuses list showed `ObjectDetectionYOLOv5Net`, `YOLO26Surveillance`, `FaceProcessing`, `ObjectDetectionYOLOv8`, and `ObjectDetectionYOLOv5-6.2`, **but not `YOLO26LSurveillance960`**. The new module's files are installed; the running server has not loaded its configuration yet. It is currently `AutoStart=false`.

Production `YOLO26Surveillance` status `Started`; started 2026-11-12 14:42:51Z; lastSeen 2026-11-13 14:50:59Z; requestCount 300296; successfulInferences **300278**, failedInferences **0**, averageInferenceMs **4.812630295925776**; numItemsFound **137262**, histogram person 83511, truck 25265, car 20197, bus 2148, dog 4207, cat 1669, bicycle 242, motorcycle 23. Reported inferenceDevice DirectML, inferenceLibrary ONNX Runtime; `canUseGPU` string `false` is in SDK status despite DirectML claim, so don't infer actual GPU-node coverage solely from this flag.

Next: clean read-only Python 3.12 dependency/import/provider check via saved G: script. Then plan a controlled CPAI service restart if needed for module discovery; doing so **will briefly interrupt existing production inference**, so do not describe it as zero-impact. Verify new module appears before activating and test endpoint. Preserve current working production model.

### Python runtime diagnostic outcome (2026-11-13)

User ran the corrected saved-G: diagnostic against `C:\Program Files\CodeProject\AI\runtimes\bin\windows\python312\python.exe` (Python 3.12.4) and all four imports failed with `ModuleNotFoundError`: `numpy`, `PIL`, `onnxruntime`, `codeproject_ai_sdk`. This only establishes that the **base Python 3.12 interpreter's import path lacks these packages**. It does **not** establish the actual running YOLO26Surveillance module environment lacks them; the current production module has more than 300k successful inferences, suggesting it runs using a separate virtual environment or an alternate interpreter/package path. Next step must **inspect live python process command lines and targeted runtime virtual-environment paths, read-only**, before any installation or CodeProject.AI restart. Avoid installing packages into the base application Python or modifying/restarting the production YOLO26s runtime speculatively.
