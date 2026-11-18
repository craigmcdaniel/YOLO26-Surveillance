# ARGUS model-analysis workbench — proposed, not deployed

Status: **ARGUS boot and host-level read-only audit completed 2026-11-18; new FiftyOne/Albumentations analysis environment not yet deployed**. See [PANOPTES milestone index](PANOPTES_MILESTONES.md) for the already-working AMD ROCm-in-Docker, training, TensorBoard and CVAT environments. Do not mistake host Python lacking torch for a missing container runtime. Keep production INFINITY CodeProject.AI, CHARON Frigate, and Blue Iris unchanged.

## Objective

Build a repeatable, source-image-preserving environment to study YOLO26 surveillance inference under input transformations and postprocessing changes, using the existing test branch as the source of truth.

- **ARGUS (Ubuntu, INFINITY GPU-partitioned guest):** proposed isolated experiment host, subject to GPU/runtime/storage verification.
- **INFINITY (Windows):** currently hosts working YOLO26 inference via CodeProject.AI and DirectML; keep production untouched.
- **CHARON:** Frigate and Blue Iris. Do not change recording, detection or authentication settings for these experiments.
- **THOTH:** optional later telemetry sink, not required for initial tests.

## Existing experimental baseline (2026-11-18 UTC)

On INFINITY, the new [RGB versus CLAHE script](../scripts/benchmark-clahe-yolo26s.py) ran successfully against 20 images from the existing test corpus:

- Source checkout: `G:\YOLO26-L960\clahe-repo`, test branch.
- Image corpus: `G:\YOLO26-L960\test-images`.
- Local output: `G:\YOLO26-L960\results\clahe-experiment\run-20261118-023145Z`.
- Model: the experimental YOLO26s-640 eight-class filtered ONNX; confidence threshold 0.25.
- CLAHE: OpenCV Lab luminance only, clip limit 2.0, 8x8 tiles.
- Result counts: **RGB 4, CLAHE 4** across 20 images (both counts include incorrect detections).
- `1 (1).jpg` **is a squirrel on a feeder**, not a cat. Original RGB predicted one *cat* (0.6495), box (77,158)-(424,351). CLAHE predicted two *cats* (0.5237 and 0.3089), boxes (228,159)-(424,350) and (77,162)-(259,349). This is a known **false-positive/box-fragmentation** case; CLAHE did not discover a second cat.
- `20260801_113807.jpg`: *cat* confidence 0.6420 RGB vs 0.5850 CLAHE; slight box shift.
- `20260801_113824.jpg`: *bicycle* confidence 0.7024 RGB vs 0.7253 CLAHE; nearly identical box.
- `20260816_115231.jpg`: tiny *car* box at 0.2594 with RGB, absent after CLAHE; ground truth unverified.
- CLAHE preprocessing overhead on the four reviewed frames: 12.774–92.257 ms. These are full-resolution image transformations, so do not conflate them with model inference timing or assume a fixed per-frame cost.

**Interpretation:** Transformation changes detector behavior, but detection count or confidence alone is not accuracy. Ground truth and false-positive analysis are needed; results do **not** justify enabling CLAHE in production.

## Candidate analysis stack

Audit before installation; isolate installations from PANOPTES training and all existing services.

- **FiftyOne:** browse image sets, ground-truth labels, detections, boxes, and compare named experiment runs.
- **Albumentations:** reproducible gamma/contrast/CLAHE/blur/noise/channel transforms.
- **OpenCV:** Lab color transforms, masks, background isolation.
- **ONNX Runtime:** inspect and infer with the filtered YOLO26 model; verify GPU execution providers *on ARGUS*, do not assume Windows DirectML is available on Ubuntu. CPU-only pilot is acceptable.
- Existing Python detector and per-class NMS remain baseline code; treat any postprocessing experiments as isolated variants.

## Proposed instrumentation (not implemented yet)

1. Save provenance: source image relative path and hash, transformation parameters, model hash, runtime/provider, threshold, timestamps, software versions.
2. Compare *identical source frames* using RGB baseline, luminance CLAHE, gentler CLAHE, gamma, channel permutations/isolation, grayscale replicated to RGB, masks/foreground composites, edge enhancement and inversion.
3. For alpha experiments, construct an actual **foreground mask**, then composite into 3-channel RGB. The ONNX model accepts [1,3,640,640], not 4-channel RGBA; a fully opaque alpha channel carries no additional information.
4. Capture the eight raw class scores per selected candidate **before argmax**, candidate boxes before confidence filtering and NMS, and final boxes after NMS. Bound logs to avoid enormous outputs.
5. Evaluate false positives, duplicate/fragmented boxes, class/box stability, missed ground-truth objects, and separate preprocessing/inference/postprocessing latency.
6. Prioritize verified squirrel/feeder example as a regression fixture; the eight-class model has no squirrel class. Evaluate whether enhanced images spuriously produce additional cat boxes.
7. Preview overlays and export CSV/JSONL; label frames with verified ground truth before claiming improvements.

## ARGUS readiness audit — next session

**Completed host audit 2026-11-18:** Ubuntu 24.04.5/kernel 7.0.0-43, `/dev/dxg` and `dxgkrnl` present, CVAT workers running, ~4.2 GiB RAM available, 117 GB local disk free and photostore mounted. Container GPU compute had *already* passed with AMD ROCDXG on 2026-11-16. Remaining: inspect existing Docker/PANOPTES environments using the existing sudo access pattern before adding anything. Confirm a non-system experiment destination before any installation or copying. Avoid INFINITY C: for project files or staging.

After audit: choose isolated venv/container and test small, labeled batch. Do not install FiftyOne/Albumentations into existing training or production Python environments by default. Record environment details and the experiment run in Git.

## Baseline code pointers

- [YOLO26s inference and preprocess](../YOLO26Surveillance/yolo26_surveillance.py): PIL RGB conversion, letterbox 640, float32/255, NCHW, ONNX Runtime; raw scores reduced by argmax/NMS.
- [YOLO26s CPAI adapter](../YOLO26Surveillance/detect_adapter.py): receives image through SDK and delegates inference.
- [RGB/CLAHE pilot](../scripts/benchmark-clahe-yolo26s.py): isolated paired image benchmark, keeps source and production unchanged.
- [640 versus 960 experiment](../scripts/benchmark-yolo26-640-vs-960.py): prior benchmark infrastructure.

## Operational boundaries

- **No deployment to production implied by these notes.**
- No inference adapter, ONNX model, Frigate YAML, Blue Iris, or ARGUS service has been modified by writing this document.
- Repo state reflects Git snapshots; audit actual deployed copies before any later modifications.
