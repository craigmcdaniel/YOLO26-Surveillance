# YOLO26 Surveillance: training, pruning, and portable backends

Status: proposed experiment, 2026-11-13. No new eight-class trained checkpoint, MIGraphX artifact, ROCm run, CUDA or TensorRT run is claimed yet.

## Why this is the next experiment

The deployed YOLO26s-640 and YOLO26l-960 proof-of-concept ONNX models preserve all original 80 class logits and add a Gather operator to expose eight surveillance classes. This is output filtering, not computational pruning. The working models are preserved as baselines.

The goal is **a learned eight-class head** using surveillance-relevant annotated images; then optimize separately for each runtime. The existing eight labels/order are:
0 person, 1 bicycle, 2 car, 3 motorcycle, 4 bus, 5 truck, 6 cat, 7 dog.

## Training dataset

- Gather actual camera images of people, vehicles, cats through glass, small/distant subjects, partial occlusion, night/IR scenes, rain/snow, and empty scenes; include diverse crowded street images where there is appropriate permission to train with them.
- The existing 862-file benchmark corpus has images and **model predictions**, not verified bounding-box ground truth. Model-generated boxes may seed annotations, but **human correction/approval is required** before treating them as labels.
- Include genuine negative frames with zero objects and verify labels against each image. Do not assume that failed CD insert HTTP requests are corrupt images.
- Split by camera, location, and/or capture day/session to avoid near-identical frames appearing in both training and validation. Preserve a genuinely untouched deployment test set. Keep provenance, annotation versions, label schema and source filenames.
- Record per-class precision, recall, mAP, and false-alert performance, especially cats and nighttime scenes; evaluate missed objects and false positives rather than maximizing detection count.

## Model construction

- Start with pretrained YOLO26 .pt weights, not an ONNX Gather artifact. The Ultralytics fine-tuning workflow accepts a custom eight-class dataset; compatible backbone/neck weights are transferred, while the class-dependent detection head is adapted to the new class count.
- Export and inspect an actual eight-class model: expected one-to-many output [1, 12, N] at fixed image size, rather than [1, 84, N] plus Gather. Confirm the parameter count/graph and actual latency before describing computational savings; reducing the number of class outputs alone doesn't guarantee substantial end-to-end speedups if the backbone dominates.
- Separately evaluate YOLO26's one-to-one end-to-end inference head (NMS-free shape such as [1, 300, 6]) and adjust adapters intentionally; do not feed that schema into the legacy raw-output decoder.

## Backend outputs

Single trained .pt checkpoint => portable, validated ONNX master(s) => platform-specific:
- ONNX Runtime DirectML (Windows AMD/NVIDIA) — working baseline.
- AMD ROCm/MIGraphX optimization or ONNX Runtime MIGraphX EP for supported Linux AMD GPU/driver stacks, plus architecture-targeted compiled artifacts where supported; compiled binaries are **not** universally portable.
- NVIDIA CUDA / TensorRT-specific build and profiling.
- CPU ONNX Runtime fallback.

Test all exports against the same held-out images and expected boxes; compare accuracy, latency, warm-up, memory usage and concurrent GPU contention. Do not claim cross-backend numerical identity without measuring tolerances.

## Deployment and provenance

- Continue development, datasets, checkpoints, build scripts, and benchmark artifacts on G:; packaged CodeProject.AI application modules/runtime may install under C:\Program Files\CodeProject\AI. H: is interim slow storage, not a deployment destination.
- Existing production YOLO26s module and live Blue Iris YOLO26l module remain unchanged. New training and compiled builds use separate names, branches and routes.
- Document configuration hashes, source version, capture/annotation time, class list, dataset split, training parameters, export flags, actual environment and measured results in GitHub.
- Preserve the original source images and old benchmarks.

## References

- AMD's ROCm Ultralytics announcement: https://rocm.blogs.amd.com/ecosystems-and-partners/ultralytics-amd/README.html
- Ultralytics YOLO26 fine-tuning: https://docs.ultralytics.com/guides/finetuning-guide
- Ultralytics YOLO26 head/output modes: https://docs.ultralytics.com/models/yolo26/
- MIGraphX: https://rocm.docs.amd.com/projects/AMDMIGraphX
- ONNX Runtime MIGraphX provider: https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html

## INFINITY operating-system correction (2026-11-13)

User confirms INFINITY is **currently booted into Windows 11**; the separate **Windows Server 2025 installation still exists and remains available**. Earlier feasibility discussion incorrectly treated Server 2025 as the active OS. The synthetic-data and GPU-compute proof of concept should target the current Windows 11 boot, with Radeon RX 9070 XT 16 GB and 32 GB system RAM. Do not propose reinstalling or migrating OSs. Keep Blender scenes, assets, generated images, annotations, training checkpoints and test artifacts on G:, not C: or H:. The server installation and working CodeProject.AI deployment should remain untouched during the experimental phase. Confirm actual versions and GPU backend support before claiming runtime success.
