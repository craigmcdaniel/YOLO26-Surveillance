# PANOPTES / ARGUS milestone index

**Updated 2026-11-18 UTC.** This index reconciles historical console evidence and previously standalone checkpoint documents with the `YOLO26-Surveillance` `test` branch. It is **documentation only**, not a deployment, test rerun, or claim of live service health.

## Milestone chronology

| When (UTC) | Component | Evidence / state |
|---|---|---|
| 2026-11-13 | YOLO26L-960 inference / CPAI tests on INFINITY | Existing [build log](YOLO26L_960_BUILD_LOG.md) through Stage 9; 862 requests, 859 successful. Independent from ARGUS ROCm training. |
| 2026-11-16 | ARGUS GPU-P + AMD ROCDXG | **Verified:** Hyper-V /dev/dxg bridge, container GPU matrix multiply, 64/64 YOLO26s Carport inference, one COCO8 training epoch. [GPU-P/ROCm checkpoint](PANOPTES_GPU_P_ROCM_2026-11-16.md). |
| 2026-11-16 | TensorBoard 2.20.0 | **Installed and event inspection passed:** 13 scalar tags from one epoch; dashboard not confirmed launched at that checkpoint. [TensorBoard checkpoint](PANOPTES_TENSORBOARD_2026-11-16.md). |
| 2026-11-16 (later) | CVAT v2.82.0 | **Active** on ARGUS (subsequently reconfirmed by process audit 2026-11-18). Earlier Task 2 had zero attached frames; later tasks used existing import scripts. [CVAT checkpoint](PANOPTES_CVAT_CHECKPOINT_2026-11-16.md). |
| 2026-11-16 (later) | IPCam-Combined inference | **Verified:** 862 images, 675 predictions, 0 inference failures; labels include squirrel (3). Source predictions on INFINITY G: and CVAT import preparation on ARGUS. Whether particular annotations were reviewed is separate. |
| 2026-11-18 | CLAHE RGB test | **Verified:** 20-image pilot, RGB 4 predictions vs CLAHE 4, but individual differences; squirrel photo misclassified as cat and fragmented under CLAHE. [ARGUS analysis plan](ARGUS_MODEL_ANALYSIS.md). |
| 2026-11-18 | ARGUS current environment | Booted Ubuntu 24.04.5 kernel 7.0.0-43, 12 vCPU, 7.7GiB RAM, 117GB root free, /mnt/photostore mounted, /dev/dxg present, CVAT Django/Uvicorn workers running. Host system Python has no torch/onnxruntime/cv2 etc. This **does not** negate verified ROCm GPU execution *inside the AMD container*. |

## Architecture / ownership

- **INFINITY (Windows 11):** AMD RX 9070 XT, physical Hyper-V host, DirectML/CodeProject.AI production inference, G: experimental model assets and benchmarks, gaming workloads.
- **ARGUS (Ubuntu Hyper-V VM):** `/home/argus/gpu-pv-lab`, experimental GPU-P + ROCDXG compute in the *AMD container*, YOLO26 training, CVAT, separate TensorBoard venv.
- **CHARON:** Frigate/Blue Iris production camera services; do not alter for ARGUS experiments.
- **THOTH:** monitoring and MQTT / Node-RED; separate from analysis.
- **Canonical documentation**: this repository `test` branch. The Windows `G:\ARGUS\gpu-pv-lab\mirage-gpupv` is an upstream third-party clone, **not** the project documentation repository.

## Verified links and access

- CVAT dashboard: http://192.168.7.240:8080/ (previous access logs; availability may change).
- CVAT tasks: http://192.168.7.240:8080/tasks
- CVAT task 3: http://192.168.7.240:8080/tasks/3
- CVAT task 4: http://192.168.7.240:8080/tasks/4
- TensorBoard **locally on ARGUS only when launched**: http://127.0.0.1:6006 . A local SSH tunnel is required for remote browser access if launched with host 127.0.0.1.
- AMD background: https://rocm.blogs.amd.com/ecosystems-and-partners/ultralytics-amd/README.html . The *working ARGUS adaptation* with Mirage / GPU-P is an experimental extension, **not** AMD's supported native Hyper-V deployment recipe.

## Guardrails

- No fresh driver installation, GPU partition changes, DDA, Docker prune, deletion of ~53GB expanded AMD Docker image, or upgrades to the working training container.
- Do not conflate DirectML on INFINITY with PyTorch ROCm inside ARGUS's AMD Docker image.
- Keep source/model/test work off Windows C:. Production installed applications can reside in Program Files per earlier user clarification.
- Dataset predictions are **not** ground-truth human-reviewed annotations; preserve prediction provenance and review status.
- Before adding FiftyOne or Albumentations, audit existing project directories, container versions, training data and VM memory. Avoid installing into existing production/CVAT/TensorBoard environments.

## Next checkpoint

Inspect existing CVAT Project 2 / Task 4 images and annotations; confirm human-review state and correct label mapping. Then choose a *new isolated* analysis environment or reuse an explicitly verified suitable environment for augmentation and FiftyOne, accounting for ARGUS's ~8GiB total RAM. Do not claim FiftyOne is installed yet.
