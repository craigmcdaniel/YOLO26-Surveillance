# ARGUS GPU-P / ROCm / YOLO26 verified checkpoint

**Observation date: 2026-11-16.** Consolidation of standalone `ARGUS-GPU-P-ROCm-Checkpoint-2026-11-16.md`. No tests rerun by this Git change.

## Confirmed working path

INFINITY (Windows 11 RX 9070 XT) → Hyper-V GPU-P → ARGUS (Ubuntu 24.04.5 kernel 7.0.0-43) → `dxgkrnl` and `/dev/dxg` → Windows DXCore + AMD driver userspace → ROCDXG → Docker `ultralytics/ultralytics:latest-amd` → PyTorch ROCm → YOLO26s.

- ARGUS: 12 virtual CPUs, 8GiB assigned RAM, GPU-P adapter for Radeon `DEV_7550`; /dev/kfd **absent** (expected for this bridge, **not** evidence GPU compute is unavailable).
- `GALLIUM_DRIVER=d3d12` explicitly forced: `D3D12 (AMD Radeon RX 9070 XT)` renderer and successful pixel readback. Without forcing D3D12, earlier smoke attempt selected `llvmpipe`.
- AMD `rocdxg-roct` version **1.2.2** installed on ARGUS. Key files: `/opt/rocm/lib/librocdxg.so`, `/opt/rocm/share/rocdxg/dids.conf`.
- GPU-aware Docker command mounts `/dev/dxg`, `/usr/lib/wsl/lib`, `/usr/lib/wsl/drivers`, ROCDXG library and device IDs; sets `LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib`, `HSA_ENABLE_DXG_DETECTION=1`, runs `ultralytics/ultralytics:latest-amd`.
- **GPU compute proved:** PyTorch 2.13.0+rocm10.0.0, HIP 7.15.26333, one AMD Radeon RX 9070 XT enumerated and GPU matrix multiplication passed.
- **YOLO26s inference:** 64/64 Carport images successful; model inference mean reported **6.2ms/image** (384x640). These are not full pipeline times.
- **Training:** one epoch COCO8 (4 training + 4 validation images) completed, peak memory ~1.43GB, ~23 seconds training progress, `best.pt` and `last.pt` saved (~20.4MB). mAP50=0.926, mAP50–95=0.662 **only for tiny COCO8**, not surveillance accuracy.
- **Known issue:** AMP consistency check failed, Ultralytics auto-disabled AMP; no forced re-enable.

## Paths and assets

- ARGUS project: `/home/argus/gpu-pv-lab`
- Source Carport images on Windows: `G:\hera\yolo-test`
- Windows SCP staging: `G:\ARGUS\yolo-smoke\images`
- ARGUS images: `/home/argus/gpu-pv-lab/images`
- Outputs: `/home/argus/gpu-pv-lab/yolo-validation/runs/camera-inference`, `.../gpu-training`
- Weights: `.../gpu-training/weights/best.pt`, `last.pt`
- Model/container image expanded size previously ~53.47GB: **do not prune**.
- Mirrored `G:\ARGUS\gpu-pv-lab\mirage-gpupv` is upstream clone, not our docs repo.

## Failure chronology / avoid regression

1. Generic AMD Linux container requires native /dev/kfd and /dev/dri; ARGUS Hyper-V GPU-P instead exposes /dev/dxg. Calling the system CPU-only from that observation was incorrect.
2. Mirage GPU-P driver setup required DKMS on kernel 7.0.0-43. An earlier PowerShell scalar/.Count bug and Secure Boot MOK step interrupted the process; secure boot was disabled for this VM.
3. DDA attempt failed VM startup `0xC035001E`. Radeon restored to host, DDA removed, GPU-P restored. **Do not repeat DDA casually.**
4. `--device=/dev/dxg` alone yielded missing `libdxcore.so`; binding WSL userspace fixed DXCore but was insufficient without ROCDXG.
5. Installing official ROCDXG 1.2.2, mounting its library and dids.conf, and setting `HSA_ENABLE_DXG_DETECTION=1` enabled GPU compute.
6. `ExposeVirtualizationExtensions=True` exposed `svm` but did not supply native /dev/kfd or physical PCI passthrough.

## Status distinctions

**PASS:** GPU enumeration, matrix multiply, 64-frame inference, 1-epoch COCO8 training, saved weights.

**NOT PROVEN:** long-term uptime/reboot persistence, real labeled surveillance fine-tune, reliable AMP, MIGraphX GPU ONNX provider, end-to-end production camera throughput.

**Do not change** working Windows GPU-P VM configuration, AMD Docker image, ROCDXG libraries, Frigate, or production CPAI for further observational work.
