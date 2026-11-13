# NYAAI-DOYA — experimental YOLO26s beta

**Not Your Average AI — Detect Once, You Are.**

Public project name: **NYAAI-DOYA**. Repository name: **YOLO26-Surveillance** (unchanged).

This `test` branch preserves working YOLO26s model integration code, experimental graph-slicing documentation, benchmarks, and subsequent development. It is not a stable or one-click-installable release.

## Where to get the YOLO26s model

The `yolo26s-filtered-test.onnx` binary is provided in the **original IP Cam Talk forum post attachment**, rather than committed to GitHub. The GitHub model directory contains adapter/source files and a checksum, **not the ONNX itself**. Download and extract the forum ZIP first.

Expected SHA-256:

`DBC9326FC46819B02E27C842E9A3132291C5B0D424012610F061EDFC44422E17`

To verify a downloaded file on Windows:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath "PATH_TO_EXTRACTED\yolo26s-filtered-test.onnx"
```

Replace the placeholder path. If the hash differs, do not assume you have the same experimental build.

## Blue Iris: minimal native ONNX trial

If your version/configuration of Blue Iris already supports custom native ONNX models, place the extracted model in your selected AI model directory, select it in the Blue Iris AI configuration, and test it using AI Inspector. Confirm the model results on a known image and set alerts according to your existing camera configuration. This is an experimental test, not a claim of universal drop-in compatibility.

## Model output and class order

The test graph exposes `surveillance_output` with shape **[1,12,8400]**, four box channels and eight class-score channels:

`person, bicycle, car, motorcycle, bus, truck, cat, dog`

Bicycle remains included in the **actual tested eight-class model**; do not silently replace it with a hypothetical seven-class mapping. This is post-inference output selection via `Gather`, **not true neural-network pruning**. The original full 80-class graph output is also retained.

## Tested environment and scope

- **INFINITY:** currently Windows 11, AMD RX 9070 XT, with working DirectML/CodeProject.AI YOLO26 experiments. Older development notes refer to an available Windows Server 2025 installation; do not interpret that as the current boot.
- **CHARON:** Windows Server 2025 Datacenter, Blue Iris, NVIDIA Quadro RTX 4000, observed native ONNX/DirectML detections.
- On-device timings differ between **pure inference**, **server processing**, and **Blue Iris round-trip analysis**. No sub-3 ms performance claim has been validated.

## CodeProject.AI integration

`YOLO26Surveillance/` contains the experimental SDK adapter, Python detector, settings, and dependency files. It expects `yolo26s-filtered-test.onnx` at the module path unless a model environment variable is specified. A standalone, clean-machine installer has **not** been validated. Don't overwrite an existing working module to try these development files.

## Source and development

- The existing YOLO26s graph transformation script has **not yet been published/recovered** for a source-to-model reproducible build.
- The YOLO26l-960 development and deployment scripts are **different experimental artifacts**; they are not needed for the forum's YOLO26s download.
- The planned fine-tuned eight-class head and accelerator-specific engines are future experiments.
- Source-model licensing and redistribution permissions should be checked before presenting a formal stable release.

See [IP Cam Talk posting notes](docs/IPCAM_RELEASE_NOTES.md), [test branch notes](docs/TEST_BRANCH.md), and [development history](docs/YOLO26L_960_BUILD_LOG.md).

**Release positioning:** experimental, working prototype; model download stays with the forum attachment while GitHub supplies source, documentation, and the public project identity.
