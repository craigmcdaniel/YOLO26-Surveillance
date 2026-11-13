# NYAAI-DOYA

**Not Your Average AI — Detect Once, You Are.**

Experimental YOLO26 surveillance models and integration tooling for Blue Iris and CodeProject.AI.

The GitHub repository remains named `YOLO26-Surveillance`; **NYAAI-DOYA** is the project name.

## YOLO26s experimental download

The working experimental **YOLO26s-640 ONNX** is distributed in the **attachment on the original IP Cam Talk forum post**, not through this GitHub repository. Download the attachment there and extract `yolo26s-filtered-test.onnx`. The ONNX binary is **not committed here**.

See the [test-branch instructions](../../tree/test) and [IP Cam Talk posting notes](../../blob/test/docs/IPCAM_RELEASE_NOTES.md) for the file mapping, SHA-256, testing notes, and limitations. GitHub is the source for the integration code and documentation; the forum attachment is the model distribution point for this experiment.

## What it detects

The current experimental model exposes a selected eight-class output, in order:

`person, bicycle, car, motorcycle, bus, truck, cat, dog`

It uses **ONNX output channel selection (Gather)**, not computationally pruned weights or a newly trained eight-class head. The original COCO output is retained. This is a useful working proof of concept, **not** a finished universally optimized inference engine.

## Blue Iris quick start

1. Download the experimental ONNX ZIP attachment from the original IP Cam Talk post.
2. Extract the ONNX model to your preferred location.
3. In a compatible Blue Iris setup, add or select the custom ONNX model using its native AI settings.
4. Check known images in AI Inspector and verify object labels before relying on an alert.

## Verify the file

Expected SHA-256 of `yolo26s-filtered-test.onnx`:

`DBC9326FC46819B02E27C842E9A3132291C5B0D424012610F061EDFC44422E17`

PowerShell (replace the example path with your extracted file location):

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath "G:\YOLO26-Migration\YOLO26Surveillance\yolo26s-filtered-test.onnx"
```

## Status

**Experimental beta; no stable release declared.** YOLO26s has produced live detections through Blue Iris and CodeProject.AI in the original developer's environment. The CodeProject.AI module and install scripts are development artifacts, **not a validated one-click installation**. GPU provider availability and response times vary across machines.

The repository's `main` branch documents the public entry point. Development remains on [`test`](../../tree/test). Additional model training, FP16, MIGraphX, CUDA, TensorRT, and other backend work are roadmap items, not currently supported release features.
