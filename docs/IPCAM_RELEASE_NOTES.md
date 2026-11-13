# NYAAI-DOYA — IP Cam Talk forum follow-up

**Project:** NYAAI-DOYA — **Not Your Average AI — Detect Once, You Are.**

**Repository:** https://github.com/craigmcdaniel/YOLO26-Surveillance

**Model binary:** the experimental `yolo26s-filtered-test.onnx` is already attached as a downloadable ZIP **to the forum post**. GitHub contains integration code and documentation, not the ONNX binary itself.

## Copy-ready forum text

I put the source and development notes for my experimental YOLO26s surveillance model on GitHub:

https://github.com/craigmcdaniel/YOLO26-Surveillance

I'm calling the project **NYAAI-DOYA** — *Not Your Average AI. Detect Once, You Are.*

The model download is still the **ZIP attached to this forum post**. You don't need to download it twice; GitHub has the technical details, the CodeProject.AI adapter, configuration notes, and ongoing experimental work.

This is the **640×640 YOLO26s ONNX** test build. It exposes eight surveillance-related labels: **person, bicycle, car, motorcycle, bus, truck, cat, dog**. The filtering is ONNX output selection (not a newly trained/pruned eight-class network). I have tested it with native Blue Iris ONNX detection and through CodeProject.AI, but it's still an experimental release and other hardware/configurations may behave differently.

For anyone who wants to verify the downloaded ONNX, the SHA-256 is:

`DBC9326FC46819B02E27C842E9A3132291C5B0D424012610F061EDFC44422E17`

If you try it, I'd be interested in what hardware and inference times you see, and whether it improves object detection for your cameras.

## Evidence and limitations

- Working testing on developer's hardware only; not a guarantee for every Blue Iris or CodeProject.AI installation.
- CUDA / TensorRT / ROCm-MIGraphX adapters and training are proposed work, not shipped capabilities.
- The source-to-ONNX graph transformation for the original 640 test model is not currently in the GitHub repository.
- The forum download may be a ZIP containing the ONNX; **hash applies to the extracted ONNX**, not the ZIP.
- Do not advertise sub-3 ms inference without a reproducible benchmark.
