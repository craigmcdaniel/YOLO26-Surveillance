# YOLO26 experimental test branch

The test branch contains the CodeProject.AI adapter and an **experimental eight-class** ONNX output. Bicycle is intentionally retained as a graph-slicing demonstration. The planned stable surveillance class set contains seven classes.

The model file `YOLO26Surveillance/yolo26s-filtered-test.onnx` is not yet committed to GitHub. The original working artifact has SHA-256:

`DBC9326FC46819B02E27C842E9A3132291C5B0D424012610F061EDFC44422E17`

## CHARON dry-run

Stage this branch outside the running native CodeProject.AI installation. CHARON has the standalone ONNX ZIP available at `D:\yolo26s-filtered-test.zip`; extract the model into `YOLO26Surveillance/` within the isolated checkout. Run the CPU smoke test first, then plan a service/module integration test. Do not overwrite the active module directory, Blue Iris, Neolink or Mosquitto.

The full source-to-ONNX graph transformation script was not contained in the uploaded archive. Do not assert reproducibility of the ONNX export until it is recovered.
