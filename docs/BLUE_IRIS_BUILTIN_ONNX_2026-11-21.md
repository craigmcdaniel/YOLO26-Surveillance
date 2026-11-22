# Blue Iris built-in ONNX inference checkpoint — 2026-11-21

## Evidence and scope

Source: screenshots of Blue Iris Settings > AI and Blue Iris Status > AI Inspector, supplied November 21, 2026 (America/Detroit). These describe the observed configuration at that time, not a guarantee of current or future configuration.

**Correction to earlier architecture notes:** The observed `yolo26s-filtered-test` detections are executed through **Blue Iris's built-in AI provider**, not via an external CodeProject.AI service. CodeProject.AI work exists separately in the project; it must not be assumed to be the inference endpoint for these events.

## Confirmed Blue Iris AI settings

| Setting | Observed value |
| --- | --- |
| Provider | Built-in (selected) |
| Inference provider/device | DirectML: NVIDIA Quadro RTX 4000 with Max-Q Design |
| Models folder | `C:\BlueIris\AI\models` |
| Primary model | `yolov8s.onnx` |
| Additional models | `yolo26s-filtered-test,ipcam-animal` |
| Maximum analyzers | 4 |
| Minimum confidence | 35% |
| Pre-load models | Unchecked |
| Local service on port 32168 | Not selected |
| External server address | Not selected; UI shows `127.0.0.1` in inactive field |
| Restart interval | 0 min |

## Triggering, motion, and video analysis (operator-confirmed)

**Critical architecture clarification:** Blue Iris's own motion detection is **disabled**. Blue Iris is used for **AI inference**, not local motion-trigger generation.

- **Trigger source:** Frigate produces the initiating alert/event.
- **Transport/orchestration:** Frigate event → MQTT → Node-RED on THOTH → Blue Iris external nudge/trigger on CHARON.
- **Blue Iris behavior:** After receiving a nudge, Blue Iris performs built-in ONNX inference against the camera video. The operator describes the workload as **constant scanning of a 2560 × 1440 (1440p) video feed**.
- **Important distinction:** Do not interpret AI Inspector's repeated analyses as evidence that Blue Iris motion detection is enabled. They are AI inference evaluations. The available screenshots do not independently establish precise per-frame sampling cadence or the duration of inference following each external trigger.
- **Model execution:** Built-in Blue Iris provider with DirectML on the NVIDIA Quadro RTX 4000 Max-Q; no local or external CodeProject.AI server is selected for the observed inference path.

This configuration is intentional: **Frigate decides when to nudge Blue Iris; Blue Iris performs the AI inference.** Do not re-enable Blue Iris motion detection as part of documentation or integration work.

## Live AI Inspector observations

- AI Inspector showed multiple detections on a cat near the steps.
- `yolo26s-filtered-test` yielded cat detections in visible samples at approximately **62%, 65%, 71%, and 80%** confidence. The displayed model-processing durations included approximately **51–89 ms**.
- `ipcam-animal` independently returned cat labels, with several observed confidence values around **64–77%** and displayed durations around **46–86 ms**.
- The inspector showed a `Nothing found` result for a `yolo26s-filtered-test` frame; the model did not return a detection on every analyzed image.
- AI Inspector footer showed **120,344 analyses**, **average 168 ms** (`t_ave=168ms`), and **0 restarts** at the moment of the screenshot. These overall figures should not be misrepresented as per-model inference times.
- An earlier inspector screenshot showed `yolo26s-filtered-test` cat 64% in 85 ms and `ipcam-animal` cat 81% in 119 ms.

## Camera labeling caveat

The AI Inspector event filename begins `Backyard`, while the video overlay reads `FrontYard`. Operator confirmed that physical cameras were switched without reassignment in Frigate to avoid disrupting an otherwise functional configuration. **Logical camera names must not be treated as reliable physical-location identifiers until reconciled.** No reassignment is recommended solely for cosmetic naming.

## Architecture and follow-up

For this observed externally triggered Blue Iris processing path:

```text
Frigate detection / alert
         |
         v
  MQTT event on CHARON
         |
         v
Node-RED bridge on THOTH
         |
         v
Blue Iris external nudge / trigger on CHARON
         |
         v
Blue Iris AI inference on 2560 x 1440 camera video
         |
         +--> yolov8s.onnx (primary)
         +--> yolo26s-filtered-test (additional)
         +--> ipcam-animal (additional)
         |
         v
Built-in DirectML / NVIDIA Quadro RTX 4000 Max-Q

Blue Iris motion detection: DISABLED
```

The diagrams distinguish **upstream event triggering** from **Blue Iris's inference workload**. Frigate, MQTT, and Node-RED are parts of the observed trigger pipeline, rather than optional unrelated components. CodeProject.AI work exists separately and is not the selected AI provider for these Blue Iris detections.

Evidence supports successful live use of the filtered ONNX model in Blue Iris; it does not establish a controlled accuracy benchmark or universal performance characteristics. Preserve original screenshots and timestamps when a formal benchmarking record is created.
