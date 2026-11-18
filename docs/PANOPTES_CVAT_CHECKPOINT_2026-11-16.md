# PANOPTES CVAT / IPCam-Combined checkpoint

**Historical evidence from 2026-11-16** plus current ARGUS process audit on **2026-11-18 03:00 UTC**. This record is a consolidation; no CVAT services or annotations were modified.

## Deployment and access

- CVAT container version observed: **2.82.0** (`cvat/server:v2.82.0`) in user-supplied worker diagnostics.
- Browser address evidenced by CVAT logs: http://192.168.7.240:8080/ ; tasks at `/tasks`; project and task URLs follow CVAT routing conventions.
- ARGUS later audit (2026-11-18): running CVAT Django task workers, Uvicorn and policy service. Docker daemon inventory **not readable from `argus` account without sudo**; expected operating access with `sudo docker`, not a reason to change socket permissions.
- CVAT tooling already exists under `/home/argus/gpu-pv-lab/tooling/cvat/`. Representative scripts include `setup_pilot.py`, `resume_representative.py`, `import_photostore.py`, `upload_representative_annotations.py`, `preannotate.py`, `upload_verified_boxes.py`, `infer_representative.py`, `import_representative.py`, `reduce_labels.py`, and `diagnose_frame_dimensions.py`. **Do not recreate tooling or reimport tasks blindly.**

## Observed chronology

- Earlier representative Task 2 had `Data attached: True` but **0 images** during import diagnosis; that was a point-in-time failure, not the state of all tasks.
- YOLO26s preannotations subsequently reported **745 boxes on Task 3**; two orientation exceptions withheld (recorded in conversation checkpoint). Needs read-back verification before claiming annotations are fully reviewed.
- Separate IPCam-Combined inference completed **862/862 source images**, **675 predictions**, **0 failures**, 40.633 seconds; evidence path `G:\ARGUS\gpu-pv-lab\benchmarks\ipcam-combined-onnx-20261116-130438\predictions.json`.
- IPCam-Combined class breakdown: car 319, person 253, truck 74, cat 6, dog 4, squirrel 3, bird 3, bicycle 2, deer 5, bear 2, pig 2, horse 1, motorcycle 1. These are **model outputs, not human-verified labels**.
- `20261116-181324` console evidence reports `VALID: IPCam-Combined predictions ready`, and confirms ARGUS CVAT import tool scripts. Later summary states Project 2 / Task 4 import checkpoint verified, but **the detailed current frame/annotation counts were not reconstructed from the live API in this documentation pass**.

## Next exact checkpoint

Open **Project 2 / Task 4** in CVAT and read back actual image count, annotation count, label order, human review status, and task state before continuing training or automated augmentation. Preserve existing tasks and annotations; never overwrite current predictions with a new run without an explicit plan and backup.
