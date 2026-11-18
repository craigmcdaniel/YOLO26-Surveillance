# PANOPTES TensorBoard checkpoint

**Observed 2026-11-16.** Originally recorded in standalone `PANOPTES-TensorBoard-Checkpoint-2026-11-16.md`, not previously committed. The statement that CVAT was not yet installed applies **only at the time this checkpoint was made**; later CVAT deployment is documented separately.

## Verified

- TensorBoard **2.20.0** in isolated host venv `/home/argus/gpu-pv-lab/tooling/tensorboard/venv`
- setuptools 80.10.2; pip 24.0; `pip check` clean; `tensorboard --version` and `tensorboard --inspect` passed.
- Training CSV: `/home/argus/gpu-pv-lab/yolo-validation/runs/gpu-training/results.csv`.
- Event logs: `/home/argus/gpu-pv-lab/tooling/tensorboard/logs/yolo26s-baseline`. **13 scalar tags**, one epoch/step: learning rates, precision, recall, mAP, train and validation losses.
- The dashboard was **not confirmed launched or visually inspected** as of that checkpoint.
- A warning about TensorFlow not being installed is expected; no reason to install TensorFlow to suppress it.

## Setup corrections (historical)

1. Python venv initially failed because Ubuntu lacked `python3.12-venv` / `ensurepip`.
2. Partial `bin/python` was incorrectly taken as proof of a complete environment; validate `python -m pip` too.
3. Installed `python3.12-venv`, `python3-pip-whl`, `python3-setuptools-whl`; recreated only dedicated TensorBoard venv.
4. Initial setuptools 84.2.1 broke `pkg_resources` import; pinned 80.10.2 in that venv, then verified.

## Existing dashboard launch (when needed)

```bash
clear

VENV="$HOME/gpu-pv-lab/tooling/tensorboard/venv"
LOGS="$HOME/gpu-pv-lab/tooling/tensorboard/logs"

"$VENV/bin/tensorboard" \
    --logdir "$LOGS" \
    --host 127.0.0.1 \
    --port 6006
```

Launch is interactive; use SSH port forwarding for remote access. Do not interpret the command or checkpoint as an assertion that the service is currently running.
