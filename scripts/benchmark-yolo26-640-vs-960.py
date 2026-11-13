"""Stage 6, paired real-image benchmark. Never modifies source images or production models.

Requires Pillow, numpy, onnxruntime-directml in G:\YOLO26-L960\venv.
Results: G:\YOLO26-L960\results\stage6\results.jsonl and summary.csv.
Reruns resume from completed image/model pairs. Errors are recorded and skipped.
"""
import csv
import hashlib
import json
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

ROOT = Path(r"G:\YOLO26-L960")
SOURCE = ROOT / "test-images"
OUT = ROOT / "results" / "stage6"
RECORDS = OUT / "results.jsonl"
SUMMARY = OUT / "summary.csv"
CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "cat", "dog"]
MODELS = {
    "YOLO26s_640": (
        Path(r"G:\YOLO26-Migration\YOLO26Surveillance\yolo26s-filtered-test.onnx"),
        "DBC9326FC46819B02E27C842E9A3132291C5B0D424012610F061EDFC44422E17",
        640,
    ),
    "YOLO26l_960": (
        ROOT / "yolo26l-surveillance-960.onnx",
        "471090E1E1304EF1E48F1801E48D8E91F574E09708171BED803F08EED5E4288A",
        960,
    ),
}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".jfif"}
THRESHOLD = 0.25
IOU = 0.45


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def create_session(name, path, expected_hash, size):
    if not path.is_file():
        raise RuntimeError(f"{name}: missing {path}")
    actual = sha256(path)
    if actual != expected_hash:
        raise RuntimeError(f"{name}: SHA-256 mismatch {actual}")

    opts = ort.SessionOptions()
    opts.enable_mem_pattern = False
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(path),
        sess_options=opts,
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    if "DmlExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"{name}: DirectML provider not available")

    inp = session.get_inputs()[0]
    outputs = {item.name: item.shape for item in session.get_outputs()}
    if inp.shape != [1, 3, size, size]:
        raise RuntimeError(f"{name}: unexpected input {inp.shape}")
    if outputs.get("surveillance_output") != [1, 12, 8400 if size == 640 else 18900]:
        raise RuntimeError(f"{name}: unexpected surveillance output {outputs}")

    print(f"{name}: provider={session.get_providers()} input={inp.shape}", flush=True)
    return session, inp.name


def letterbox(image, side):
    width, height = image.size
    scale = min(side / width, side / height)
    w = min(side, max(1, round(width * scale)))
    h = min(side, max(1, round(height * scale)))
    resized = image.resize((w, h), Image.Resampling.BILINEAR)
    left = (side - w) // 2
    top = (side - h) // 2
    canvas = Image.new("RGB", (side, side), (114, 114, 114))
    canvas.paste(resized, (left, top))
    tensor = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])
    return tensor, scale, left, top


def non_max_suppression(boxes, scores):
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order):
        best = int(order[0])
        keep.append(best)
        if len(order) == 1:
            break
        others = order[1:]
        xa = np.maximum(boxes[best, 0], boxes[others, 0])
        ya = np.maximum(boxes[best, 1], boxes[others, 1])
        xb = np.minimum(boxes[best, 2], boxes[others, 2])
        yb = np.minimum(boxes[best, 3], boxes[others, 3])
        intersection = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
        a = np.maximum(0, boxes[best, 2] - boxes[best, 0]) * np.maximum(0, boxes[best, 3] - boxes[best, 1])
        b = np.maximum(0, boxes[others, 2] - boxes[others, 0]) * np.maximum(0, boxes[others, 3] - boxes[others, 1])
        overlap = intersection / np.maximum(a + b - intersection, 1e-9)
        order = others[overlap <= IOU]
    return keep


def decode_predictions(output, original_size, scale, left, top):
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 12:
        raise RuntimeError(f"Unexpected prediction tensor: {output.shape}")
    if not np.isfinite(output).all():
        raise RuntimeError("Non-finite model output")
    pred = output[0].T
    scores = pred[:, 4:]
    labels = np.argmax(scores, axis=1)
    confidence = np.max(scores, axis=1)
    eligible = confidence >= THRESHOLD
    pred, labels, confidence = pred[eligible], labels[eligible], confidence[eligible]
    xywh = pred[:, :4]
    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    width, height = original_size
    detections = []
    for label in np.unique(labels):
        positions = np.where(labels == label)[0]
        for selected in non_max_suppression(boxes[positions], confidence[positions]):
            index = positions[selected]
            box = boxes[index]
            x1 = int(round(np.clip((box[0] - left) / scale, 0, width)))
            y1 = int(round(np.clip((box[1] - top) / scale, 0, height)))
            x2 = int(round(np.clip((box[2] - left) / scale, 0, width)))
            y2 = int(round(np.clip((box[3] - top) / scale, 0, height)))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                "label": CLASSES[int(label)],
                "confidence": round(float(confidence[index]), 5),
                "box": [x1, y1, x2, y2],
            })
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections


def load_existing():
    done = set()
    if RECORDS.exists():
        with RECORDS.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("status") == "ok":
                    done.add((item["image"], item["model"]))
    return done


def write_record(item):
    with RECORDS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, separators=(",", ":"), ensure_ascii=False) + "\n")
        stream.flush()


def make_summary():
    latest = {}
    with RECORDS.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "ok":
                latest[(record["image"], record["model"])] = record

    with SUMMARY.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "images", "detections", "images_with_detections",
                         "median_inference_ms", "mean_inference_ms", "labels"])
        for name in MODELS:
            rows = [row for (_, model), row in latest.items() if model == name]
            timings = [row["inference_ms"] for row in rows]
            labels = Counter(d["label"] for row in rows for d in row["detections"])
            writer.writerow([
                name, len(rows), sum(len(row["detections"]) for row in rows),
                sum(bool(row["detections"]) for row in rows),
                round(statistics.median(timings), 2) if timings else "",
                round(statistics.mean(timings), 2) if timings else "",
                json.dumps(labels, sort_keys=True),
            ])

    paired = sum(
        all((image, name) in latest for name in MODELS)
        for image in {image for image, _ in latest}
    )
    print(f"Complete pairs: {paired}", flush=True)
    print(f"Results: {RECORDS}", flush=True)
    print(f"Summary: {SUMMARY}", flush=True)


def main():
    if not SOURCE.is_dir():
        raise RuntimeError(f"Source image directory not found: {SOURCE}")
    candidates = sorted(
        (p for p in SOURCE.rglob("*")
         if p.is_file() and p.suffix.lower() in EXTENSIONS),
        key=lambda p: str(p).lower(),
    )
    if not candidates:
        raise RuntimeError(f"No recognized images in {SOURCE}")

    if "DmlExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("Install DirectML runtime in isolated G: virtual environment first")

    OUT.mkdir(parents=True, exist_ok=True)
    model_sessions = {}
    for name, (path, digest, side) in MODELS.items():
        session, inp = create_session(name, path, digest, side)
        model_sessions[name] = (session, inp, side)

    print(f"Image files found: {len(candidates)}", flush=True)
    print("Threshold=0.25; IoU=0.45; same originals; per-class NMS", flush=True)
    print("Session timing excludes image decode, resize, NMS and disk writes", flush=True)
    done = load_existing()
    started = time.perf_counter()
    try:
        for position, path in enumerate(candidates, 1):
            relative = path.relative_to(SOURCE).as_posix()
            pending = [name for name in MODELS if (relative, name) not in done]
            if not pending:
                continue

            try:
                with Image.open(path) as source:
                    frame = ImageOps.exif_transpose(source)
                    frame.seek(0)
                    frame = frame.convert("RGB")
                if frame.width < 1 or frame.height < 1:
                    raise RuntimeError("Invalid image dimensions")
            except Exception as exc:
                for name in pending:
                    write_record({
                        "time_utc": datetime.now(timezone.utc).isoformat(),
                        "image": relative, "model": name,
                        "status": "decode_error", "error": str(exc)[:300],
                    })
                print(f"[{position}/{len(candidates)}] SKIPPED {relative}: {exc}", flush=True)
                continue

            # Rotate execution order per image to reduce ordering bias.
            pending.sort(reverse=(position % 2 == 0))
            for name in pending:
                session, input_name, side = model_sessions[name]
                try:
                    tensor, scale, left, top = letterbox(frame, side)
                    start = time.perf_counter()
                    output = session.run(["surveillance_output"], {input_name: tensor})[0]
                    inference_ms = (time.perf_counter() - start) * 1000
                    found = decode_predictions(output, frame.size, scale, left, top)
                    item = {
                        "time_utc": datetime.now(timezone.utc).isoformat(),
                        "image": relative, "model": name, "status": "ok",
                        "width": frame.width, "height": frame.height,
                        "inference_ms": round(inference_ms, 3),
                        "detections": found,
                    }
                    done.add((relative, name))
                except Exception as exc:
                    item = {
                        "time_utc": datetime.now(timezone.utc).isoformat(),
                        "image": relative, "model": name,
                        "status": "inference_error", "error": str(exc)[:500],
                    }
                    print(f"ERROR {relative} {name}: {exc}", flush=True)
                write_record(item)

            if position == 1 or position % 25 == 0 or position == len(candidates):
                print(f"Progress: {position}/{len(candidates)}; elapsed {time.perf_counter()-started:.1f}s", flush=True)
                make_summary()
    except KeyboardInterrupt:
        print("Interrupted; completed image/model records saved.", flush=True)
    finally:
        make_summary()
    print("Production unchanged.", flush=True)


if __name__ == "__main__":
    main()
