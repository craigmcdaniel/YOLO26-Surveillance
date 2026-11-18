"""Read-only RGB vs CLAHE paired test for YOLO26s on INFINITY.

Run in the existing G: test venv after installing opencv-python-headless.
Does not alter model files, camera images, Frigate, or CodeProject.AI.
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "YOLO26Surveillance"))
from yolo26_surveillance import YOLO26Surveillance

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".jfif"}


def clahe_rgb(image, clip_limit, tiles):
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tiles, tiles),
    ).apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced, mode="RGB")


def main():
    parser = argparse.ArgumentParser(description="Paired YOLO26 original RGB / CLAHE benchmark")
    parser.add_argument("--images", type=Path, default=Path(r"G:\YOLO26-L960\test-images"))
    parser.add_argument("--model", type=Path, default=Path(r"G:\YOLO26-Migration\YOLO26Surveillance\yolo26s-filtered-test.onnx"))
    parser.add_argument("--output", type=Path, default=Path(r"G:\YOLO26-L960\results\clahe-experiment"))
    parser.add_argument("--limit", type=int, default=20, help="Number of images; 0 for all")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument("--tiles", type=int, default=8)
    args = parser.parse_args()

    if not args.images.is_dir():
        parser.error(f"Image directory missing: {args.images}")
    if not args.model.is_file():
        parser.error(f"ONNX model missing: {args.model}")
    if args.limit < 0 or args.tiles < 1 or args.clip_limit <= 0:
        parser.error("limit must be >= 0, tiles >= 1, clip-limit > 0")
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between 0 and 1")

    images = sorted(
        (p for p in args.images.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS),
        key=lambda p: str(p).lower(),
    )
    if args.limit:
        images = images[:args.limit]
    if not images:
        parser.error("No source images found")

    detector = YOLO26Surveillance(
        str(args.model),
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    if "DmlExecutionProvider" not in detector.session.get_providers():
        parser.error("DirectML provider unavailable; refusing misleading CPU comparison")

    run_name = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%SZ")
    destination = args.output / run_name
    destination.mkdir(parents=True, exist_ok=False)

    with (destination / "detections.jsonl").open("w", encoding="utf-8") as detailed, \
         (destination / "comparison.csv").open("w", newline="", encoding="utf-8") as tabular:
        writer = csv.writer(tabular)
        writer.writerow([
            "image", "original_count", "clahe_count",
            "original_top_confidence", "clahe_top_confidence",
            "original_inference_ms", "clahe_inference_ms",
            "clahe_transform_ms", "original_total_ms", "clahe_total_ms",
        ])
        for index, path in enumerate(images, 1):
            relative = path.relative_to(args.images).as_posix()
            try:
                with Image.open(path) as source:
                    original = ImageOps.exif_transpose(source).convert("RGB")

                start_transform = time.perf_counter()
                enhanced = clahe_rgb(original, args.clip_limit, args.tiles)
                transform_ms = (time.perf_counter() - start_transform) * 1000
                if index <= 5:
                    enhanced.save(destination / f"clahe-preview-{index:03d}.png")

                outputs = {}
                variants = [
                    ("original", original),
                    ("clahe", enhanced),
                ]
                if index % 2 == 0:
                    variants.reverse()
                for mode, frame in variants:
                    started = time.perf_counter()
                    result = detector.detect(frame, confidence_threshold=args.threshold)
                    total_ms = (time.perf_counter() - started) * 1000
                    outputs[mode] = result
                    outputs[mode]["total_ms"] = round(total_ms, 3)

                original_result = outputs["original"]
                clahe_result = outputs["clahe"]
                original_predictions = original_result["predictions"]
                clahe_predictions = clahe_result["predictions"]
                writer.writerow([
                    relative, len(original_predictions), len(clahe_predictions),
                    round(original_predictions[0]["confidence"], 5) if original_predictions else "",
                    round(clahe_predictions[0]["confidence"], 5) if clahe_predictions else "",
                    original_result["inferenceMs"], clahe_result["inferenceMs"],
                    round(transform_ms, 3),
                    original_result["total_ms"], clahe_result["total_ms"],
                ])
                detailed.write(json.dumps({
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "image": relative,
                    "clahe_clip_limit": args.clip_limit,
                    "clahe_tiles": args.tiles,
                    "threshold": args.threshold,
                    "transform_ms": round(transform_ms, 3),
                    "original": original_result,
                    "clahe": clahe_result,
                }) + "\n")
                detailed.flush()
                tabular.flush()
                print(f"[{index}/{len(images)}] {relative}: RGB {len(original_predictions)} | CLAHE {len(clahe_predictions)}", flush=True)
            except (OSError, ValueError, RuntimeError) as exc:
                detailed.write(json.dumps({
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "image": relative,
                    "status": "error",
                    "error": str(exc),
                }) + "\n")
                detailed.flush()
                print(f"[{index}/{len(images)}] ERROR {relative}: {exc}", flush=True)

    print(f"Results: {destination}")
    print("Counts and confidence are diagnostics, not proof of improved accuracy.")
    print("Production unchanged.")


if __name__ == "__main__":
    main()
