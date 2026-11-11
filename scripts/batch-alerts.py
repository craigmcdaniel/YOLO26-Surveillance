#!/usr/bin/env python3
"""Read-only batch test of Blue Iris JPG alerts against the experimental model.

Writes all results outside the Blue Iris alert directory.
Camera02 paths and filenames are deliberately excluded, case-insensitively.
"""

import argparse
import csv
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "YOLO26Surveillance"))

from yolo26_surveillance import YOLO26Surveillance  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alerts", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "YOLO26Surveillance" / "yolo26s-filtered-test.onnx")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--date", default=date.today().isoformat(), help="Date based on JPG modified timestamp, YYYY-MM-DD")
    parser.add_argument("--all-dates", action="store_true", help="Ignore JPG modified timestamp filter")
    parser.add_argument("--provider", choices=("cpu", "dml"), default="cpu")
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()

    if not args.alerts.is_dir():
        parser.error(f"Alert directory not found: {args.alerts}")
    if not args.model.is_file():
        parser.error(f"ONNX model not found: {args.model}")
    if args.out.resolve() == args.alerts.resolve() or args.alerts.resolve() in args.out.resolve().parents:
        parser.error("Output must be outside the Blue Iris alerts directory")
    if not args.all_dates:
        try:
            target_date = date.fromisoformat(args.date)
        except ValueError:
            parser.error("--date must be YYYY-MM-DD")
    else:
        target_date = None

    candidates = []
    excluded_camera02 = 0
    excluded_date = 0
    for image in sorted(args.alerts.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        relative = image.relative_to(args.alerts)
        if "camera02" in str(relative).lower():
            excluded_camera02 += 1
            continue
        if target_date and datetime.fromtimestamp(image.stat().st_mtime).date() != target_date:
            excluded_date += 1
            continue
        candidates.append(image)

    print(f"JPGs selected: {len(candidates)}; Camera02 skipped: {excluded_camera02}; other dates skipped: {excluded_date}")
    if not candidates:
        print("No matching JPGs. No results written.")
        return 0

    requested_provider = {
        "cpu": ["CPUExecutionProvider"],
        "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    }[args.provider]
    detector = YOLO26Surveillance(str(args.model), providers=requested_provider)
    active = detector.session.get_providers()
    print("Active ONNX providers:", active)
    if args.provider == "dml" and "DmlExecutionProvider" not in active:
        print("ERROR: DirectML requested but unavailable; not silently falling back to CPU.", file=sys.stderr)
        return 2

    rows = []
    detailed = []
    started = time.perf_counter()
    for index, image in enumerate(candidates, 1):
        relative = str(image.relative_to(args.alerts))
        try:
            with Image.open(image) as opened:
                rgb = opened.convert("RGB")
            t0 = time.perf_counter()
            result = detector.detect(rgb, confidence_threshold=args.threshold)
            wall_ms = round(1000 * (time.perf_counter() - t0), 3)
            predictions = result.get("predictions", [])
            detailed.append({
                "image": relative,
                "success": result.get("success"),
                "inference_ms": result.get("inferenceMs"),
                "total_ms": wall_ms,
                "predictions": predictions,
            })
            if not predictions:
                rows.append({
                    "image": relative, "status": "no_detection",
                    "label": "", "confidence": "", "inference_ms": result.get("inferenceMs"),
                    "total_ms": wall_ms, "error": "",
                })
            for prediction in predictions:
                rows.append({
                    "image": relative, "status": "detected",
                    "label": prediction["label"],
                    "confidence": prediction["confidence"],
                    "inference_ms": result.get("inferenceMs"),
                    "total_ms": wall_ms, "error": "",
                })
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            rows.append({
                "image": relative, "status": "error",
                "label": "", "confidence": "", "inference_ms": "",
                "total_ms": "", "error": message,
            })
            detailed.append({"image": relative, "success": False, "error": message})
        if index % 25 == 0 or index == len(candidates):
            print(f"Processed {index}/{len(candidates)}")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "detections.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "image", "status", "label", "confidence", "inference_ms", "total_ms", "error"
        ])
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "detections.json").write_text(json.dumps({
        "source": str(args.alerts),
        "model": str(args.model),
        "date_filter": None if args.all_dates else args.date,
        "provider_requested": args.provider,
        "provider_active": active,
        "confidence_threshold": args.threshold,
        "images_selected": len(candidates),
        "camera02_skipped": excluded_camera02,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "results": detailed,
    }, indent=2), encoding="utf-8")
    print("Saved:", args.out / "detections.csv")
    print("Saved:", args.out / "detections.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
