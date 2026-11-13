"""Stage 9: replay original camera image corpus through the live CPAI 960 route.

Uses G: development Python; does not modify originals, Frigate, Blue Iris or CPAI.
Results are stored separately from Stage 6; resume completed successes.
"""
import csv
import json
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(r"G:\YOLO26-L960")
IMAGES = ROOT / "test-images"
RESULTS = ROOT / "results" / "stage9-cpai-960"
JSONL = RESULTS / "responses.jsonl"
SUMMARY = RESULTS / "summary.csv"
URL = "http://127.0.0.1:32168/v1/vision/yolo26l-surveillance-960"
STATUS_URL = "http://127.0.0.1:32168/v1/module/list/status"
EXPECTED_MODULE = "YOLO26LSurveillance960"
EXT = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
CONFIDENCE = "0.25"


def append(record):
    with JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def existing_successes():
    completed = set()
    if JSONL.exists():
        with JSONL.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("status") == "ok":
                    completed.add(record["image"])
    return completed


def summarize():
    latest = {}
    errors = Counter()
    with JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "ok":
                latest[record["image"]] = record
            if record.get("status") != "ok":
                errors[record.get("status", "unknown")] += 1

    rows = list(latest.values())
    labels = Counter(p.get("label", "unknown") for r in rows for p in r["predictions"])
    latency = [r["client_round_trip_ms"] for r in rows]
    server_latency = [r["analysisRoundTripMs"] for r in rows if isinstance(r.get("analysisRoundTripMs"), (float, int))]
    infer = [r["inferenceMs"] for r in rows if isinstance(r.get("inferenceMs"), (float, int))]
    with SUMMARY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["module", "images_completed", "total_predictions", "images_with_predictions",
                         "client_median_ms", "server_roundtrip_median_ms", "inference_median_ms", "labels", "error_attempts"])
        writer.writerow([
            EXPECTED_MODULE, len(rows), sum(len(r["predictions"]) for r in rows),
            sum(bool(r["predictions"]) for r in rows),
            round(statistics.median(latency), 2) if latency else "",
            round(statistics.median(server_latency), 2) if server_latency else "",
            round(statistics.median(infer), 2) if infer else "",
            json.dumps(labels, sort_keys=True), json.dumps(errors, sort_keys=True),
        ])
    print(f"Completed: {len(rows)} | Predictions: {sum(len(r['predictions']) for r in rows)} | Errors recorded: {sum(errors.values())}", flush=True)
    print(f"JSONL: {JSONL}", flush=True)
    print(f"CSV: {SUMMARY}", flush=True)


def main():
    if not IMAGES.is_dir():
        raise SystemExit(f"Missing image folder: {IMAGES}")
    files = sorted((p for p in IMAGES.rglob("*") if p.is_file() and p.suffix.lower() in EXT),
                   key=lambda p: str(p).lower())
    if not files:
        raise SystemExit("No images found")

    with requests.Session() as session:
        print("=== STAGE 9: LIVE CODEPROJECT.AI REPLAY ===", flush=True)
        print("Target:", URL, flush=True)
        print("Images:", len(files), "Confidence:", CONFIDENCE, flush=True)

        try:
            status_response = session.get(STATUS_URL, timeout=10)
            status_response.raise_for_status()
            statuses = status_response.json().get("statuses", [])
            matching = [s for s in statuses if s.get("moduleId") == EXPECTED_MODULE]
        except (requests.RequestException, ValueError) as exc:
            raise SystemExit(f"STOPPED: Cannot verify CodeProject.AI module status: {exc}")

        if len(matching) != 1 or matching[0].get("status") != "Started":
            raise SystemExit(f"STOPPED: {EXPECTED_MODULE} is not reported Started: {matching}")

        print("Verified dedicated module:", matching[0]["status"], flush=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        done = existing_successes()
        print("Previously completed:", len(done), flush=True)
        started = time.perf_counter()
        for index, file in enumerate(files, 1):
            rel = file.relative_to(IMAGES).as_posix()
            if rel in done:
                continue
            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "image": rel,
            }
            try:
                with file.open("rb") as handle:
                    start = time.perf_counter()
                    response = session.post(
                        URL,
                        files={"image": (file.name, handle)},
                        data={"min_confidence": CONFIDENCE},
                        timeout=40,
                    )
                    record["client_round_trip_ms"] = round((time.perf_counter() - start) * 1000, 3)
                response.raise_for_status()
                payload = response.json()
                if payload.get("moduleId") != EXPECTED_MODULE:
                    raise RuntimeError(f"Unexpected moduleId: {payload.get('moduleId')!r}")
                if payload.get("success") is not True:
                    raise RuntimeError(f"Server reported failure: {payload.get('error') or payload.get('message')}")
                predictions = payload.get("predictions")
                if not isinstance(predictions, list):
                    raise RuntimeError("No predictions array in successful response")
                record.update({
                    "status": "ok",
                    "predictions": predictions,
                    "count": len(predictions),
                    "moduleId": payload.get("moduleId"),
                    "inferenceMs": payload.get("inferenceMs"),
                    "processMs": payload.get("processMs"),
                    "analysisRoundTripMs": payload.get("analysisRoundTripMs"),
                    "executionProvider": payload.get("executionProvider"),
                })
                done.add(rel)
            except (OSError, requests.RequestException, ValueError, RuntimeError) as exc:
                record["status"] = "error"
                record["error"] = str(exc)[:500]
                print(f"FAILED [{index}/{len(files)}] {rel}: {exc}", flush=True)
                append(record)
                if isinstance(exc, RuntimeError) and "Unexpected moduleId" in str(exc):
                    print("STOPPED: Server routed request to wrong module.", flush=True)
                    break
                continue
            append(record)
            if index == 1 or index % 25 == 0 or index == len(files):
                print(f"Progress {index}/{len(files)}, elapsed {time.perf_counter()-started:.1f}s", flush=True)
                summarize()
        summarize()
        print("Production and source images unchanged.", flush=True)


if __name__ == "__main__":
    main()
