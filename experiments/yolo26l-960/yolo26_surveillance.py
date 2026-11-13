import sys
import time

import numpy as np
import onnxruntime as ort
from PIL import Image


CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "cat",
    "dog",
]

IOU_THRESHOLD = 0.45


class YOLO26Surveillance:
    def __init__(self, model_path, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            model_path,
            providers=providers,
        )

        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        shape = input_info.shape
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
            raise ValueError(f"Unsupported model input dimensions: {shape}")
        if not isinstance(shape[2], int) or not isinstance(shape[3], int):
            raise ValueError(f"Expected static input dimensions: {shape}")
        self.input_height = shape[2]
        self.input_width = shape[3]
        outputs = {value.name: value.shape for value in self.session.get_outputs()}
        shape_out = outputs.get("surveillance_output")
        if shape_out is None or len(shape_out) != 3 or shape_out[0] != 1 or shape_out[1] != 12:
            raise ValueError(f"Missing eight-class surveillance output: {outputs}")

    def preprocess(self, image):
        image = image.convert("RGB")

        width, height = image.size

        scale = min(
            self.input_width / width,
            self.input_height / height,
        )

        new_width = round(width * scale)
        new_height = round(height * scale)

        resized = image.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR,
        )

        pad_x = (self.input_width - new_width) // 2
        pad_y = (self.input_height - new_height) // 2

        canvas = Image.new(
            "RGB",
            (self.input_width, self.input_height),
            (114, 114, 114),
        )

        canvas.paste(
            resized,
            (pad_x, pad_y),
        )

        array = np.asarray(
            canvas,
            dtype=np.float32,
        )

        array = array / 255.0
        array = np.transpose(array, (2, 0, 1))
        array = np.expand_dims(array, axis=0)

        tensor = np.ascontiguousarray(array)

        return tensor, scale, pad_x, pad_y

    def calculate_iou(self, box, boxes):
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection = (
            np.maximum(0, x2 - x1)
            * np.maximum(0, y2 - y1)
        )

        area_a = (
            max(0, box[2] - box[0])
            * max(0, box[3] - box[1])
        )

        area_b = (
            np.maximum(0, boxes[:, 2] - boxes[:, 0])
            * np.maximum(0, boxes[:, 3] - boxes[:, 1])
        )

        union = area_a + area_b - intersection

        return intersection / np.maximum(union, 1e-9)

    def nms(self, boxes, scores):
        order = np.argsort(scores)[::-1]
        keep = []

        while len(order) > 0:
            current = order[0]
            keep.append(current)

            if len(order) == 1:
                break

            remaining = order[1:]

            overlaps = self.calculate_iou(
                boxes[current],
                boxes[remaining],
            )

            order = remaining[
                overlaps <= IOU_THRESHOLD
            ]

        return keep

    def detect(self, image, confidence_threshold=0.25):
        width, height = image.size

        tensor, scale, pad_x, pad_y = self.preprocess(image)

        started = time.perf_counter()

        filtered = self.session.run(
            ["surveillance_output"],
            {self.input_name: tensor},
        )[0][0]

        inference_ms = (
            time.perf_counter() - started
        ) * 1000.0

        predictions = filtered.T

        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]

        class_indices = np.argmax(
            class_scores,
            axis=1,
        )

        confidences = np.max(
            class_scores,
            axis=1,
        )

        valid = confidences >= confidence_threshold

        boxes_xywh = boxes_xywh[valid]
        class_indices = class_indices[valid]
        confidences = confidences[valid]

        boxes = np.zeros_like(boxes_xywh)

        boxes[:, 0] = (
            boxes_xywh[:, 0]
            - boxes_xywh[:, 2] / 2
        )

        boxes[:, 1] = (
            boxes_xywh[:, 1]
            - boxes_xywh[:, 3] / 2
        )

        boxes[:, 2] = (
            boxes_xywh[:, 0]
            + boxes_xywh[:, 2] / 2
        )

        boxes[:, 3] = (
            boxes_xywh[:, 1]
            + boxes_xywh[:, 3] / 2
        )

        detections = []

        for class_id in np.unique(class_indices):
            selected = np.where(
                class_indices == class_id
            )[0]

            keep = self.nms(
                boxes[selected],
                confidences[selected],
            )

            for local_index in keep:
                index = selected[local_index]

                x1, y1, x2, y2 = boxes[index]

                x1 = np.clip(
                    (x1 - pad_x) / scale,
                    0,
                    width,
                )

                y1 = np.clip(
                    (y1 - pad_y) / scale,
                    0,
                    height,
                )

                x2 = np.clip(
                    (x2 - pad_x) / scale,
                    0,
                    width,
                )

                y2 = np.clip(
                    (y2 - pad_y) / scale,
                    0,
                    height,
                )

                if x2 <= x1:
                    continue

                if y2 <= y1:
                    continue

                detections.append({
                    "label": CLASS_NAMES[int(class_id)],
                    "confidence": float(confidences[index]),
                    "x_min": int(round(x1)),
                    "y_min": int(round(y1)),
                    "x_max": int(round(x2)),
                    "y_max": int(round(y2)),
                })

        detections.sort(
            key=lambda item: item["confidence"],
            reverse=True,
        )

        return {
            "success": True,
            "predictions": detections,
            "count": len(detections),
            "inferenceMs": round(inference_ms, 2),
        }


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python yolo26_surveillance.py "
            "<model.onnx> <image>"
        )
        return

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    detector = YOLO26Surveillance(
        model_path,
    )

    print("Provider:", detector.session.get_providers())

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    result = detector.detect(
        image,
        confidence_threshold=0.25,
    )

    print()
    print("Success:", result["success"])
    print("Inference:", result["inferenceMs"], "ms")
    print("Detections:", result["count"])
    print()

    for prediction in result["predictions"]:
        print(
            "%-12s %.1f%%  (%d,%d)-(%d,%d)"
            % (
                prediction["label"],
                prediction["confidence"] * 100.0,
                prediction["x_min"],
                prediction["y_min"],
                prediction["x_max"],
                prediction["y_max"],
            )
        )


if __name__ == "__main__":
    main()
