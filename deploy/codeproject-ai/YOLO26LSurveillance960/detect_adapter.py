import os
import sys

from PIL import Image

from codeproject_ai_sdk import (
    JSON,
    LogMethod,
    ModuleRunner,
    RequestData,
)

from yolo26_surveillance import YOLO26Surveillance


class YOLO26LSurveillance960Adapter(ModuleRunner):

    def __init__(self):
        super().__init__()

        self.detector = None

        self.model_path = os.environ.get(
            "YOLO26_SURVEILLANCE_MODEL",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "yolo26l-surveillance-960.onnx",
            ),
        )

        self._num_items_found = 0
        self._histogram = {}

    def initialise(self):

        if not self.launched_by_server:
            self.queue_name = "yolo26l_surveillance_960_queue"

        self.log(
            LogMethod.Info | LogMethod.Server,
            {
                "filename": __file__,
                "loglevel": "information",
                "method": sys._getframe().f_code.co_name,
                "message": (
                    "Loading YOLO26 Surveillance model: "
                    + self.model_path
                ),
            },
        )

        self.detector = YOLO26Surveillance(
            self.model_path,
            providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        )

        self.inference_device = "DirectML"
        self.inference_library = "ONNX Runtime"

    def process(self, data: RequestData) -> JSON:

        if data.command != "detect":
            return {
                "success": False,
                "error": "Unknown command: " + str(data.command),
            }

        if self.detector is None:
            return {
                "success": False,
                "error": "Detector is not initialized",
            }

        threshold = float(
            data.get_value(
                "min_confidence",
                "0.4",
            )
        )

        image: Image.Image = data.get_image(0)

        result = self.detector.detect(
            image,
            confidence_threshold=threshold,
        )

        labels = [
            prediction["label"]
            for prediction in result.get("predictions", [])
        ]

        if labels:
            result["message"] = "Found " + ", ".join(labels)

        if not labels:
            result["message"] = "No objects found"
        result["error"] = None
        result["processMs"] = result["inferenceMs"]

        return result

    def module_status(self) -> JSON:

        status_data = super().module_status()

        status_data["numItemsFound"] = self._num_items_found
        status_data["histogram"] = self._histogram

        return status_data

    def update_statistics(self, response):

        super().update_statistics(response)

        if not response.get("success"):
            return

        predictions = response.get(
            "predictions",
            [],
        )

        self._num_items_found += len(predictions)

        for prediction in predictions:
            label = prediction["label"]

            if label not in self._histogram:
                self._histogram[label] = 0

            self._histogram[label] += 1

    def selftest(self) -> JSON:

        if self.detector is None:
            return {
                "success": False,
                "message": "Detector is not initialized",
            }

        # Self-test inference with a deterministic blank frame.
        # Real-object detection is validated separately against alerts.
        image = Image.new("RGB", (640, 640), (114, 114, 114))

        result = self.detector.detect(
            image,
            confidence_threshold=0.25,
        )

        return {
            "success": bool(result.get("success", True)),
            "message": "YOLO26 Surveillance inference completed",
            "inferenceMs": result.get("inferenceMs"),
        }


if __name__ == "__main__":
    YOLO26SurveillanceAdapter().start_loop()
