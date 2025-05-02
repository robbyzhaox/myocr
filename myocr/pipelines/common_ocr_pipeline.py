import logging
from pathlib import Path

import cv2
import yaml  # type: ignore

from myocr.base import Pipeline, Predictor
from myocr.config import MODEL_PATH
from myocr.modeling.model import ModelZoo
from myocr.processors import (
    TextDetectionProcessor,
    TextDirectionProcessor,
    TextRecognitionProcessor,
)

logger = logging.getLogger(__name__)


class CommonOCRPipeline(Pipeline):
    def __init__(self, device):
        current_file = Path(__file__)
        config_path = current_file.parent / "config" / f"{current_file.stem}.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        det_model = ModelZoo.load_model("onnx", MODEL_PATH + config["model"]["detection"], device)
        cls_model = ModelZoo.load_model(
            "onnx", MODEL_PATH + config["model"]["cls_direction"], device
        )
        rec_model = ModelZoo.load_model("onnx", MODEL_PATH + config["model"]["recognition"], device)

        self.dec_predictor = Predictor(det_model, TextDetectionProcessor(det_model.device))
        self.cls_predictor = Predictor(cls_model, TextDirectionProcessor())
        self.rec_predictor = Predictor(rec_model, TextRecognitionProcessor())

    def process(self, img_path: str):
        orig_image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
        if orig_image is None:
            raise ValueError(f"path invalid: {img_path}")
        detected = self.dec_predictor.predict(orig_image)
        if not detected:
            return None

        detected = self.cls_predictor(detected)
        rec = self.rec_predictor.predict(detected)
        rec.original(width=orig_image.shape[1], height=orig_image.shape[0])  # type: ignore
        logger.debug(f"recognized texts is: {rec}")
        return rec
