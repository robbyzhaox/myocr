import logging
from pathlib import Path

import yaml  # type: ignore
from PIL import Image

from myocr.config import MODEL_PATH
from myocr.modeling.model import ModelZoo
from myocr.predictors.text_detection_predictor import TextDetectionParamConverter
from myocr.predictors.text_direction_predictor import TextDirectionParamConverter
from myocr.predictors.text_recognition_predictor import TextRecognitionParamConverter

from ..base import Pipeline

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

        self.dec_predictor = det_model.predictor(TextDetectionParamConverter(det_model.device))
        self.cls_predictor = cls_model.predictor(TextDirectionParamConverter())
        self.rec_predictor = rec_model.predictor(TextRecognitionParamConverter())

    def process(self, img_path: str):
        orig_image = Image.open(img_path).convert("RGB")
        detected = self.dec_predictor.predict(orig_image)
        if not detected:
            return None

        detected = self.cls_predictor(detected)
        rec = self.rec_predictor.predict(detected)
        rec.original(orig_image.size[0], orig_image.size[1])  # type: ignore
        logger.debug(f"recognized texts is: {rec}")
        return rec
