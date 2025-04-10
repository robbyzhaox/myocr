import logging

import yaml  # type: ignore
from PIL import Image

from myocr.config import MODEL_PATH
from myocr.modeling.model import ModelZoo
from myocr.predictors.text_detection_predictor import TextDetectionParamConverter
from myocr.predictors.text_recognition_predictor import TextRecognitionParamConverter

from ..base import Pipeline

logger = logging.getLogger(__name__)


class CommonOCRPipeline(Pipeline):
    def __init__(self, device):
        parts = __file__.split(".")[0].rsplit("/", 1)
        with open(parts[0] + "/config/" + parts[1] + ".yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        det_model = ModelZoo.load_model("onnx", MODEL_PATH + config["model"]["detection"], device)
        rec_model = ModelZoo.load_model("onnx", MODEL_PATH + config["model"]["recognition"], device)

        self.dec_predictor = det_model.predictor(TextDetectionParamConverter(det_model.device))

        cvt = TextRecognitionParamConverter()
        self.rec_predictor = rec_model.predictor(cvt)

    def __call__(self, img_path: str):
        orig_image = Image.open(img_path).convert("RGB")
        detected = self.dec_predictor.predict(orig_image)
        if not detected:
            return None

        rec = self.rec_predictor.predict(detected)
        logger.debug(f"recognized texts is: {rec}")
        return rec
