import logging
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import yaml  # type: ignore

from ..base import Pipeline, Predictor
from ..config import MODEL_PATH
from ..modeling.model import ModelZoo
from ..processors import (
    TextDetectionProcessor,
    TextDirectionProcessor,
    TextRecognitionProcessor,
)
from ..types import OCRResult

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

    def process(self, img: Union[bytes, str, np.ndarray]):
        start_time = time.time()
        if isinstance(img, bytes):
            np_arr = np.frombuffer(img, dtype=np.uint8)
            orig_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR_RGB)
        elif isinstance(img, str):
            orig_image = cv2.imread(img, cv2.IMREAD_COLOR_RGB)
        elif isinstance(img, np.ndarray):
            orig_image = img

        if orig_image is None:
            raise ValueError("imgage invalid, please check")
        detected = self.dec_predictor.predict(orig_image)
        texts = []
        if detected[1]:  # type: ignore
            detected = self.cls_predictor(detected)
            texts = self.rec_predictor.predict(detected)
            logger.debug(f"recognized texts is: {texts}")
        return OCRResult.build(orig_image, texts, time.time() - start_time)  # type: ignore
