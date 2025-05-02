import os
import time

import cv2
import pytest
import torch

from myocr.base import Predictor
from myocr.config import MODEL_PATH
from myocr.modeling.model import ModelZoo
from myocr.processors import (
    ImageClassificationProcessor,
    RestNetImageClassificationProcessor,
)


def test_restnet():
    model = ModelZoo.load_model("pt", "resnet152", "cuda:0" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    p = Predictor(model, RestNetImageClassificationProcessor(model.device))
    print(p.predict(cv2.imread("tests/images/flower.png", cv2.IMREAD_COLOR_RGB)))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"执行时间: {execution_time} 秒")


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason=f"{MODEL_PATH} not exist, skip test.")
def test_mlp():
    model = ModelZoo.load_model(
        "custom",
        "myocr/modeling/configs/mlp.yaml",
        "cuda:0" if torch.cuda.is_available() else "cpu",
    )

    start_time = time.time()
    p = Predictor(
        model, ImageClassificationProcessor(model.device, resize=28, center_crop=28, channels=1)
    )
    print(p.predict(cv2.imread("tests/images/flower.png", cv2.IMREAD_GRAYSCALE)))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"执行时间: {execution_time} 秒")
