import os
import time

import pytest
import torch
from PIL import Image

from myocr.config import MODEL_PATH
from myocr.modeling.model import ModelZoo
from myocr.predictors.classification_predictor import ImageClassificationParamConverter


def test_restnet():
    model = ModelZoo.load_model("pt", "resnet152", "cuda:0" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    p = model.predictor(ImageClassificationParamConverter(model.device))
    print(p.predict(Image.open("tests/images/flower.png").convert("RGB")))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"执行时间: {execution_time} 秒")


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason=f"{MODEL_PATH} not exist, skip test.")
def test_mlp():
    model = ModelZoo.load_model(
        "custom", "myocr/modeling/mlp.py", "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    start_time = time.time()
    p = model.predictor(
        ImageClassificationParamConverter(model.device, resize=28, center_crop=28, channels=1)
    )
    print(p.predict(Image.open("tests/images/flower.png").convert("L")))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"执行时间: {execution_time} 秒")
