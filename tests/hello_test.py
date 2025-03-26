import time

import pytest
import torch
from PIL import Image

from myocr.models.model import ModelZoo
from myocr.predictors.classification_predictor import ImageClassificationParamConverter


def test_hello():
    print("Hello, World!")


@pytest.fixture
def model():
    return ModelZoo.load_model("pt", "resnet152", "cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("iteration", range(1))
def test_model(iteration, model):
    start_time = time.time()
    p = model.predictor(ImageClassificationParamConverter(model.device))
    print(p.predict(Image.open("tests/flower.png").convert("RGB")))
    end_time = time.time()  # 记录循环结
    execution_time = end_time - start_time
    print(f"执行时间: {execution_time} 秒")
