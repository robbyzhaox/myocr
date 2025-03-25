import os

import pytest
import torch
from PIL import Image

from myocr.models.model import ModelZoo
from myocr.predictors.classification_predictor import ImageClassificationParamConverter


def test_hello():
    print("Hello, World!")


@pytest.mark.parametrize("iteration", range(1))
def test_model(iteration):

    model = ModelZoo.load_model("pt", "resnet18", "cuda:0" if torch.cuda.is_available() else "cpu")
    p = model.predictor(ImageClassificationParamConverter())
    print(os.getcwd())
    print(p.predict(Image.open("tests/flower.png").convert("RGB")))
