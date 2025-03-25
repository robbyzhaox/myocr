import pytest
import torch
from PIL import Image

from myocr.models.model import ModelZoo
from myocr.predictors.classification_predictor import ImageClassificationParamConverter


def test_hello():
    print("Hello, World!")


@pytest.mark.parametrize("iteration", range(1))
def test_model(iteration):
    model = ModelZoo.load_model("pt", "resnet152", "cuda:0" if torch.cuda.is_available() else "cpu")
    p = model.predictor(ImageClassificationParamConverter(model.device))
    print(p.predict(Image.open("tests/flower.png").convert("RGB")))
