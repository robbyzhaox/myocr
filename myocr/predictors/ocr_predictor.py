import torch

from ..base import Predictor


class TextRecognizePredictor(Predictor):
    pass


class TextDectectPredictor(Predictor):
    def __init__(self, model_path, device):
        # self.model
        torch.load(model_path, map_location=device, weights_only=False)
        pass


class TableRecgnizePredictor(Predictor):
    pass
