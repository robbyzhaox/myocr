import torch

from ..base import BasePredictor


class TextRecognizePredictor(BasePredictor):
    pass


class TextDectectPredictor(BasePredictor):
    def __init__(self, model_path, device):
        # self.model
        torch.load(model_path, map_location=device, weights_only=False)
        pass


class TableRecgnizePredictor(BasePredictor):
    pass
