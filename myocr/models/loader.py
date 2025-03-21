from abc import ABC, abstractmethod

from myocr.base import Device
from myocr.models.model import Model


class ModelLoader(ABC):
    def __init__(self):
        super().__init__()

    def load(self, model_path) -> Model:
        return Model("model_name", "model_path", Device(""))
