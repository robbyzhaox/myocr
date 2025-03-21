import re
from abc import ABC, abstractmethod
from typing import Optional

from myocr.models.loader import ModelLoader

from ..base import Device, ParamConverter, Predictor


class Model:
    def __init__(self, model_name, model_path, device: Device) -> None:
        self.name = model_name
        self.path = model_path
        self.device = device
        self.loaded_model = None

    def predictor(self, converter: Optional[ParamConverter]) -> Predictor:
        """
        build predictor by processors
        """
        predictor = Predictor(self, converter)
        return predictor


class ModelZoo(ABC):
    def __init__(self):
        super().__init__()


class ModelProvider(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_zoo(self) -> ModelZoo:
        pass


class OnnxModelProvider(ModelProvider):
    def __init__(self, name):
        self.name = name

    def get_zoo(self):
        return OnnxModelZoo()


class OnnxModelZoo(ModelZoo):
    def __init__(self):
        self.model_loader = ModelLoader()
        self.model_provider = OnnxModelProvider("onnx")

    def load_model(self, name):
        return self.model_loader.load("")
