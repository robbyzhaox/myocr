import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..base import ParamConverter, Predictor


class Device:
    def __init__(self, device_name):
        self.name = device_name


class Model:
    def __init__(self, model_name, device: Device | str) -> None:
        self.name = model_name
        self.device = device
        self.loaded_model = None
        self.loaded = False

    def predictor(self, converter: Optional[ParamConverter]) -> Predictor:
        """
        build predictor by processors
        """
        predictor = Predictor(self, converter)
        return predictor

    def load(self, model_path) -> None:
        raise RuntimeError("method load should be implemented in sub class")


class PyTorchModel(Model):
    def __init__(self, model_name, device):
        super().__init__(model_name, device)

    def load(self, model_path) -> None:
        if self.loaded:
            return

        self.model_dir = model_path
        file = Path(self.model_dir)  # .joinpath("model.pt")
        if not file.exists():
            raise FileNotFoundError(f"model not found in {self.model_dir}")
        import torch

        if isinstance(self.device, Device):
            self.device = self.device.name

        self.loaded_model = torch.load(file, map_location=self.device, weights_only=False)
        print(f"model loaded from {model_path} to {self.device}")
        self.loaded = True


class ModelLoader(ABC):
    def __init__(self):
        super().__init__()

    def load(self, model_name, model_path, device: Device | str) -> Model:
        m = PyTorchModel(model_name, device)
        m.load(model_path)
        return m


class ModelZoo:
    model_loaders: Dict[str, ModelLoader] = {
        "onnx": ModelLoader(),
        "pt": ModelLoader(),
    }

    @staticmethod
    def _get_loader(group_id) -> ModelLoader:
        loader = ModelZoo.model_loaders.get(group_id)
        if loader is None:
            loader = ModelLoader()
        return loader

    @staticmethod
    def load_model(group_id, name, path, device: Device | str = Device("cpu")) -> Model:
        return ModelZoo._get_loader(group_id).load(name, path, device)
