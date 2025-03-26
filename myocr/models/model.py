import importlib.util
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torchvision
from torch import nn

from ..base import ParamConverter, Predictor


def is_cuda_available():
    return torch.cuda.is_available()


class Device:
    def __init__(self, device_name):
        self.name = device_name


class Model:
    def __init__(self, device: Device | str) -> None:
        self.model_name_or_path = None
        self.device = device
        self.loaded_model = None
        self.loaded = False

    def predictor(self, converter: Optional[ParamConverter]) -> Predictor:
        """
        build predictor by processors
        """
        predictor = Predictor(self, converter)
        return predictor

    def __call__(self, *args, **kwds):
        if self.loaded_model:
            with torch.no_grad():
                return self.loaded_model(*args, **kwds)
        else:
            raise RuntimeError("model not loaded")

    def load(self, model_name_or_path) -> None:
        raise RuntimeError("method load should be implemented in sub class")


class PyTorchModel(Model):
    def __init__(self, device):
        super().__init__(device)

    def load(self, model_name_or_path) -> None:
        if self.loaded:
            return

        # self.model_dir = Path(model_name_or_path)
        # file = Path(self.model_dir)  # .joinpath("model.pt")
        # if not file.exists():
        #     raise FileNotFoundError(f"model not found in {self.model_dir}")

        if isinstance(self.device, Device):
            self.device = self.device.name

        model_fn = getattr(torchvision.models, model_name_or_path)
        self.loaded_model: nn.Module = model_fn()

        # state_dict = torch.load(file, map_location=self.device, weights_only=False)
        # load by config or name
        self.loaded_model.to(self.device)
        print(f"model {model_name_or_path} loaded to {self.device}")
        self.loaded = True


class CustomModel(Model):
    def __init__(self, device):
        super().__init__(device)

    def load(self, model_name_or_path, **kwargs) -> None:
        if self.loaded:
            return

        model_path = Path(model_name_or_path)
        spec = importlib.util.spec_from_file_location("custom_model", model_path)
        if spec:
            module = importlib.util.module_from_spec(spec)
            if spec.loader:
                spec.loader.exec_module(
                    module,
                )

                # model name 'CustomModel'
                model_class = getattr(module, "CustomModel")
                model = model_class(**kwargs)

        # model weights
        # if pretrained:
        #     weight_path = model_path.parent / "weights.pth"
        #     model.load_state_dict(torch.load(weight_path))


class ModelLoader(ABC):
    def __init__(self):
        super().__init__()

    def load(self, model_name_path, device: Device | str) -> Model:
        m = PyTorchModel(device)
        m.load(model_name_path)
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
    def load_model(group_id, model_name_or_path, device: Device | str = Device("cpu")) -> Model:
        return ModelZoo._get_loader(group_id).load(model_name_or_path, device)
