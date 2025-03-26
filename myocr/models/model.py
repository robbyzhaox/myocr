import importlib.util
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import onnxruntime as ort
import torch
import torchvision
from torch import nn

from ..base import ParamConverter, Predictor

logging.basicConfig(level=logging.INFO)


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

    def forward_internal(self, *args, **kwds):
        raise RuntimeError("Should implement forward_internal in sub class")

    def __call__(self, *args, **kwds):
        return self.forward_internal(*args, **kwds)

    def load(self, model_name_or_path) -> None:
        raise RuntimeError("method load should be implemented in sub class")


class OrtModel(Model):

    def __init__(self, device):
        super().__init__(device)

    def load(self, model_name_or_path) -> None:
        if self.loaded:
            return

        self.model_path = Path(model_name_or_path)
        file = Path(self.model_path)
        if not file.exists():
            raise FileNotFoundError(f"Onnx model not found in {self.model_path}")

        providers: List[tuple[str, Dict[str, Any]] | str] = []
        if isinstance(self.device, Device):
            self.device = self.device.name
        if "cpu" in self.device:  # type: ignore
            providers.append("CPUExecutionProvider")
        else:
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": self.device,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        #  'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                )
            )  # type: ignore

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_profiling = False

        self.session = ort.InferenceSession(
            self.model_path, sess_options=sess_options, providers=providers
        )

        def get_input_output_info() -> Dict:
            input_info = {}
            output_info = {}

            for input_meta in self.session.get_inputs():
                input_info[input_meta.name] = {
                    "shape": input_meta.shape,
                    "type": input_meta.type,
                }

            for output_meta in self.session.get_outputs():
                output_info[output_meta.name] = {
                    "shape": output_meta.shape,
                    "type": output_meta.type,
                }

            return {
                "inputs": input_info,
                "outputs": output_info,
            }

        logging.info(f"Onnx model {model_name_or_path} loaded to {self.device}")
        self.loaded = True

    def forward_internal(self, *args, **kwds):
        if args:
            input_data = {"x": args[0]}
            outputs = self.session.run(None, input_data)
            return outputs
        else:
            raise IndexError("args wrong")


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
        print(f"Pytorch model {model_name_or_path} loaded to {self.device}")
        self.loaded = True

    def forward_internal(self, *args, **kwds):
        if self.loaded_model:
            with torch.no_grad():
                return self.loaded_model(*args, **kwds)
        else:
            raise RuntimeError("model not loaded")


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

    def load(self, model_format, model_name_path, device: Device | str) -> Model:
        m: Model
        if model_format == "pt":
            m = PyTorchModel(device)
        elif model_format == "onnx":
            m = OrtModel(device)
        else:
            raise RuntimeError(f"model format {model_format} not supported")
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
        return ModelZoo._get_loader(group_id).load(group_id, model_name_or_path, device)
