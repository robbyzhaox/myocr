import logging
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml  # type: ignore

from .architectures import build_model

logger = logging.getLogger(__name__)


class Device:
    def __init__(self, device_name):
        self.name = device_name


class Model:
    def __init__(self, device: Union[Device, str]) -> None:
        self.model_name_or_path = None
        self.device = device
        self.loaded_model = None
        self.loaded = False

    def forward_internal(self, *args, **kwargs):
        raise NotImplementedError("Should implement forward_internal in sub class")

    def __call__(self, *args, **kwargs):
        return self.forward_internal(*args, **kwargs)

    def load(self, model_name_or_path) -> None:
        raise NotImplementedError("method load should be implemented in sub class")

    def train(self) -> None:
        raise NotImplementedError("method train should be implemented in sub class")

    def eval(self) -> None:
        raise NotImplementedError("method eval should be implemented in sub class")

    def parameters(self) -> Any:
        raise NotImplementedError("method parameters should be implemented in sub class")

    def to_onnx(self, file_path: Union[str, Path], input_sample) -> None:
        raise NotImplementedError("method to_onnx should be implemented in sub class")


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

        providers: List[Union[tuple[str, Dict[str, Any]], str]] = []
        if isinstance(self.device, Device):
            self.device = self.device.name
        if "cpu" in self.device:  # type: ignore
            providers.append("CPUExecutionProvider")
        else:
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        #  'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                )
            )  # type: ignore

        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_profiling = False
        sess_options.log_severity_level = 3

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

        logger.info(
            f"""Onnx model {model_name_or_path} loaded to {self.device},
                    input output info: {get_input_output_info()}"""
        )
        self.loaded = True

    def forward_internal(self, *args, **kwargs):
        if args:
            input_name = self.session.get_inputs()[0].name
            input_data = {input_name: args[0]}
            outputs = self.session.run(None, input_data)
            return outputs
        else:
            raise IndexError("args wrong")


class PyTorchModel(Model):
    """
    PyTorchModel is responsible for load pytorch predefined models
    """

    def __init__(self, device):
        super().__init__(device)
        import torch

        self.torch = torch

    def load(self, model_name_or_path) -> None:
        if self.loaded:
            return

        # self.model_dir = Path(model_name_or_path)
        # file = Path(self.model_dir)  # .joinpath("model.pt")
        # if not file.exists():
        #     raise FileNotFoundError(f"model not found in {self.model_dir}")

        if isinstance(self.device, Device):
            self.device = self.device.name

        import torchvision
        from torch import nn

        model_fn = getattr(torchvision.models, model_name_or_path)
        self.loaded_model: nn.Module = model_fn()

        # state_dict = torch.load(file, map_location=self.device, weights_only=False)
        # load by config or name
        self.loaded_model.to(self.device)
        logger.info(f"Pytorch model {model_name_or_path} loaded to {self.device}")
        self.loaded = True

    def forward_internal(self, *args, **kwargs):
        if self.loaded_model:
            with self.torch.no_grad():
                return self.loaded_model(*args, **kwargs)
        else:
            raise RuntimeError("model not loaded")


class CustomModel(Model):
    """
    Custom models defined by yaml config
    """

    def __init__(self, device):
        super().__init__(device)
        import torch

        self.torch = torch
        if isinstance(self.device, Device):
            self.device = self.device.name

    def load(self, model_name_or_path, **kwargs) -> None:
        if self.loaded:
            return

        model_path = Path(model_name_or_path)

        with open(model_path, "r") as f:
            config = yaml.safe_load(f.read())

        import copy

        from torch import nn

        self.loaded_model: nn.Module = build_model(copy.deepcopy(config["Architecture"]))
        self.loaded_model.to(self.device)
        logger.info(f"loaded custom model from {model_path}")
        # pretrained model weights
        if "pretrained" in config and self.loaded_model:
            weight_path = config.pretrained
            self.loaded_model.load_state_dict(self.torch.load(weight_path))

    def forward_internal(self, *args, **kwargs):
        if self.loaded_model:
            # with torch.no_grad():
            return self.loaded_model(*args, **kwargs)
        else:
            raise RuntimeError("model not loaded")

    def train(self) -> None:
        if self.loaded_model:
            self.loaded_model.train()

    def eval(self) -> None:
        if self.loaded_model:
            self.loaded_model.eval()

    def parameters(self):
        if self.loaded_model:
            return self.loaded_model.parameters()

    def to_onnx(self, file_path: Union[str, Path], input_sample) -> None:
        file_path = str(file_path) if isinstance(file_path, Path) else file_path
        self.torch.onnx.export(
            self.loaded_model,
            input_sample,
            file_path,
            input_names=["input"],
            output_names=["outputs"],
            export_params=True,
            dynamic_axes={"input": {0: "batch_size", 2: "width", 3: "height"}},
        )
        logger.info(f"successfully exported model to {file_path}")


class ModelLoader(ABC):
    def __init__(self):
        super().__init__()

    def load(self, model_format, model_name_path, device: Union[Device, str]) -> Model:
        m: Model
        if model_format == "pt":
            m = PyTorchModel(device)
        elif model_format == "onnx":
            m = OrtModel(device)
        elif model_format == "custom":
            m = CustomModel(device)
        else:
            raise RuntimeError(f"model format {model_format} not supported")
        m.load(model_name_path)
        return m


class ModelZoo:
    default_loader = ModelLoader()
    model_loaders: Dict[str, ModelLoader] = {
        "onnx": default_loader,
        "pt": default_loader,
        "custom": default_loader,
    }

    @staticmethod
    def _get_loader(group_id) -> ModelLoader:
        loader = ModelZoo.model_loaders.get(group_id)
        if loader is None:
            loader = ModelLoader()
        return loader

    @staticmethod
    def load_model(
        group_id, model_name_or_path, device: Union[Device, str] = Device("cpu")
    ) -> Model:
        return ModelZoo._get_loader(group_id).load(group_id, model_name_or_path, device)
