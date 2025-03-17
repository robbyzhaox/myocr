import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from ..base import BaseProcessor
from ..config import detection_models
from ..models.DBNet.DBNet import DBNet
from ..util import load_model


class OcrProcessor(BaseProcessor):
    def __init__(self):
        pass


class OcrDetectionProcessor(OcrProcessor):
    def __init__(self, backbone="resnet18", device="cuda:0", model_storage_directory=None):
        self.dbnet = DBNet(
            initialize_model=False,
            dynamic_import_relative_path=os.path.join("myocr", "models", "DBNet"),
            device=device,
            verbose=0,
        )
        load_model(detection_models["dbnet18"])

        resnet_pth = "/home/robby/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth"

        self.dbnet.initialize_model(self.dbnet.configs[backbone]["model"], weight_path=resnet_pth)

        self.dbnet.model = torch.nn.DataParallel(self.dbnet.model).to(device)  # type: ignore
        cudnn.benchmark = False

        self.dbnet.model.eval()

    def process(self, input: np.ndarray, **kwargs):
        return self.dbnet.inference(input)


p = OcrDetectionProcessor()
