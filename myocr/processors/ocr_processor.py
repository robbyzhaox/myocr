import os

import torch
import torch.backends.cudnn as cudnn

from ..base import BaseProcessor
from ..config import MODULE_PATH
from ..models.DBNet.DBNet import DBNet
from ..util import load_model


class OcrProcessor(BaseProcessor):
    def __init__(self):
        pass


class OcrDetectionProcessor(OcrProcessor):
    def __init__(self, backbone="resnet18", device="cuda:0", model_storage_directory=None):
        dbnet = DBNet(
            initialize_model=False,
            dynamic_import_relative_path=os.path.join("myocr", "models", "DBNet"),
            device=device,
            verbose=0,
        )
        print(dbnet.configs[backbone])
        load_model(dbnet.configs[backbone])

        dbnet.initialize_model(dbnet.configs[backbone]["model"], weight_path=MODULE_PATH + "model")

        dbnet.model = torch.nn.DataParallel(dbnet.model).to(device)  # type: ignore
        cudnn.benchmark = False

        dbnet.model.eval()

    def process(self, input, **kwargs):
        return super().process(input, **kwargs)


p = OcrDetectionProcessor()
