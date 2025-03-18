import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from ..base import BaseProcessor
from ..config import detection_models, recognition_models
from ..models.DBNet.DBNet import DBNet
from ..models.vgg_model import Model
from ..util import diff, group_text_box, load_model


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
        bbox = self.dbnet.inference(input)
        polys_list = bbox
        text_box_list = [
            [np.array(box).astype(np.int32).reshape((-1)) for box in polys] for polys in polys_list
        ]

        min_size = 20
        horizontal_list_agg, free_list_agg = [], []

        for text_box in text_box_list:
            horizontal_list, free_list = group_text_box(text_box)
            if min_size:
                horizontal_list = [
                    i for i in horizontal_list if max(i[1] - i[0], i[3] - i[2]) > min_size
                ]
                free_list = [
                    i
                    for i in free_list
                    if max(diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size
                ]
            horizontal_list_agg.append(horizontal_list)
            free_list_agg.append(free_list)

        # the first result of the detection
        return horizontal_list_agg[0], free_list_agg[0]


class OcrRecognizationProcessor(OcrProcessor):
    def __init__(self):
        model_config = recognition_models["gen2"]["zh_sim_g2"]

        device = "cuda:0"

        # recog_network = "generation2"

        dict_character = list(model_config["characters"])

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1
        self.character = ["[blank]"] + dict_character
        load_model(model_config)
        network_params = {"input_channel": 1, "output_channel": 256, "hidden_size": 256}
        model_path = "/home/robby/.MyOCR//model/zh_sim_g2.pth"
        # state_dict = torch.load(model_path, map_location=device, weights_only=False)

        model = Model(num_class=len(self.character), **network_params)
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))

        print("init rec model")

    def process(self, input, **kwargs):
        # reformat image
        image = input
        if type(image) is np.ndarray:
            if len(image.shape) == 2:  # grayscale
                img_cv_grey = image
                img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                img_cv_grey = np.squeeze(image)
                img = cv2.cvtColor(img_cv_grey, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 3:  # BGRscale
                img = image
                img_cv_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBAscale
                img = image[:, :, :3]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # return super().process(input, **kwargs)


p = OcrRecognizationProcessor()
