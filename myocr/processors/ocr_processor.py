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
    def __init__(self, horizontal_list, free_list):
        self.horizontal_list = horizontal_list
        self.free_list = free_list

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

        self.model = Model(num_class=len(self.character), **network_params)
        self.model = torch.nn.DataParallel(self.model).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))

        print("init rec model")

    def process(self, input, **kwargs):
        # reformat image
        image = input
        # if type(image) is np.ndarray:
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

        self.model.eval()

        for bbox in self.horizontal_list:
            h_list = [bbox]

            for box in h_list:
                x_min = max(0, box[0])
                x_max = box[1]
                y_min = max(0, box[2])
                y_max = box[3]
                crop_img = img_cv_grey[y_min:y_max, x_min:x_max]
                # print(f"x_min={x_min},x_max={x_max},y_min={y_min},y_max={y_max}, crop_img.shape={crop_img.shape}")
                image_list = []
                image_list.append(
                    ([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], crop_img)
                )
                [item[0] for item in image_list]
                img_list = [item[1] for item in image_list]
                tensor = torch.tensor(img_list)
                tensor = tensor.unsqueeze(0)
                print(f"tensor shape:{tensor.shape}")

                text_for_pred = torch.LongTensor(1, 1).fill_(0).to("cuda:0")

                preds = self.model(tensor, text_for_pred)
                print(f"preds is {preds}")
        # with torch.no_grad():

        # return super().process(input, **kwargs)
