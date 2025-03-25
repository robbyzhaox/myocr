import json
from typing import Optional

import torch
import torchvision.transforms as transforms
from PIL.Image import Image
from torch import Tensor

from myocr.base import ParamConverter


class Classification:

    def __init__(self, name, confidence):
        self.name = name
        self.confidence = confidence

    def __str__(self):
        return f"classification name: {self.name}, confidence: {self.confidence} \n"


class ImageClassificationParamConverter(ParamConverter[Image, Classification]):
    def __init__(self, device, cls_name_mapping: dict = {}):
        super().__init__()
        self.device = device
        self.transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.name = self.__class__

    def convert_input(self, input: Image) -> Optional[Tensor]:
        tensor = self.transforms(input).to(self.device)  # type: ignore
        batch_tensor = tensor.unsqueeze(0)
        return batch_tensor

    def convert_output(self, internal_result: Tensor) -> Optional[Classification]:
        probabilities = torch.nn.functional.softmax(internal_result[0], dim=0)
        top1_prob, top1_catid = torch.topk(probabilities, 1)
        with open("myocr/predictors/imagenet-simple-labels.json", "r") as f:
            imagenet_classes = json.load(f)
        predicted_class_name = imagenet_classes[top1_catid.item()]
        return Classification(predicted_class_name, top1_prob)
