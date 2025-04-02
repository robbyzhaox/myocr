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
    def __init__(self, device, resize=256, center_crop=224, channels=3):
        super().__init__()
        self.device = device
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if channels == 1:
            mean = [0.5]
            std = [0.5]
        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(center_crop),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.name = self.__class__

    def convert_input(self, input: Image) -> Optional[Tensor]:

        tensor = self.transforms(input).to(self.device)  # type: ignore
        batch_tensor = tensor.unsqueeze(0)
        print(f"input.size={batch_tensor.shape}")
        return batch_tensor

    def convert_output(self, internal_result: Tensor) -> Optional[Classification]:
        probabilities = torch.nn.functional.softmax(internal_result[0], dim=0)
        top1_prob, top1_catid = torch.topk(probabilities, 1)
        with open("myocr/predictors/imagenet-simple-labels.json", "r") as f:
            imagenet_classes = json.load(f)
        predicted_class_name = imagenet_classes[top1_catid.item()]
        return Classification(predicted_class_name, top1_prob)
