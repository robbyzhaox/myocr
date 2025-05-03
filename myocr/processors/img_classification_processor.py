import json
from typing import Optional

import numpy as np

from myocr.base import CompositeProcessor
from myocr.types import Classification

from .base import ImgCenterCrop, ImgNormalize, ImgResize, ToTensor


class ImageClassificationProcessor(CompositeProcessor[np.ndarray, Classification]):
    def __init__(self, device, resize=256, center_crop=224, channels=3):
        super().__init__()
        self.device = device
        self.resize = resize
        self.center_crop = center_crop
        self.channels = channels
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        if channels == 1:
            self.mean = [0.5]
            self.std = [0.5]
        self.name = self.__class__

        import torch

        self.torch = torch

    def preprocess(self, input_data: np.ndarray) -> Optional[np.ndarray]:
        img_np = ImgResize((self.resize, self.resize)).process(input_data)
        img_np = ImgCenterCrop((self.center_crop, self.center_crop)).process(img_np)
        img_np = ImgNormalize(mean=self.mean, std=self.std).process(img_np)
        img_np = ToTensor().process(img_np)

        batch_tensor = img_np[np.newaxis, :, :]
        print(f"input.size={batch_tensor.shape}")
        return batch_tensor.astype(np.float32)

    def postprocess(self, internal_result: np.ndarray) -> Optional[Classification]:
        probabilities = self.torch.nn.functional.softmax(internal_result[0], dim=0)
        top1_prob, top1_catid = self.torch.topk(probabilities, 1)
        with open("myocr/processors/imagenet-simple-labels.json", "r") as f:
            imagenet_classes = json.load(f)
        predicted_class_name = imagenet_classes[top1_catid.item()]
        return Classification(predicted_class_name, top1_prob)


class RestNetImageClassificationProcessor(ImageClassificationProcessor):
    def __init__(self, device, resize=256, center_crop=224, channels=3):
        super().__init__(device, resize, center_crop, channels)

    def preprocess(self, input_data):
        data = super().preprocess(input_data)
        return self.torch.from_numpy(data)
