import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..base import CompositeProcessor
from .base import BoxScaling, ContourToBox, ImgNormalize, ResizeToMultipleOf, ToTensor

logger = logging.getLogger(__name__)


class TextDetectionProcessor(CompositeProcessor[np.ndarray, Tuple[np.ndarray, List[Tuple]]]):
    def __init__(self, device, cls_name_mapping: dict = {}):
        super().__init__()
        self.device = device
        self.name = self.__class__
        self.origin_image = None
        self.origin_w = 0
        self.origin_h = 0
        self.resizer = ResizeToMultipleOf(32)
        self.normalizer = ImgNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def preprocess(self, input_data: np.ndarray) -> Optional[np.ndarray]:
        if input_data.ndim == 3:
            self.origin_w = input_data.shape[1]
            self.origin_h = input_data.shape[0]
        else:
            raise RuntimeError("input img must have 3 dim")
        self.origin_image = input_data
        image_resized = self.resizer.process(input_data)
        image_resized = self.normalizer.process(image_resized)
        image_np = ToTensor().process(image_resized)
        image_np = image_np[np.newaxis, :, :, :]
        return image_np.astype(np.float32)

    def postprocess(self, internal_result: np.ndarray) -> Tuple[np.ndarray, List[Tuple]]:
        assert self.origin_image is not None, "origin_image is none"
        output = internal_result[0]
        output = output[0, 0]
        logger.debug(f"text detection output shape: {output.shape}")
        threshold = 0.3
        binary_map = (output > threshold).astype(np.uint8) * 255  # type: ignore

        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        scale_x = self.origin_w / binary_map.shape[1]
        scale_y = self.origin_h / binary_map.shape[0]
        for cnt in contours:
            result = ContourToBox(output).process(cnt)
            if not result:
                continue

            box, score = result
            box = BoxScaling(scale_x, scale_y, self.origin_w, self.origin_h).process(box)

            # left, top, right, bottom, detection score
            box = (int(box[0][0]), int(box[0][1]), int(box[2][0]), int(box[2][1]), float(score))
            boxes.append(box)
        if not boxes:
            return self.origin_image, []
        # sort box
        boxes = sorted(boxes, key=lambda box: (box[1], box[0]))
        return self.origin_image, boxes
