import logging
from typing import Optional

import cv2
import numpy as np

from myocr.base import CompositeProcessor
from myocr.types import DetectedObjects, RectBoundingBox

from .base import BoxScaling, ContourToBox, ImgNormalize, ResizeToMultipleOf, ToTensor

logger = logging.getLogger(__name__)


class TextDetectionProcessor(CompositeProcessor[np.ndarray, DetectedObjects]):
    def __init__(self, device, cls_name_mapping: dict = {}):
        super().__init__()
        self.device = device
        self.name = self.__class__
        self.origin_image = None
        self.origin_w = 0
        self.origin_h = 0

    def preprocess(self, input_data: np.ndarray) -> Optional[np.ndarray]:
        self.origin_image = input_data
        if input_data.ndim == 3:
            self.origin_w = input_data.shape[1]
            self.origin_h = input_data.shape[0]
        else:
            raise RuntimeError("input img must have 3 dim")

        image_resized = ResizeToMultipleOf(32).process(input_data)
        image_resized = ImgNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).process(
            image_resized
        )
        image_np = ToTensor().process(image_resized)
        image_np = image_np[np.newaxis, :, :, :]
        return image_np.astype(np.float32)

    def postprocess(self, internal_result: np.ndarray) -> Optional[DetectedObjects]:
        if self.origin_image is None:
            return None

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

            box = RectBoundingBox(
                left=int(box[0][0]),
                bottom=int(box[2][1]),
                right=int(box[2][0]),
                top=int(box[0][1]),
                score=float(score),
            )
            boxes.append(box)
        if not boxes:
            return None
        # sort box
        boxes = sorted(boxes, key=lambda box: (box.top, box.left))
        return DetectedObjects(self.origin_image, boundingBoxes=boxes)
