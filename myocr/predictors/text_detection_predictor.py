from typing import List, Optional

import numpy as np
from PIL import Image as PIL
from PIL.Image import Image
from torch import Tensor

from myocr.base import ParamConverter
from myocr.predictors.base import RectBoundingBox


class DetectedObjects:
    def __init__(self, image: Image, binary_map, boundingBoxes: Optional[List[RectBoundingBox]]):
        self.image = image
        self.binary_map = binary_map
        self.bounding_boxes = boundingBoxes


class TextDetectionParamConverter(ParamConverter[Image, DetectedObjects]):
    def __init__(self, device, cls_name_mapping: dict = {}):
        super().__init__()
        self.device = device
        self.name = self.__class__

    def convert_input(self, input: Image) -> Optional[np.ndarray]:
        self.origin_image = input
        self.origin_w = input.size[0]
        self.origin_h = input.size[1]
        image_resized = input.resize(
            ((self.origin_w // 32) * 32, (self.origin_h // 32) * 32), PIL.Resampling.BILINEAR
        )
        image_np = np.array(image_resized).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = (image_np - mean) / std
        image_np = image_np.transpose(2, 0, 1)[np.newaxis, :, :, :]
        return image_np.astype(np.float32)

    def convert_output(self, internal_result: Tensor | np.ndarray) -> Optional[DetectedObjects]:
        output = internal_result[0]
        output = output[0, 0]

        threshold = 0.3
        binary_map = (output > threshold).astype(np.uint8) * 255  # type: ignore
        self.binary_map = binary_map
        from scipy.ndimage import label

        labeled_array, num_features = label(binary_map)  # type: ignore

        boxes = []
        scale_x = self.origin_w / binary_map.shape[1]
        scale_y = self.origin_h / binary_map.shape[0]

        for i in range(1, num_features + 1):
            points = np.argwhere(labeled_array == i)
            if len(points) < 10:  # 过滤小区域
                continue
            # 计算最小外接矩形（替代OpenCV的minAreaRect）
            min_y, min_x = np.min(points, axis=0) - (7, 7)  # 临时修复
            max_y, max_x = np.max(points, axis=0) + (7, 7)

            box = RectBoundingBox(
                left=round(min_x * scale_x),
                bottom=round(max_y * scale_y),
                right=round(max_x * scale_x),
                top=round(min_y * scale_y),
            )
            boxes.append(box)
        if not boxes:
            return None
        return DetectedObjects(self.origin_image, self.binary_map, boundingBoxes=boxes)
