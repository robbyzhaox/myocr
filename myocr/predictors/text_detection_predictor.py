import logging
from typing import List, Optional, Union

import cv2
import numpy as np
import pyclipper
from PIL import Image as PIL
from PIL.Image import Image
from shapely.geometry import Polygon
from torch import Tensor

from myocr.base import ParamConverter
from myocr.predictors.base import RectBoundingBox

logger = logging.getLogger(__name__)


class DetectedObjects:
    def __init__(self, image: Image, boundingBoxes: Optional[List[RectBoundingBox]]):
        self.image = image
        self.bounding_boxes = boundingBoxes


class TextDetectionParamConverter(ParamConverter[Image, DetectedObjects]):
    def __init__(self, device, cls_name_mapping: dict = {}):
        super().__init__()
        self.device = device
        self.name = self.__class__
        self.origin_image = None
        self.origin_w = 0
        self.origin_h = 0

    def convert_input(self, input_data: Image) -> Optional[np.ndarray]:
        self.origin_image = input_data
        self.origin_w = input_data.size[0]
        self.origin_h = input_data.size[1]
        image_resized = input_data.resize(
            ((self.origin_w // 32) * 32, (self.origin_h // 32) * 32), PIL.Resampling.BILINEAR
        )
        image_np = np.array(image_resized).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = (image_np - mean) / std
        image_np = image_np.transpose(2, 0, 1)[np.newaxis, :, :, :]
        return image_np.astype(np.float32)

    def _get_min_boxes(self, contour):
        rect = cv2.minAreaRect(contour)
        box_pts = cv2.boxPoints(rect)

        box_pts = sorted(box_pts, key=lambda x: x[0])
        if box_pts[0][1] > box_pts[1][1]:
            box_pts[0], box_pts[1] = box_pts[1], box_pts[0]
        if box_pts[2][1] < box_pts[3][1]:
            box_pts[2], box_pts[3] = box_pts[3], box_pts[2]

        return box_pts, min(rect[1])

    def _unclip(self, box_points, unclip_ratio=1.5):
        """输入多边形坐标[N,2]，返回扩展后的多边形坐标"""
        poly = Polygon(box_points)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()  # type: ignore
        offset.AddPath(box_points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)  # type: ignore
        expanded = offset.Execute(distance)
        if not expanded:
            return box_points
        return np.array(expanded[0])

    def _box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        coords_min = np.floor(box.min(axis=0)).astype(np.int32)
        coords_max = np.ceil(box.max(axis=0)).astype(np.int32)
        xmin, ymin = np.clip(coords_min, 0, [w - 1, h - 1])
        xmax, ymax = np.clip(coords_max, 0, [w - 1, h - 1])

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)  # type: ignore
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def convert_output(
        self, internal_result: Union[Tensor, np.ndarray]
    ) -> Optional[DetectedObjects]:
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
            if len(cnt) < 10:
                continue

            points, min_length = self._get_min_boxes(cnt)
            if min_length < 3:
                continue

            points = np.array(points)
            score = self._box_score_fast(output, points.reshape(-1, 2))  # N * 2
            if score < 0.3:
                continue

            # 膨胀操作
            box = self._unclip(points, unclip_ratio=2.3).reshape(-1, 1, 2)
            box, min_length = self._get_min_boxes(box)
            if min_length < 5:
                continue

            # 等比例缩放
            box = np.array(box).astype(np.int32)
            box[:, 0] = np.clip(np.round(box[:, 0] * scale_x), 0, binary_map.shape[1])
            box[:, 1] = np.clip(np.round(box[:, 1] * scale_y), 0, binary_map.shape[0])
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
