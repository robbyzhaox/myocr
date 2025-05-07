import logging
from typing import List

import cv2
import pyclipper

from myocr.utils import softmax

from ..base import Processor
from ..types import NDArray, np

logger = logging.getLogger(__name__)


class ToTensor(Processor):
    def process(self, data: NDArray) -> NDArray:
        # in shape: (H, W, C)
        # out shape: (C, H, W)
        if data.dtype == np.uint8:
            data = data.astype(np.float32) / 255.0
        if data.ndim == 3:
            return np.transpose(data, (2, 0, 1))
        else:
            return data


class ImgNormalize(Processor):
    def __init__(self, mean: List[float], std: List[float]):
        super().__init__()
        self.mean = mean
        self.std = std

    def process(self, data: NDArray) -> NDArray:
        # RGB image, shape: [H, W, C]
        data = data.astype(np.float32) / 255.0
        if data.ndim == 3:
            mean = np.array(self.mean).reshape(1, 1, -1)
            std = np.array(self.std).reshape(1, 1, -1)
        elif data.ndim == 2:
            # Grayscale image
            mean = float(self.mean[0])
            std = float(self.std[0])
        else:
            raise ValueError(f"Unsupported image shape: {data.shape}")

        return (data - mean) / std


class ImgResize(Processor):
    """
    size: h,w
    """

    def __init__(self, size: tuple[int, int]):
        super().__init__()
        self.size = size

    def process(self, data: NDArray) -> np.ndarray:
        return cv2.resize(data, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)


class ResizeToMultipleOf(Processor):
    def __init__(self, multiple: int):
        super().__init__()
        self.multiple = multiple

    def process(self, data: NDArray) -> np.ndarray:
        W = data.shape[1] // self.multiple * self.multiple
        H = data.shape[0] // self.multiple * self.multiple
        return cv2.resize(data, (W, H), interpolation=cv2.INTER_LINEAR)


class ImgCenterCrop(Processor):
    """
    crop_size: h, w
    """

    def __init__(self, crop_size: tuple[int, int]):
        super().__init__()
        self.crop_size = crop_size

    def process(self, data: NDArray) -> NDArray:
        h, w = data.shape[:2]
        ch, cw = self.crop_size
        top = max((h - ch) // 2, 0)
        left = max((w - cw) // 2, 0)
        return data[top : top + ch, left : left + cw]


class ToNdarray(Processor):
    def __init__(self):
        super().__init__()

    def process(self, data) -> NDArray:
        import torch

        if isinstance(data, torch.Tensor):
            if data.ndim == 3:
                data = data.permute(1, 2, 0)  # CHW → HWC
            return data.detach().cpu().numpy()

        elif isinstance(data, np.ndarray):
            if data.ndim == 3:
                data = data.transpose(1, 2, 0)  # CHW → HWC
        return data


class RotateImg(Processor):
    def __init__(self, angle=0):
        super().__init__()
        self.angle = angle

    def process(self, data) -> NDArray:
        image = data
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        scale = 1.0
        M = cv2.getRotationMatrix2D(center, self.angle, scale)
        img_np = cv2.warpAffine(image, M, (w, h))

        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=0)
        return img_np.astype(np.float32)


class PadNdarray(Processor):
    """
    Pad ndarray to the max width
    """

    def process(self, data: List) -> NDArray:
        max_width = max(t.shape[1] for t in data)
        padded_batch = []
        for tensor in data:
            if tensor.ndim == 2:
                tensor = np.expand_dims(tensor, axis=0)

            pad_width = max_width - tensor.shape[1]
            if pad_width > 0:
                padded = np.pad(
                    tensor, ((0, 0), (0, pad_width), (0, 0)), mode="constant", constant_values=0
                )
            else:
                padded = tensor
            padded_batch.append(padded)
        # (B, C, H, W) eg: (8, 3, 32, 94)
        return np.stack(padded_batch, axis=0).transpose(0, 3, 1, 2)


class LabelDecoder(Processor):
    def __init__(self, translator, bounding_boxes):
        self.translator = translator
        self.bounding_boxes = bounding_boxes

    def process(self, data):
        preds = data[0]
        # preds shape: (Time Steps, Batch Size, Num Classes)
        preds = preds.transpose(1, 0, 2)  # type: ignore
        length = preds.shape[1]
        blank_label = 0
        decode_result = []
        for i in range(length):
            box = self.bounding_boxes[i]  # type: ignore
            sample_pred = preds[:, i, :]

            probs = softmax(sample_pred)

            length = sample_pred.shape[0]
            pred_indices = np.argmax(sample_pred, axis=1)

            non_blank_indices = pred_indices != blank_label

            char_confidences = np.max(probs, axis=-1)[non_blank_indices]
            if len(char_confidences) > 0:
                text_confidence = np.mean(char_confidences).item()
            else:
                text_confidence = 0.0

            text = self.translator.decode(pred_indices, length, raw=False)
            decode_result.append((text, text_confidence, box))
        return decode_result


class ContourToBox(Processor):
    def __init__(self, bitmap):
        super().__init__()
        self.bitmap = bitmap

    def _get_min_boxes(self, contour):
        rect = cv2.minAreaRect(contour)
        box_pts = cv2.boxPoints(rect)

        box_pts = sorted(box_pts, key=lambda x: x[0])
        if box_pts[0][1] > box_pts[1][1]:
            box_pts[0], box_pts[1] = box_pts[1], box_pts[0]
        if box_pts[2][1] < box_pts[3][1]:
            box_pts[2], box_pts[3] = box_pts[3], box_pts[2]

        return box_pts, min(rect[1])

    def _poly_area(self, points):
        area = 0.0
        for i in range(len(points)):
            j = (i + 1) % len(points)
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        return abs(area) / 2.0

    def _poly_perimeter(self, points):
        perimeter = 0.0
        for i in range(len(points)):
            j = (i + 1) % len(points)
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            perimeter += (dx * dx + dy * dy) ** 0.5
        return perimeter

    def _unclip(self, box_points, unclip_ratio=1.5):
        """输入多边形坐标[N,2]，返回扩展后的多边形坐标"""
        area = self._poly_area(box_points)
        perimeter = self._poly_perimeter(box_points)
        distance = area * unclip_ratio / perimeter
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

    def process(self, data):
        cnt = data

        if len(cnt) < 10:
            return None

        points, min_length = self._get_min_boxes(cnt)
        if min_length < 3:
            return None

        points = np.array(points)
        score = self._box_score_fast(self.bitmap, points.reshape(-1, 2))  # N * 2
        if score < 0.3:
            return None

        # 膨胀操作
        box = self._unclip(points, unclip_ratio=2.3).reshape(-1, 1, 2)
        box, min_length = self._get_min_boxes(box)
        if min_length < 5:
            return None
        return box, score


class BoxScaling(Processor):
    """
    scale_x = self.origin_w / binary_map.shape[1] #w
    scale_y = self.origin_h / binary_map.shape[0] #h
    """

    def __init__(self, scale_x, scale_y, origin_w, origin_h):
        super().__init__()
        self.scale_x = scale_x
        self.scale_y = scale_y

        self.origin_w = origin_w
        self.origin_h = origin_h

    def process(self, data):
        box = data
        box = np.array(box).astype(np.int32)
        # 等比例缩放
        box[:, 0] = np.clip(np.round(box[:, 0] * self.scale_x), 0, self.origin_w)
        box[:, 1] = np.clip(np.round(box[:, 1] * self.scale_y), 0, self.origin_h)
        return box
