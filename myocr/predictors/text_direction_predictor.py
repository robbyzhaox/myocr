import logging
from typing import Optional

import numpy as np
from torch import Tensor

from myocr.util import crop_rectangle

from ..base import ParamConverter
from .base import DetectedObjects

logger = logging.getLogger(__name__)


class TextDirectionParamConverter(ParamConverter[DetectedObjects, DetectedObjects]):
    def __init__(self):
        super().__init__()
        self.labels = [0, 180]

    def convert_input(self, input_data: DetectedObjects) -> Optional[np.ndarray]:
        self.detected_objects = input_data
        self.bounding_boxes = input_data.bounding_boxes
        batch_tensors = []
        for box in input_data.bounding_boxes:  # type: ignore
            resized_img = crop_rectangle(input_data.image, box, target_height=48)
            box.set_croped_img(resized_img)
            img_np = np.array(resized_img, dtype=np.float32)
            img_np = (img_np / 255.0 - 0.5) / 0.5

            if img_np.ndim == 2:
                img_np = np.expand_dims(img_np, axis=0)

            batch_tensors.append(img_np)
        max_width = max(t.shape[1] for t in batch_tensors)

        padded_batch = []
        for tensor in batch_tensors:
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

    def convert_output(self, internal_result: Tensor | np.ndarray) -> Optional[DetectedObjects]:
        preds = internal_result[0]
        pred_idxs = np.argmax(preds, axis=1)

        decode_out = [(self.labels[idx], float(preds[i, idx])) for i, idx in enumerate(pred_idxs)]

        for i in range(preds.shape[0]):
            box = self.bounding_boxes[i]  # type: ignore
            box.angle = decode_out[i]

        return self.detected_objects
