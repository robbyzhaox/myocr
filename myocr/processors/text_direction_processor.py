import logging
from typing import List, Optional, Tuple

import numpy as np

from ..base import CompositeProcessor
from .base import PadNdarray, crop_rectangle

logger = logging.getLogger(__name__)


class TextDirectionProcessor(
    CompositeProcessor[Tuple[np.ndarray, List[Tuple]], Tuple[np.ndarray, List[Tuple]]]
):
    def __init__(self):
        super().__init__()
        self.labels = [0, 180]

    def preprocess(self, input_data: Tuple[np.ndarray, List[Tuple]]) -> Optional[np.ndarray]:
        self.image, self.bounding_boxes = input_data
        batch_tensors = []
        for box in self.bounding_boxes:  # type: ignore
            # TODO support dynamic height
            # h = box.bottom - box.top
            resized_img = crop_rectangle(self.image, box, target_height=48)
            # img_np is (h,w,c)
            img_np = np.array(resized_img, dtype=np.float32)
            img_np = (img_np / 255.0 - 0.5) / 0.5
            if img_np.ndim == 2:
                img_np = np.expand_dims(img_np, axis=0)
            batch_tensors.append(img_np)

        padded_batch = PadNdarray().process(batch_tensors)
        return padded_batch

    def postprocess(self, internal_result: np.ndarray) -> Tuple[np.ndarray, List[Tuple]]:
        preds = internal_result[0]
        pred_idxs = np.argmax(preds, axis=1)

        decode_out = [(self.labels[idx], float(preds[i, idx])) for i, idx in enumerate(pred_idxs)]

        box_with_angles: List[Tuple[Tuple, float]] = []
        for i in range(preds.shape[0]):
            box = self.bounding_boxes[i]
            box_with_angles.append((box, decode_out[i][0]))  # type: ignore

        return self.image, box_with_angles
