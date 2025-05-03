from typing import List, Optional, TypeAlias

import numpy as np

NDArray: TypeAlias = np.ndarray  # shape: (H, W, C), dtype: float32 or uint8
try:
    import torch

    Tensor: TypeAlias = torch.Tensor  # shape: (C, H, W), dtype: float32
except ImportError:
    pass


class BoundingBox:
    pass


class RectBoundingBox(BoundingBox):

    def __init__(
        self,
        left,
        bottom,
        right,
        top,
        score,
    ):
        self.left = left
        self.bottom = bottom
        self.right = right
        self.top = top
        self.score = score

    def get_width(self):
        return self.right - self.left

    def get_height(self):
        return self.bottom - self.top

    def set_angle(self, angle):
        self.angle = angle

    def set_croped_img(self, img):
        self.croped_img = img

    def get_croped_img(self):
        return self.croped_img

    def __str__(self):
        return f"(left={self.left}, bottom={self.bottom}, right={self.right}, top={self.top}, angle={self.angle}, score={self.score})"

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        self.__dict__.pop("croped_img", None)
        return self.__dict__


class DetectedObjects:
    def __init__(self, image: NDArray, boundingBoxes: Optional[List[RectBoundingBox]]):
        self.image = image
        self.bounding_boxes = boundingBoxes


class TextItem:
    def __init__(self, text, confidence, bounding_box: Optional[BoundingBox]):
        self.text = text
        self.confidence = confidence
        self.bounding_box = bounding_box

    def __str__(self):
        return (
            f"(text={self.text}, confidence={self.confidence}, bounding_box={self.bounding_box})\n"
        )

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        self.__dict__["bounding_box"] = self.bounding_box.to_dict()  # type: ignore
        return self.__dict__


class RecognizedTexts:
    def __init__(self):
        self.text_items: List[TextItem] = []

    def add_text(self, text, confidence, bounding_box: Optional[BoundingBox]) -> None:
        item = TextItem(text, confidence, bounding_box)
        self.text_items.append(item)

    def get_content_text(self):
        return "\n".join(map(str, [item.text for item in self.text_items]))

    def original(self, /, width, height):
        self.original_width = width
        self.original_height = height

    def __str__(self):
        return str(self.text_items)

    def to_dict(self):
        self.__dict__["text_items"] = [item.to_dict() for item in self.text_items]
        return self.__dict__


class Classification:

    def __init__(self, name, confidence):
        self.name = name
        self.confidence = confidence

    def __str__(self):
        return f"classification name: {self.name}, confidence: {self.confidence} \n"
