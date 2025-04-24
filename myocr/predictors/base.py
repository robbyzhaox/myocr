from typing import List, Optional

from PIL.Image import Image


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
        print(f"dict: {self.__dict__}")
        return self.__dict__


class DetectedObjects:
    def __init__(self, image: Image, boundingBoxes: Optional[List[RectBoundingBox]]):
        self.image = image
        self.bounding_boxes = boundingBoxes
