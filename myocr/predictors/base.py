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

    def __str__(self):
        return f"(left={self.left}, bottom={self.bottom}, right={self.right}, top={self.top}, score={self.score})"

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return self.__dict__
