class BoundingBox:
    def __init__(self, text=None):
        self.text = text


class RectBoundingBox(BoundingBox):

    def __init__(self, left, bottom, right, top, text=None):
        super().__init__(text)
        self.left = left
        self.bottom = bottom
        self.right = right
        self.top = top

    def get_width(self):
        return self.right - self.left

    def get_height(self):
        return self.top - self.bottom
