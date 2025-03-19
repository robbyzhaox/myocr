import numpy as np
from myocr.processors import OcrDetectionProcessor, OcrRecognizationProcessor

def test_hello():
    print("Hello, World!")


def test_processor():
    image = "/home/robby/code/myocr/tests/123.png"
    p = OcrDetectionProcessor()

    img, origin_shape = p.dbnet.load_image(image)
    horizontal_list, free_list = p.process(image)
    img = img.numpy().squeeze()
    img = np.transpose(img, (2, 1, 0))

    p2 = OcrRecognizationProcessor(horizontal_list, free_list)
    p2.process(img)
