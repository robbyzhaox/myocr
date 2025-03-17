import numpy as np
import cv2
import os
from myocr.processors import OcrDetectionProcessor

def test_hello():
    print("Hello, World!")

def test_processor():
    image = "/home/robby/code/myocr/tests/123.png"
    p = OcrDetectionProcessor()
    # img, _ =p.dbnet.load_image(image, detection_size=100)
    res = p.process(image)
