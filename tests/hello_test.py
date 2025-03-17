import pytest

from myocr.processors import OcrDetectionProcessor


def test_hello():
    print("Hello, World!")


@pytest.mark.skip
def test_processor():
    image = "/home/robby/code/myocr/tests/123.png"
    p = OcrDetectionProcessor()
    # img, _ =p.dbnet.load_image(image, detection_size=100)
    p.process(image)
