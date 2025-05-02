import logging

import cv2
import numpy as np
import pytest

from myocr.processors import base

logger = logging.getLogger(__name__)


@pytest.fixture
def img():
    flower = cv2.imread("tests/images/test_ocr2.png", cv2.IMREAD_COLOR_RGB)
    logger.debug(f"img loaded by cv2, shape:{flower.shape}")
    # H, W, C
    return flower


def test_ToTensor(img):
    H, W, C = img.shape
    # C, H, W
    res = base.ToTensor().process(img)
    shp = res.shape
    assert shp[0] == C
    assert shp[1] == H
    assert shp[2] == W
    assert np.all((res >= 0) & (res <= 1))


def test_ToNdarray(img):
    res = base.ToNdarray().process(img)
    logger.info(f"ndarray:{img.shape}, {res.shape}")
    assert res.shape[2] == img.shape[0]


def test_ImgNormalize(img):
    arr = base.ImgNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).process(img)
    assert np.all((arr >= -10) & (arr <= 1255))


def test_ImgResize(img):
    cliped = base.ImgResize((2, 10)).process(img)
    logger.info(f"resized img shape: {cliped.shape}")
    assert cliped.shape[0] == 2
    assert cliped.shape[1] == 10


def test_ImgCenterCrop(img):
    cliped = base.ImgCenterCrop((10, 2)).process(img)

    logger.info(f"croped img shape: {cliped.shape}")


def test_PadNdarray():
    img1 = np.ones((3, 10, 7))
    img2 = np.ones((3, 12, 7))
    img3 = np.ones((3, 8, 7))
    img4 = np.ones((3, 15, 7))

    batch_img = [img1, img2, img3, img4]

    paaded_img = base.PadNdarray().process(batch_img)
    logger.info(f"paaded_img shape: {paaded_img.shape}")
    assert paaded_img.shape[3] == 15


def test_LabelDecoder():
    from myocr.utils import LabelTranslator

    trans = LabelTranslator("这是一个测试This is Test")

    # 1 * 5 * 2 * 8
    data = np.array(
        [
            [
                [[1, 2, 3, 2, 1, 10, 2, 3], [7, 2, 3, 4, 1, 1, 2, 3]],
                [[1, 2, 3, 2, 1, 1, 2, 30], [7, 2, 3, 4, 1, 1, 2, 3]],
                [[1, 2, 3, 2, 1, 1, 2, 3], [7, 2, 3, 4, 1, 1, 20, 3]],
                [[1, 9, 3, 1, 1, 1, 20, 3], [8, 2, 3, 4, 1, 1, 2, 3]],
                [[9, 2, 3, 8, 1, 10, 2, 3], [2, 2, 4, 9, 10, 1, 2, 3]],
            ]
        ]
    )
    bounding_boxes = [1, 2, 3, 4, 5]
    decoded = base.LabelDecoder(trans, bounding_boxes).process(data)
    assert len(decoded) == 5


def create_sample_contour():
    return np.array(
        [
            [[10, 10]],
            [[15, 5]],
            [[20, 10]],
            [[10, 30]],
            [[15, 35]],
            [[20, 30]],
            [[50, 30]],
            [[55, 35]],
            [[60, 30]],
            [[50, 10]],
            [[55, 5]],
            [[60, 10]],
        ],
        dtype=np.int32,
    )


def test_ContourToBox_success():
    contour = create_sample_contour()

    bitmap = np.zeros((120, 120), dtype=np.float32)
    cv2.fillPoly(bitmap, [contour], 1.0)  # type: ignore
    box = base.ContourToBox(bitmap).process(contour)
    if box is not None:
        box, _ = box
    box = np.array(box)
    assert box is not None
    assert box.shape == (4, 2)


def test_ContourToBox_low_score():
    contour = create_sample_contour()

    bitmap = np.zeros((64, 64), dtype=np.float32)  # 全 0

    box = base.ContourToBox(bitmap).process(contour)
    assert box is None


def test_BoxScaling():
    box = np.array([[10, 10], [10, 30], [50, 30], [50, 10]])
    out = base.BoxScaling(0.8, 0.9, 20, 50).process(box)
    out = np.array(out)
    assert np.all(out[:, 0] < box[:, 0])
    assert np.all(out[:, 1] < box[:, 1])


def test_RotateImg(img):
    out = base.RotateImg(180).process(img)
    assert out is not None
