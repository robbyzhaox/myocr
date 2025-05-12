import json

import cv2
import numpy as np

from myocr.types import OCRDocumentResult, OCRPageResult, RectBoundingBox


def test_rect_boundingbox():
    # img = cv2.imread("tests/images/test_ocr1.png", cv2.IMREAD_COLOR_RGB)
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 50), (180, 95), (0, 255, 0), 2)
    bbox = RectBoundingBox(left=95, top=45, right=180, bottom=95, angle=180, score=0.95)
    cropped = bbox.crop_image(img, target_height=48)
    h = cropped.shape[0]
    w = cropped.shape[1]
    assert w == bbox.width
    assert h == 48
    # cv2.imshow("Original", img)
    # cv2.imshow("Cropped", cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def test_OCRDocumentResult():

    with open("tests/ocr_result.json", "r", encoding="utf-8") as f:
        json_str = f.read()
    data = json.loads(json_str)
    page1: OCRPageResult = OCRPageResult.from_dict(data)
    page1.page_info = {
        "format": "text",
    }

    result = OCRDocumentResult([page1], {"test1": "value"})
    print(result.get_all_text())
    print(result.to_dict())
    assert len(result.get_all_text()) > 0
    assert len(result.to_dict()) > 0
