import logging
from typing import Union

import numpy as np

from .common_ocr_pipeline import CommonOCRPipeline

logger = logging.getLogger(__name__)


class DocumentOCRPipeline(CommonOCRPipeline):
    def __init__(self, device, json_schema):
        super().__init__(device)

    def process(self, img: Union[bytes, str, np.ndarray]):
        rec = super().process(img)

        return rec
