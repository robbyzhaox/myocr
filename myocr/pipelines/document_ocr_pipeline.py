import logging

from myocr.pipelines.common_ocr_pipeline import CommonOCRPipeline

logger = logging.getLogger(__name__)


class DocumentOCRPipeline(CommonOCRPipeline):
    def __init__(self, device, json_schema):
        super().__init__(device)

    def process(self, img_path: str):
        rec = super().process(img_path)

        return rec
