import logging

import yaml  # type: ignore

from myocr.extractor.chat_extractor import OpenAiChatExtractor
from myocr.pipelines.common_ocr_pipeline import CommonOCRPipeline

logger = logging.getLogger(__name__)


class StructuredOutputOCRPipeline(CommonOCRPipeline):
    def __init__(self, device, json_schema):
        super().__init__(device)

        parts = __file__.split(".")[0].rsplit("/", 1)
        with open(parts[0] + "/config/" + parts[1] + ".yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.extractor = OpenAiChatExtractor(
            model=config["chat_bot"]["model"],
            base_url=config["chat_bot"]["base_url"],
            api_key=config["chat_bot"]["api_key"],
        )
        self.set_response_format(json_schema)

    def process(self, img_path: str):
        rec = super().process(img_path)
        return self.extractor.extract_with_format(rec.get_content_text(), self.response_format)

    def set_response_format(self, json_schema):
        from pydantic import BaseModel

        self.response_format: BaseModel = json_schema
