import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml  # type: ignore
from pydantic import BaseModel

from ..base import Pipeline
from ..extractor.chat_extractor import OpenAiChatExtractor
from ..pipelines.common_ocr_pipeline import CommonOCRPipeline

logger = logging.getLogger(__name__)


class StructuredOutputOCRPipeline(Pipeline):
    def __init__(self, device, json_schema):
        self.ocr = CommonOCRPipeline(device)

        current_file = Path(__file__)
        config_path = current_file.parent / "config" / f"{current_file.stem}.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        model = os.getenv("CHAT_BOT_MODEL", config["chat_bot"]["model"])
        base_url = os.getenv("CHAT_BOT_BASEURL", config["chat_bot"]["base_url"])
        api_key = os.getenv("CHAT_BOT_APIKEY", config["chat_bot"]["api_key"])
        sys_prompt = config["chat_bot"]["sys_prompt"]
        logger.info(f"Init structured_output_pipeline with model:{model}, base_url:{base_url}")
        self.extractor = OpenAiChatExtractor(
            model=model, base_url=base_url, api_key=api_key, sys_prompt=sys_prompt
        )
        self.set_response_format(json_schema)

    def process(self, img: Union[bytes, str, np.ndarray]) -> Optional[BaseModel]:
        rec = self.ocr(img)
        text = rec.get_plain_text()
        if text is not None:
            return self.extractor.extract(text, self.response_format)
        return None

    def set_response_format(self, json_schema):
        from pydantic import BaseModel

        self.response_format: BaseModel = json_schema
