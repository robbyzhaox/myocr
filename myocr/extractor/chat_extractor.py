import logging
from typing import Optional

from pydantic import BaseModel

from .base import Extractor

logger = logging.getLogger(__name__)


class OpenAiChatExtractor(Extractor):

    def __init__(self, model, base_url, api_key):
        from openai import OpenAI

        super().__init__()
        self.model = model
        self.chat_client = OpenAI(api_key=api_key, base_url=base_url)

    def extract_with_format(self, content, response_format: BaseModel) -> Optional[BaseModel]:
        logger.debug(
            f"Extract infomation via OpanAI client with format:{response_format} from OCR content: \n{content}"
        )
        completion = self.chat_client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are good at extracting information from the invoice, please extract corresponding information from the given texts recognized via OCR",
                },
                {"role": "user", "content": content},
            ],
            response_format=response_format,  # type: ignore
        )
        return completion.choices[0].message.parsed
