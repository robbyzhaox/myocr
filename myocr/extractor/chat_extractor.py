import logging
from typing import Optional

from pydantic import BaseModel

from .base import Extractor

logger = logging.getLogger(__name__)


class OpenAiChatExtractor(Extractor):
    """
    Currently support
        - OpenAI API
        - Ollama API
    """

    def __init__(self, model, base_url, api_key, sys_prompt):
        from openai import OpenAI

        super().__init__()
        self.model = model
        self.chat_client = OpenAI(api_key=api_key, base_url=base_url)
        self.sys_prompt = sys_prompt

    def extract(self, content, data_model: BaseModel) -> Optional[BaseModel]:
        logger.debug(
            f"Extract infomation via OpanAI client with format:{data_model} from OCR content: \n{content}"
        )
        completion = self.chat_client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.sys_prompt,
                },
                {"role": "user", "content": content},
            ],
            response_format=data_model,  # type: ignore
        )
        return completion.choices[0].message.parsed
