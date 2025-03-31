from typing import Optional

from openai import OpenAI
from pydantic import BaseModel

from .base import Extractor


class OpenAiChatExtractor(Extractor):
    def __init__(self, model, base_url, api_key):
        super().__init__()
        self.model = model
        self.chat_client = OpenAI(api_key=api_key, base_url=base_url)

    def extract_with_format(self, content, response_format: BaseModel) -> Optional[BaseModel]:
        completion = self.chat_client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Extract the invoice information from the given texts recognized via OCR",
                },
                {"role": "user", "content": content},
            ],
            response_format=response_format,  # type: ignore
        )
        return completion.choices[0].message.parsed
