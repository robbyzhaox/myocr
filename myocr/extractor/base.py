from typing import Optional

from pydantic import BaseModel


class Extractor:
    def extract(self, content):
        raise NotImplementedError("extract() not implemented")

    def extract_with_format(self, content, response_format: BaseModel) -> Optional[BaseModel]:
        raise NotImplementedError("extract_with_format() not implemented")
