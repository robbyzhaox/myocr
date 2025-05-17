from typing import Optional

from pydantic import BaseModel


class Extractor:
    """
    Extract structured information with provided format from the plain text content.
    """

    def extract(self, content, data_model: BaseModel) -> Optional[BaseModel]:
        raise NotImplementedError("extract() not implemented")
