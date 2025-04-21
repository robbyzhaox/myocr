__all__ = ["build_model"]
from typing import Any


def build_model(config, **kwargs) -> Any:
    import copy

    from .base_model import BaseModel

    config = copy.deepcopy(config)
    module_class = BaseModel(config, **kwargs)
    return module_class
