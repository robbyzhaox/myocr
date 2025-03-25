from abc import ABC
from typing import Generic, Optional, TypeVar

import numpy as np

# class BasePipeline(ABC):
#     """
#     High level abstraction for this package, when we want to deal with a real problem
#     we need to create a pipeline, a pipeline corresponding to a specific real problem.

#     A pipeline can be nested to another pipeline.
#     """

#     def __init__(self):
#         pass

#     def __call__(self, *args, **kwargs):
#         pass

#     @abstractmethod
#     def predict(self):
#         raise NotImplementedError("Method `predict` has not been implemented.")

#     def create_pipeline(self, config: dict):
#         pass


InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class ParamConverter(ABC, Generic[InputType, OutputType]):
    def __init__(self):
        super().__init__()

    def convert_input(self, input: InputType) -> Optional[np.ndarray]:
        pass

    def convert_output(self, internal_result: np.ndarray) -> Optional[OutputType]:
        pass


class Predictor(ABC):
    """
    A predictor is a step of a pipeline, for example, text detetion
    and text recognization are specific steps for OCR
    Predictor can be build on processors, and then we can build pipeline
    by some certain predictors.
    """

    def __init__(self, model, device, converter: Optional[ParamConverter] = None):
        self.model = model
        self.device = model.device
        self.converter = converter

    def predict(self, input, **kwargs):
        pass

    def __call__(self, input, **kwargs):
        pass


# class BaseProcessor(ABC):
#     """
#     A processor do a specific thing for a predictor, such as route an image
#     """

#     def __init__(self):
#         pass

#     def __call__(self, input, **kwargs):
#         pass

#     @abstractmethod
#     def process(self, input, **kwargs):
#         pass
