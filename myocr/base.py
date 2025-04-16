from abc import ABC
from typing import Generic, Optional, TypeVar, Union

import numpy as np
from torch import Tensor

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class ParamConverter(ABC, Generic[InputType, OutputType]):
    def __init__(self):
        super().__init__()

    def convert_input(self, input_data: InputType) -> Optional[Union[Tensor, np.ndarray]]:
        pass

    def convert_output(self, internal_result: Union[Tensor, np.ndarray]) -> Optional[OutputType]:
        pass


class Predictor:
    """
    A predictor is a step of a pipeline, for example, text detetion
    and text recognization are specific steps for OCR
    Predictor can be build on processors, and then we can build pipeline
    by some certain predictors.
    """

    def __init__(self, model, converter: Optional[ParamConverter] = None):
        self.model = model
        self.device = model.device
        self.converter = converter

    def predict(self, input_data, **kwargs):
        if self.converter:
            arr = self.converter.convert_input(input_data)
            res = self.model(arr)
            return self.converter.convert_output(res)
        else:
            return self.model(input_data)

    def __call__(self, input_data, **kwargs):
        return self.predict(input_data, **kwargs)


class Pipeline(ABC):
    """
    High level abstraction for this package, when we want to deal with a real problem
    we need to create a pipeline, a pipeline corresponding to a specific real problem.

    A pipeline can be nested to another pipeline.
    """

    def __init__(self):
        pass

    def process(self):
        raise NotImplementedError("Method `process` has not been implemented.")

    def create_pipeline(self, config: dict):
        pass
