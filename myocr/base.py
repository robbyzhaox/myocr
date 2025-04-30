from abc import ABC
from typing import Any, Generic, Optional, TypeVar

import numpy as np

""" Generic type definition """
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class Processor(ABC, Generic[InputType, OutputType]):
    def __init__(self):
        super().__init__()

    def pre_process(self, input_data: InputType) -> Any:
        pass

    def post_process(self, output_data: Any) -> Optional[OutputType]:
        pass


class ParamConverter(ABC, Generic[InputType, OutputType]):
    """
    The implementation of this converter is responsible for converting the
    input and output for a specific model.
    """

    def __init__(self):
        super().__init__()

    def convert_input(self, input_data: InputType) -> Optional[np.ndarray]:
        pass

    def convert_output(self, internal_result: np.ndarray) -> Optional[OutputType]:
        pass


class Predictor:
    """
    A predictor is a combination of a model and its input & output
    parameter converter.

    It will first convert the input for the model, then do inference
    by the model, finally convert the model output.
    """

    def __init__(self, model, converter: Optional[ParamConverter] = None):
        self.model = model
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
    High level abstraction for doing a series of work, subclass

    A pipeline can be nested to another pipeline(TBD).
    """

    def __init__(self):
        pass

    def process(self, *args, **kwargs):
        raise NotImplementedError("Method `process` has not been implemented.")

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)
