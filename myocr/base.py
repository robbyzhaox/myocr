from abc import ABC, abstractmethod
from typing import Dict, Generic, Optional, TypeVar

from myocr.modeling.model import Model

from .types import NDArray

""" Generic type definition """
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class Processor(ABC):
    pass


class PreProcessor(Processor, Generic[InputType], ABC):
    @abstractmethod
    def preprocess(self, input_data: InputType) -> Optional[NDArray]:
        pass


class PostProcessor(Processor, Generic[OutputType], ABC):
    @abstractmethod
    def postprocess(self, internal_result: NDArray) -> Optional[OutputType]:
        pass


class CompositeProcessor(
    PreProcessor[InputType], PostProcessor[OutputType], Generic[InputType, OutputType], ABC
):
    """
    The implementation of this processor is responsible for processing the
    input and output for a specific model for referencing
    """

    def __init__(self, context: Dict = {}):
        super().__init__()
        self.context = context

    @abstractmethod
    def preprocess(self, input_data: InputType) -> Optional[NDArray]:
        pass

    @abstractmethod
    def postprocess(self, internal_result: NDArray) -> Optional[OutputType]:
        pass


class Predictor:
    """
    A predictor is a combination of a model and its input & output
    parameter processors.

    It will first convert the input for the model, then do inference
    by the model, finally convert the model output.
    """

    def __init__(self, model: Model, processor: Optional[CompositeProcessor] = None):
        self.model = model
        self.processor = processor

    def predict(self, input_data, **kwargs):
        if self.processor:
            arr = self.processor.preprocess(input_data)
            res = self.model(arr)
            return self.processor.postprocess(res)
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
