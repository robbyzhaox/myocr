from abc import ABC, abstractmethod


class BasePipeline(ABC):
    """
    High level abstraction for this package, when we want to deal with a real problem
    we need to create a pipeline, a pipeline corresponding to a specific real problem.

    A pipeline can be nested to another pipeline.
    """

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        self.predict()

    @abstractmethod
    def predict(self):
        raise NotImplementedError("Method `predict` has not been implemented.")

    def create_pipeline(self, config: dict):
        pass


class BasePredictor(ABC):
    """
    A predictor is a step of a pipeline, for example, text detetion
    and text recognization are specific steps for OCR
    Predictor can be build on processors, and then we can build pipeline
    by some certain predictors.
    """

    def __init__(
        self,
    ):
        super().__init__()

    def predict(self, input, **kwargs):
        pass

    def __call__(self, input, **kwargs):
        return self.predict(input, **kwargs)


class BaseProcessor(ABC):
    """
    A processor do a specific thing for a predictor, such as route an image
    """

    def __init__(self):
        super().__init__()

    def __call__(self, input, **kwargs):
        return self.process(input, **kwargs)

    @abstractmethod
    def process(self, input, **kwargs):
        pass
