# Creating Custom Predictors

Predictors in MyOCR act as the bridge between a loaded `Model` (ONNX or PyTorch) and the end-user or pipeline. They encapsulate the necessary pre-processing and post-processing logic required to make a model easily usable for a specific task.

While MyOCR provides standard predictors (via processors like `TextDetectionProcessor`, `TextRecognitionProcessor`), you might need a custom predictor if:

*   Your model requires unique input pre-processing (e.g., different normalization, resizing, input format).
*   Your model produces output that needs custom decoding or formatting (e.g., different bounding box formats, specialized classification labels, structured output not handled by existing pipelines).
*   You want to create a predictor for a completely new task beyond detection, recognition, or classification.

The key to building a custom predictor is creating a custom **`CompositeProcessor`** class.

## 1. Understand the Role of `CompositeProcessor`

A predictor itself is a simple wrapper (defined in `myocr.base.Predictor`). The actual work happens within its associated `CompositeProcessor` (a class inheriting from `myocr.base.CompositeProcessor`). The Processor has two main jobs:

1.  **`preprocess(user_input)`:** Takes the data provided by the user or pipeline (e.g., a PIL Image) and transforms it into the precise format expected by the model's inference method (e.g., a normalized, batch-dimensioned NumPy array).
2.  **`postprocess(model_output)`:** Takes the raw output from the model's inference method (e.g., NumPy arrays representing heatmaps or sequence probabilities) and transforms it into a user-friendly, structured format (e.g., a list of bounding boxes with text and scores, like `TextRegion`).

## 2. Create a Custom `CompositeProcessor` Class

1.  **Inherit:** Create a Python class that inherits from `myocr.base.CompositeProcessor`.
2.  **Specify Types (Optional but Recommended):** Use generics to indicate the expected input type for `preprocess` and the output type for `postprocess`. For example, `CompositeProcessor[PIL.Image.Image, List[RectBoundingBox]]` means it takes a PIL Image and returns `List[RectBoundingBox]`.
3.  **Implement `__init__`:** Initialize any necessary parameters, such as thresholds, label mappings, or references needed during conversion.
4.  **Implement `preprocess`:** Write the code to transform the input data into the model-ready format.
5.  **Implement `postprocess`:** Write the code to transform the raw model output into the desired structured result.

**Note**: take the Available predictors for example

## 3. Create the Predictor Instance

Once you have your custom `CompositeProcessor` and have loaded your model, you can create the predictor instance.

## 4. Integrate into a Pipeline (Optional)

If your custom predictor is part of a larger workflow, you can integrate it into a [Custom Pipeline](./../pipelines/build-pipeline.md) by initializing it within the pipeline's `__init__` method and calling its `predict` method within the pipeline's `process` method.

By following these steps, you can create specialized predictors tailored to your specific models and tasks within the MyOCR framework. 