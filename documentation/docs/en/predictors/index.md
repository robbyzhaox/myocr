# Predictors

Predictors are responsible for handling the inference logic for specific models within MyOCR. They bridge the gap between raw model outputs and usable results by incorporating pre-processing and post-processing steps.

Predictors are typically associated with a `Model` object and a `CompositeProcessor`.

*   **Model:** Provides the core `forward_internal` method (e.g., ONNX session run, PyTorch model forward pass).
*   **CompositeProcessor:** Handles the conversion of input data into the format expected by the model, and the conversion of the model's raw output into a structured, meaningful format.

## Base Components

*   **`myocr.base.Predictor`:** A simple wrapper that calls the `CompositeProcessor`'s input conversion, the `Model`'s forward pass, and the `CompositeProcessor`'s output conversion.
*   **`myocr.base.CompositeProcessor`:** An abstract base class defining `preprocess` and `postprocess` methods.


## Available Predictors and Processors

Predictors are created by calling the `Predictor(model, processor)`. The key components are the `CompositeProcessor` implementations:

###  Text Detection (`TextDetectionProcessor`)

*   **File:** `myocr/processors/text_detection_processor.py`
*   **Associated Model:** Typically a DBNet/DBNet++ ONNX model.

###  Text Direction Classification (`TextDirectionProcessor`)

*   **File:** `myocr/processors/text_direction_processor.py`
*   **Associated Model:** Typically a simple CNN classifier ONNX model.

###  Text Recognition (`TextRecognitionProcessor`)

*   **File:** `myocr/processors/text_recognition_processor.py`
*   **Associated Model:** Typically a CRNN-based ONNX model.



## Performance Tips

### Batch Processing

```python
# Process multiple regions
results = [predictor.predict(region) for region in regions]
```
