# Predictors

Predictors are responsible for handling the inference logic for specific models (detection, recognition, classification) within MyOCR. They bridge the gap between raw model outputs and usable results by incorporating pre-processing and post-processing steps.

Predictors are typically associated with a `Model` object and a `CompositeProcessor`.

*   **Model:** Provides the core `forward_internal` method (e.g., ONNX session run, PyTorch model forward pass).
*   **CompositeProcessor:** Handles the conversion of input data into the format expected by the model, and the conversion of the model's raw output into a structured, meaningful format.

## Base Components

*   **`myocr.base.Predictor`:** A simple wrapper that calls the `CompositeProcessor`'s input conversion, the `Model`'s forward pass, and the `CompositeProcessor`'s output conversion.
*   **`myocr.base.CompositeProcessor`:** An abstract base class defining `preprocess` and `postprocess` methods.
*   **`myocr.predictors.base`:** Defines common data structures like `BoundingBox`, `RectBoundingBox`, `DetectedObjects`, `TextItem`, and `RecognizedTexts` used as inputs and outputs by different processors.

## Available Predictors and Processors

Predictors are implicitly created when calling the `Predictor(processor)` method on a loaded `Model` instance. The key components are the `CompositeProcessor` implementations:

### 1. Text Detection (`TextDetectionProcessor`)

*   **File:** `myocr/processors/text_detection_processor.py`
*   **Input:** `PIL.Image`
*   **Output:** `DetectedObjects` (containing original image and list of `RectBoundingBox`)
*   **Associated Model:** Typically a DBNet/DBNet++ ONNX model.

### 2. Text Direction Classification (`TextDirectionProcessor`)

*   **File:** `myocr/processors/text_direction_processor.py`
*   **Input:** `DetectedObjects`
*   **Output:** `DetectedObjects` (with `angle` attribute updated in each `RectBoundingBox`)
*   **Associated Model:** Typically a simple CNN classifier ONNX model.

### 3. Text Recognition (`TextRecognitionProcessor`)

*   **File:** `myocr/processors/text_recognition_processor.py`
*   **Input:** `DetectedObjects` (output from Text Direction)
*   **Output:** `RecognizedTexts` (containing list of `TextItem`)
*   **Associated Model:** Typically a CRNN-based ONNX model.

## Usage Example (Conceptual)

```python
import cv2
from myocr.modeling.model import ModelLoader, Device
from myocr.processors import TextDetectionProcessor, TextDirectionProcessor, TextRecognitionProcessor

# Assume models are loaded
det_model = ModelLoader().load('onnx', 'path/to/det_model.onnx', Device('cuda:0'))
cls_model = ModelLoader().load('onnx', 'path/to/cls_model.onnx', Device('cuda:0'))
rec_model = ModelLoader().load('onnx', 'path/to/rec_model.onnx', Device('cuda:0'))

# Create predictors by associating models with processor
dec_predictor = Predictor(det_model, TextDetectionProcessor(det_model.device))
cls_predictor = Predictor(cls_model, TextDirectionProcessor())
rec_predictor = Predictor(rec_model, TextRecognitionProcessor())

# Load image
img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)

# Run prediction steps
detected_objects = det_predictor.predict(img)
if detected_objects:
  classified_objects = cls_predictor.predict(detected_objects) # Predict calls the processor steps + model forward
  recognized_texts = rec_predictor.predict(classified_objects)

  print(recognized_texts.get_content_text())
```


## Performance Tips

### Batch Processing

```python
# Process multiple regions
results = [predictor.predict(region) for region in regions]
```

### Memory Optimization

```python
# Clear GPU memory
import torch
torch.cuda.empty_cache()
```

## Error Handling

Predictors handle various error cases:

- Invalid input format
- Model loading errors
- GPU memory issues
- Inference errors

See the [Troubleshooting Guide](../faq.md) for common issues and solutions. 