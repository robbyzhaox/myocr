# Predictors

Predictors are responsible for handling the inference logic for specific models (detection, recognition, classification) within MyOCR. They bridge the gap between raw model outputs and usable results by incorporating pre-processing and post-processing steps.

Predictors are typically associated with a `Model` object and a `ParamConverter`.

*   **Model:** Provides the core `forward_internal` method (e.g., ONNX session run, PyTorch model forward pass).
*   **ParamConverter:** Handles the conversion of input data into the format expected by the model, and the conversion of the model's raw output into a structured, meaningful format.

## Base Components

*   **`myocr.base.Predictor`:** A simple wrapper that calls the `ParamConverter`'s input conversion, the `Model`'s forward pass, and the `ParamConverter`'s output conversion.
*   **`myocr.base.ParamConverter`:** An abstract base class defining `convert_input` and `convert_output` methods.
*   **`myocr.predictors.base`:** Defines common data structures like `BoundingBox`, `RectBoundingBox`, `DetectedObjects`, `TextItem`, and `RecognizedTexts` used as inputs and outputs by different converters.

## Available Predictors and Converters

Predictors are implicitly created when calling the `.predictor(converter)` method on a loaded `Model` instance. The key components are the `ParamConverter` implementations:

### 1. Text Detection (`TextDetectionParamConverter`)

*   **File:** `myocr/predictors/text_detection_predictor.py`
*   **Input:** `PIL.Image`
*   **Output:** `DetectedObjects` (containing original image and list of `RectBoundingBox`)
*   **Associated Model:** Typically a DBNet/DBNet++ ONNX model.
*   **Preprocessing (`convert_input`):**
    *   Resizes image dimensions to be divisible by 32.
    *   Normalizes pixel values (subtract mean, divide by std).
    *   Transposes channels to CHW format.
    *   Adds batch dimension.
*   **Postprocessing (`convert_output`):**
    *   Takes the raw probability map from the model.
    *   Applies a threshold (0.3) to create a binary map.
    *   Finds contours in the binary map.
    *   Filters contours based on length.
    *   Calculates minimum area rotated rectangles (`cv2.minAreaRect`).
    *   Filters rectangles based on minimum side length.
    *   Calculates a confidence score based on the mean probability within the contour.
    *   Filters based on confidence score (>= 0.3).
    *   Expands the bounding box polygon (`_unclip` function with ratio 2.3).
    *   Calculates the minimum area rectangle of the expanded box.
    *   Filters again by minimum side length.
    *   Scales the final box coordinates back to the original image dimensions.
    *   Creates `RectBoundingBox` objects.
    *   Sorts boxes top-to-bottom, then left-to-right.
    *   Wraps results in a `DetectedObjects` container.

### 2. Text Direction Classification (`TextDirectionParamConverter`)

*   **File:** `myocr/predictors/text_direction_predictor.py`
*   **Input:** `DetectedObjects`
*   **Output:** `DetectedObjects` (with `angle` attribute updated in each `RectBoundingBox`)
*   **Associated Model:** Typically a simple CNN classifier ONNX model.
*   **Preprocessing (`convert_input`):**
    *   Iterates through bounding boxes in the input `DetectedObjects`.
    *   Crops each text region from the original image using `myocr.util.crop_rectangle` (target height 48).
    *   Stores the cropped image within the `RectBoundingBox` object (`set_croped_img`).
    *   Normalizes the cropped image pixels (`/ 255.0 - 0.5) / 0.5`).
    *   Ensures 3 channels (expands dims if grayscale).
    *   Pads batches horizontally to the maximum width found in the batch.
    *   Stacks images into a batch tensor (BCHW).
*   **Postprocessing (`convert_output`):**
    *   Takes the raw classification logits/probabilities from the model.
    *   Finds the index of the maximum probability for each box (0 or 1).
    *   Maps the index to an angle (0 or 180 degrees).
    *   Calculates confidence score (probability of the predicted class).
    *   Updates the `.angle` attribute of the corresponding `RectBoundingBox` in the input `DetectedObjects`.
    *   Returns the modified `DetectedObjects`.

### 3. Text Recognition (`TextRecognitionParamConverter`)

*   **File:** `myocr/predictors/text_recognition_predictor.py`
*   **Input:** `DetectedObjects` (output from Text Direction)
*   **Output:** `RecognizedTexts` (containing list of `TextItem`)
*   **Associated Model:** Typically a CRNN-based ONNX model.
*   **Preprocessing (`convert_input`):**
    *   Retrieves the pre-cropped image stored in each `RectBoundingBox` (`get_croped_img`).
    *   Rotates the image if the `angle` attribute indicates 180 degrees.
    *   Normalizes pixel values (`/ 255.0 - 0.5) / 0.5`).
    *   Ensures 3 channels.
    *   Pads batches horizontally to the maximum width.
    *   Stacks images into a batch tensor (BCHW).
*   **Postprocessing (`convert_output`):**
    *   Takes the raw sequence output from the model (Time Steps, Batch Size, Num Classes).
    *   Transposes to (Batch Size, Time Steps, Num Classes).
    *   Iterates through each item in the batch:
        *   Applies Softmax to get probabilities.
        *   Calculates character confidences (max probability per time step).
        *   Calculates overall text confidence (mean of character confidences).
        *   Performs CTC decoding: Gets character indices (argmax per time step) and uses `myocr.util.LabelTranslator` (initialized with a large Chinese+English alphabet) to decode the sequence, removing blanks and duplicates.
        *   Creates a `TextItem` with the decoded text, confidence, and the original `RectBoundingBox`.
    *   Collects all `TextItem`s into a `RecognizedTexts` object.

## Usage Example (Conceptual)

```python
from myocr.modeling.model import ModelLoader, Device
from myocr.predictors import TextDetectionParamConverter, TextRecognitionParamConverter, TextDirectionParamConverter
from PIL import Image

# Assume models are loaded
det_model = ModelLoader().load('onnx', 'path/to/det_model.onnx', Device('cuda:0'))
cls_model = ModelLoader().load('onnx', 'path/to/cls_model.onnx', Device('cuda:0'))
rec_model = ModelLoader().load('onnx', 'path/to/rec_model.onnx', Device('cuda:0'))

# Create predictors by associating models with converters
det_predictor = det_model.predictor(TextDetectionParamConverter(det_model.device))
cls_predictor = cls_model.predictor(TextDirectionParamConverter())
rec_predictor = rec_model.predictor(TextRecognitionParamConverter())

# Load image
img = Image.open('path/to/image.png').convert("RGB")

# Run prediction steps
detected_objects = det_predictor.predict(img)
if detected_objects:
  classified_objects = cls_predictor.predict(detected_objects) # Predict calls the converter steps + model forward
  recognized_texts = rec_predictor.predict(classified_objects)

  print(recognized_texts.get_content_text())
```

## Predictor Configuration

Predictors can be configured through their parameter converters:

```python
class CustomParamConverter(ParamConverter):
    def convert_input(self, input_data):
        # Custom input preprocessing
        pass
    
    def convert_output(self, internal_result):
        # Custom output postprocessing
        pass
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