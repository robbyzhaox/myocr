# Building Custom Predictors

Predictors in MyOCR act as the bridge between a loaded `Model` (ONNX or PyTorch) and the end-user or pipeline. They encapsulate the necessary pre-processing and post-processing logic required to make a model easily usable for a specific task.

While MyOCR provides standard predictors (via converters like `TextDetectionParamConverter`, `TextRecognitionParamConverter`), you might need a custom predictor if:

*   Your model requires unique input pre-processing (e.g., different normalization, resizing, input format).
*   Your model produces output that needs custom decoding or formatting (e.g., different bounding box formats, specialized classification labels, structured output not handled by existing pipelines).
*   You want to create a predictor for a completely new task beyond detection, recognition, or classification.

The key to building a custom predictor is creating a custom **`ParamConverter`** class.

## 1. Understand the Role of `ParamConverter`

A predictor itself is a simple wrapper (defined in `myocr.base.Predictor`). The actual work happens within its associated `ParamConverter` (a class inheriting from `myocr.base.ParamConverter`). The converter has two main jobs:

1.  **`convert_input(user_input)`:** Takes the data provided by the user or pipeline (e.g., a PIL Image) and transforms it into the precise format expected by the model's inference method (e.g., a normalized, batch-dimensioned NumPy array).
2.  **`convert_output(model_output)`:** Takes the raw output from the model's inference method (e.g., NumPy arrays representing heatmaps or sequence probabilities) and transforms it into a user-friendly, structured format (e.g., a list of bounding boxes with text and scores, like `DetectedObjects` or `RecognizedTexts`).

## 2. Create a Custom `ParamConverter` Class

1.  **Inherit:** Create a Python class that inherits from `myocr.base.ParamConverter`.
2.  **Specify Types (Optional but Recommended):** Use generics to indicate the expected input type for `convert_input` and the output type for `convert_output`. For example, `ParamConverter[PIL.Image.Image, DetectedObjects]` means it takes a PIL Image and returns `DetectedObjects`.
3.  **Implement `__init__`:** Initialize any necessary parameters, such as thresholds, label mappings, or references needed during conversion.
4.  **Implement `convert_input`:** Write the code to transform the input data into the model-ready format.
5.  **Implement `convert_output`:** Write the code to transform the raw model output into the desired structured result.

```python
import logging
from typing import Optional, Tuple, List, Any
import numpy as np
from PIL import Image as PILImage

from myocr.base import ParamConverter
# Import any necessary base structures or create your own
from myocr.predictors.base import BoundingBox 

logger = logging.getLogger(__name__)

# --- Define a custom output structure (Example) ---
class CustomResult:
    def __init__(self, label: str, score: float, details: Any):
        self.label = label
        self.score = score
        self.details = details

    def __repr__(self):
        return f"CustomResult(label='{self.label}', score={self.score:.4f}, details={self.details})"

# --- Create the Custom Converter ---
# Example: Takes a PIL Image, outputs a CustomResult
class MyTaskConverter(ParamConverter[PILImage.Image, CustomResult]):
    def __init__(self, threshold: float = 0.5, target_size: Tuple[int, int] = (224, 224)):
        super().__init__()
        self.threshold = threshold
        self.target_size = target_size
        self.input_image_for_output = None # Store context if needed for output conversion
        logger.info(f"MyTaskConverter initialized with threshold={threshold}, target_size={target_size}")

    def convert_input(self, input_data: PILImage.Image) -> Optional[np.ndarray]:
        """Prepares a PIL Image for a hypothetical classification model."""
        self.input_image_for_output = input_data # Save for later use if needed
        
        try:
            # 1. Resize
            image_resized = input_data.resize(self.target_size, PILImage.Resampling.BILINEAR)
            
            # 2. Convert to NumPy array
            image_np = np.array(image_resized).astype(np.float32)
            
            # 3. Normalize (Example: simple /255)
            image_np /= 255.0
            
            # 4. Add Batch Dimension and Channel Dimension if needed (e.g., HWC -> NCHW)
            if image_np.ndim == 2: # Grayscale
                image_np = np.expand_dims(image_np, axis=-1)
            # Assume model wants NCHW
            image_np = np.expand_dims(image_np.transpose(2, 0, 1), axis=0) 
            
            logger.debug(f"Converted input image to shape: {image_np.shape}")
            return image_np.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error during input conversion: {e}")
            return None

    def convert_output(self, internal_result: Any) -> Optional[CustomResult]:
        """Processes the raw output of a hypothetical classification model."""
        try:
            # Assume model output is a list/tuple containing a NumPy array of scores
            scores = internal_result[0] # Example: [[0.1, 0.8, 0.1]]
            if scores.ndim > 1: # Handle potential batch dimension
                scores = scores[0]
                
            # 1. Find best prediction
            pred_index = np.argmax(scores)
            pred_score = float(scores[pred_index])
            
            logger.debug(f"Raw scores: {scores}, Predicted index: {pred_index}, Score: {pred_score}")

            # 2. Apply threshold
            if pred_score < self.threshold:
                logger.info(f"Prediction score {pred_score} below threshold {self.threshold}")
                return None # Or return a default/negative result
                
            # 3. Map index to label (Assume a predefined mapping)
            labels = ["cat", "dog", "other"] # Example labels
            pred_label = labels[pred_index] if pred_index < len(labels) else "unknown"
            
            # 4. Format into CustomResult
            # Include any extra details, potentially using self.input_image_for_output
            result = CustomResult(label=pred_label, score=pred_score, details={"original_size": self.input_image_for_output.size})
            
            return result

        except Exception as e:
            logger.error(f"Error during output conversion: {e}")
            return None
```

## 3. Create the Predictor Instance

Once you have your custom converter and have loaded your model, you can create the predictor instance.

```python
from myocr.modeling.model import ModelLoader, Device
from PIL import Image
# Assume MyTaskConverter is defined as above

# 1. Load your model (ONNX or Custom PyTorch)
model_path = "path/to/your/custom_model.onnx" # Or path to YAML for CustomModel
model_format = "onnx" # Or "custom"
device = Device('cuda:0')

loader = ModelLoader()
model = loader.load(model_format, model_path, device)

# 2. Instantiate your custom converter
custom_converter = MyTaskConverter(threshold=0.6, target_size=(256, 256)) # Use custom params if needed

# 3. Create the predictor
custom_predictor = model.predictor(custom_converter)

# 4. Use the predictor
input_image = Image.open("path/to/test_image.jpg").convert("RGB")
prediction_result = custom_predictor.predict(input_image) # Returns CustomResult or None

if prediction_result:
    print(f"Prediction: {prediction_result}")
else:
    print("Prediction failed or below threshold.")
```

## 4. Integrate into a Pipeline (Optional)

If your custom predictor is part of a larger workflow, you can integrate it into a [Custom Pipeline](./../pipelines/build-pipeline.md) by initializing it within the pipeline's `__init__` method and calling its `predict` method within the pipeline's `process` method.

By following these steps, you can create specialized predictors tailored to your specific models and tasks within the MyOCR framework. 