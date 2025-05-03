# Building Custom Pipelines

MyOCR's pipelines orchestrate multiple predictors to perform complex tasks. While the library provides standard pipelines like `CommonOCRPipeline` and `StructuredOutputOCRPipeline`, you might need to create a custom pipeline for specific workflows, such as:

*   Using different combinations of models or predictors.
*   Adding custom pre-processing or post-processing steps.
*   Integrating components beyond standard OCR (e.g., image enhancement, layout analysis before OCR).
*   Handling different input/output types.

This guide explains the steps to build your own pipeline.

## 1. Inherit from `base.Pipeline`

All pipelines should inherit from the abstract base class `myocr.base.Pipeline`.

```python
from myocr.base import Pipeline

class MyCustomPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        # Initialization logic here
        pass

    def process(self, input_data):
        # Processing logic here
        pass
```

## 2. Initialize Predictors in `__init__`

The `__init__` method is where you typically load the models and create the predictor instances your pipeline will use.

*   **Load Models:** Use `myocr.modeling.model.ModelLoader` to load the necessary ONNX or custom PyTorch models.
*   **Instantiate Processors:** Create instances of the required `CompositeProcessor` classes (e.g., `TextDetectionProcessor`, `TextRecognitionProcessor`, or custom ones).
*   **Create Predictors:** Combine the loaded models and `CompositeProcessor` using the `Predictor(processor)` method.
*   **Store Predictors:** Store the created predictor instances as attributes of your pipeline class (e.g., `self.det_predictor`).

```python
import logging
from myocr.base import Pipeline,Predictor
from myocr.modeling.model import ModelLoader, Device
from myocr.config import MODEL_PATH # Default model directory path
from myocr.processors import TextDetectionProcessor

logger = logging.getLogger(__name__)

class MyDetectionOnlyPipeline(Pipeline):
    def __init__(self, device: Device, detection_model_name: str = "dbnet++.onnx"):
        super().__init__()
        self.device = device
        # --- Load Detection Model ---
        det_model_path = MODEL_PATH + detection_model_name
        det_model = ModelLoader().load("onnx", det_model_path, self.device)
        
        # --- Create Detection Predictor ---
        det_processor = TextDetectionProcessor(det_model.device)
        self.det_predictor = Predictor(det_processor)
        logger.info(f"DetectionOnlyPipeline initialized with {detection_model_name} on {device.name}")
        
    def process(self, input_data):
        # Implementation in the next step
        pass
```

## 3. Implement the `process` Method

This method defines the core logic of your pipeline. It takes the input data (e.g., an image path, a PIL Image), calls the `predict` method of the initialized predictors in sequence, handles intermediate results, and returns the final output.

```python
from PIL import Image
from typing import Optional
from myocr.base import Predictor
from myocr.types import DetectedObjects # Import necessary data structures

class MyDetectionOnlyPipeline(Pipeline):
    def __init__(self, device: Device, detection_model_name: str = "dbnet++.onnx"):
        # ... (Initialization from previous step) ...
        super().__init__()
        self.device = device
        
        det_model_path = MODEL_PATH + detection_model_name
        det_model = ModelLoader().load("onnx", det_model_path, self.device)
        det_processor = TextDetectionProcessor(det_model.device)
        self.det_predictor = Predictor(det_processor)
        logger.info(f"DetectionOnlyPipeline initialized with {detection_model_name} on {device.name}")
        

    def process(self, image_path: str) -> Optional[DetectedObjects]:
        """Processes an image file and returns detected objects."""
        # 1. Load Image (Example: handling path input)
        image = Image.open(image_path).convert("RGB")
        if image is None:
            return None
            
        # 2. Run Detection Predictor
        detected_objects = self.det_predictor.predict(image)

        # 3. Return Results
        if detected_objects:
            logger.info(f"Detection successful: Found {len(detected_objects.bounding_boxes)} boxes.")
        else:
            logger.info("Detection successful: No text boxes found.")
            
        return detected_objects # Return the output of the detection predictor
```

**Example: Combining Predictors (Conceptual)**

If you need multiple steps, you chain the predictor calls, passing the output of one step as the input to the next (if compatible).

```python
class MyFullOCRPipeline(Pipeline):
    def __init__(self, device: Device):
        super().__init__()
        self.device = device
        # --- Load det, cls, rec models --- (Assume paths are correct)
        det_model = ModelLoader().load("onnx", MODEL_PATH + "dbnet++.onnx", device)
        cls_model = ModelLoader().load("onnx", MODEL_PATH + "cls.onnx", device)
        rec_model = ModelLoader().load("onnx", MODEL_PATH + "rec.onnx", device)
        
        # --- Create predictors ---
        self.det_predictor = Predictor(TextDetectionProcessor(device))
        self.cls_predictor = Predictor(TextDirectionProcessor())
        self.rec_predictor = Predictor(TextRecognitionProcessor())
        logger.info(f"MyFullOCRPipeline initialized on {device.name}")

    def process(self, image_path: str):
        logger.debug(f"Processing {image_path}")
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None

        # Step 1: Detection
        detected = self.det_predictor.predict(image)
        if not detected or not detected.bounding_boxes:
            logger.info("No text detected.")
            return None
        logger.debug(f"Detected {len(detected.bounding_boxes)} regions.")

        # Step 2: Classification
        classified = self.cls_predictor.predict(detected)
        if not classified:
            logger.warning("Classification step failed, proceeding without angle correction.")
            classified = detected # Use original detections if classification fails
        logger.debug("Classification complete.")
            
        # Step 3: Recognition
        recognized_texts = self.rec_predictor.predict(classified)
        if not recognized_texts:
            logger.warning("Recognition step failed.")
            return None
        logger.info("Recognition complete.")
        
        # Add original image size info if needed by consumers
        recognized_texts.original(image.size[0], image.size[1])
        return recognized_texts # Final result
```

## 4. Using Your Custom Pipeline

Once defined, you can import and use your custom pipeline just like the built-in ones.

```python
# from your_module import MyDetectionOnlyPipeline # Or MyFullOCRPipeline
from myocr.modeling.model import Device

pipeline = MyDetectionOnlyPipeline(device=Device('cuda:0'))
results = pipeline.process('path/to/image.jpg')

if results:
    # Process the results from your custom pipeline
    print(f"Found {len(results.bounding_boxes)} text regions.")
```

Remember to handle potential errors during model loading or prediction steps within your pipeline logic.