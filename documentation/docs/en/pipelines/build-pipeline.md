# Building Custom Pipelines

MyOCR's pipelines orchestrate multiple predictors to perform complex tasks. While the library provides standard pipelines like `CommonOCRPipeline` and `StructuredOutputOCRPipeline`, you might need to create a custom pipeline for specific workflows, such as:

*   Using different combinations of models or predictors.
*   Adding custom pre-processing or post-processing steps.
*   Integrating components beyond standard OCR (e.g., image enhancement, layout analysis before OCR).
*   Handling different input/output types.

This guide explains the steps to build your own pipeline.

## 1. Inherit from `base.Pipeline`

All pipelines should inherit from the abstract base class `myocr.base.Pipeline`.


## 2. Initialize Predictors in `__init__`

The `__init__` method is where you typically load the models and create the predictor instances your pipeline will use.

*   **Load Models:** Use `myocr.modeling.model.ModelLoader` to load the necessary ONNX or custom PyTorch models.
*   **Instantiate Processors:** Create instances of the required `CompositeProcessor` classes (e.g., `TextDetectionProcessor`, `TextRecognitionProcessor`, or custom ones).
*   **Create Predictors:** Combine the loaded models and `CompositeProcessor` using the `Predictor(processor)` method.
*   **Store Predictors:** Store the created predictor instances as attributes of your pipeline class (e.g., `self.det_predictor`).


## 3. Implement the `process` Method

This method defines the core logic of your pipeline. It takes the input data (e.g., an image path, a PIL Image), calls the `predict` method of the initialized predictors in sequence, handles intermediate results, and returns the final output.

After the above steps, the implementation code will be like:
```python
from PIL import Image
from typing import Optional
from myocr.base import Predictor
from myocr.types import OCRResult # Import necessary data structures

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
        

    def process(self, image_path: str) -> OCRResult:
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
        
        return buildOcrResult(detected_objects) # Return the output of the detection predictor
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