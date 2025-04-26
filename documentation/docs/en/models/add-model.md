# Adding New Models

MyOCR's modular design allows you to integrate new or custom models into the system. The process depends on the type of model you are adding.

## Option 1: Adding a Pre-trained ONNX Model

This is the simplest way, especially if your model fits one of the standard tasks (detection, classification, recognition) and has compatible input/output formats with existing `ParamConverter` classes.

1.  **Place the Model File:**
    *   Copy your pre-trained `.onnx` model file into the default model directory (`~/.MyOCR/models/`) or another location accessible by your application.

2.  **Update Pipeline Configuration:**
    *   Identify the pipeline that will use your model (e.g., `CommonOCRPipeline`).
    *   Edit its corresponding YAML configuration file (e.g., `myocr/pipelines/config/common_ocr_pipeline.yaml`).
    *   Modify the `model:` section to point to your new model's filename. If the model is in the default directory, just the filename is needed. If it's elsewhere, you might need to adjust `myocr.config.MODEL_PATH` or use absolute paths (less recommended).

    ```yaml
    # Example in myocr/pipelines/config/common_ocr_pipeline.yaml
    model:
      detection: "your_new_detection_model.onnx" # Replace default with yours
      cls_direction: "cls.onnx" # Keep default or replace
      recognition: "your_new_recognition_model.onnx" # Replace default with yours
    ```

3.  **Verify Compatibility:**
    *   Ensure your ONNX model's input and output shapes/types are compatible with the `ParamConverter` used by the pipeline for that step (e.g., `TextDetectionParamConverter` for detection). If not, you might need to create a custom converter (see Option 3).

## Option 2: Adding a Custom PyTorch Model (Architecture & Weights)

If you have a custom model defined in PyTorch (using components potentially from `myocr.modeling` or external libraries), you can integrate it using MyOCR's custom model loading.

1.  **Define Model Architecture (if new):**
    *   If your architecture isn't already defined, you might need to implement its components (e.g., new backbones, heads) following the structure within `myocr/modeling/`.

2.  **Create YAML Configuration:**
    *   Create a `.yaml` file that defines how your architecture components are connected. This file specifies the classes for the backbone, neck (optional), and head, along with their parameters.
    *   Optionally, include a `pretrained:` key pointing to a `.pth` file containing the trained weights for the entire model.

    ```yaml
    # Example: config/my_custom_detector.yaml
    Architecture:
      model_type: det
      backbone:
        name: YourCustomBackbone # Class name under myocr.modeling.backbones
        param1: value1
      neck:
        name: YourCustomNeck
        param2: value2
      head:
        name: YourCustomHead
        param3: value3

    pretrained: /path/to/your/custom_model_weights.pth # Optional: Full model weights
    ```

3.  **Load the Custom Model:**
    *   Use the `ModelLoader` or `CustomModel` class to load your model using its YAML configuration.

    ```python
    from myocr.modeling.model import ModelLoader, Device

    loader = ModelLoader()
    device = Device('cuda:0')
    custom_model = loader.load(
        model_format='custom',
        model_name_path='config/my_custom_detector.yaml',
        device=device
    )
    ```

4.  **Create Predictor (with appropriate Converter):**
    *   You will likely need a `ParamConverter` that matches your custom model's input pre-processing and output post-processing needs. You might be able to reuse an existing one (e.g., `TextDetectionParamConverter` if your output is similar) or you may need to create a custom converter class inheriting from `myocr.base.ParamConverter`.

    ```python
    # Option A: Reuse existing converter (if compatible)
    from myocr.predictors import TextDetectionParamConverter
    predictor = custom_model.predictor(TextDetectionParamConverter(custom_model.device))

    # Option B: Create and use a custom converter
    # from my_custom_converters import MyCustomParamConverter 
    # predictor = custom_model.predictor(MyCustomParamConverter(...))
    ```

5.  **Integrate into a Pipeline (Optional):**
    *   You can use your custom predictor directly or integrate it into a custom pipeline class that inherits from `myocr.base.Pipeline`.

## Option 3: Load Existing PyTorch Models

It's very easy to load a pre-trained PyTorch models with its weights like following:

```python
from myocr.modeling.model import ModelZoo
model = ModelZoo.load_model("pt", "resnet152", "cuda:0" if torch.cuda.is_available() else "cpu")
```

