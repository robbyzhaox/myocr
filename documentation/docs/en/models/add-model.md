# Adding New Models

MyOCR's modular design allows you to integrate new or custom models into the system. The process depends on the type of model you are adding.


## Option 1: Adding a Custom PyTorch Model (Architecture & Weights)

If you have a custom model defined in PyTorch (using components potentially from `myocr.modeling` or external libraries), you can integrate it using MyOCR's custom model loading. This will be the powerful way to define your own model.

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

4.  **Create Predictor (with appropriate Processor):**
    *   You will likely need a `CompositeProcessor` that matches your custom model's input pre-processing and output post-processing needs. You might be able to reuse an existing one (e.g., `TextDetectionProcessor` if your output is similar) or you may need to create a custom processor class inheriting from `myocr.base.CompositeProcessor`.

    ```python
    # Option A: Reuse existing processor (if compatible)
    from myocr.processors import TextDetectionProcessor
    predictor = custom_model.predictor(TextDetectionProcessor(custom_model.device))

    # Option B: Create and use a custom processor
    # from my_custom_processors import MyCustomProcessor 
    # predictor = custom_model.predictor(MyCustomProcessor(...))
    ```

5.  **Integrate into a Pipeline (Optional):**
    *   You can use your custom predictor directly or integrate it into a custom pipeline class that inherits from `myocr.base.Pipeline`.


## Option 2: Adding a Pre-trained ONNX Model

This is the simplest way, you only need to put your model file to the model directory and load the model by `ModelLoader`.

1.  **Place the Model File:**
    *   Copy your pre-trained `.onnx` model file into the default model directory (`~/.MyOCR/models/`) or another location accessible by your application.

2.  **Load the Model:**

    ```python
    from myocr.modeling.model import ModelLoader, Device

    # Load an ONNX model for CPU inference
    loader = ModelLoader()
    onnx_model = loader.load(model_format='onnx', model_name_path='path/to/your/model.onnx', device=Device('cpu'))
    ```
Other steps are similar to Option 1


## Option 3: Load Existing PyTorch Models

It's very easy to load a pre-trained PyTorch models with its weights like following:

```python
from myocr.modeling.model import ModelZoo
model = ModelZoo.load_model("pt", "resnet152", "cuda:0" if torch.cuda.is_available() else "cpu")
```
Other steps are similar to Option 1

