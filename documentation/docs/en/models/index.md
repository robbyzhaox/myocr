# Models

This section provides details about the deep learning models used within the MyOCR project for tasks like text detection, recognition, and direction classification.

## Model Loading and Management

MyOCR utilizes a flexible model loading system defined in `myocr/modeling/model.py`. It supports loading models in different formats:

*   **ONNX (`OrtModel`):** Loads and runs optimized models using the ONNX Runtime (`onnxruntime`). This is often preferred for inference due to performance benefits.
*   **PyTorch (`PyTorchModel`):** Loads standard PyTorch models, potentially leveraging pre-defined architectures from libraries like `torchvision`.
*   **Custom PyTorch (`CustomModel`):** Loads custom PyTorch models defined via YAML configuration files. These configurations specify the model's architecture, including backbones, necks, and heads, using components defined within `myocr/modeling/`.

A `ModelLoader` class acts as a factory to instantiate the correct model type based on the specified format (`onnx`, `pt`, `custom`).

```python
# Example (Conceptual)
from myocr.modeling.model import ModelLoader, Device

# Load an ONNX model for CPU inference
loader = ModelLoader()
onnx_model = loader.load(model_format='onnx', model_name_path='path/to/your/model.onnx', device=Device('cpu'))

# Load a custom model defined by YAML for GPU inference
custom_model = loader.load(model_format='custom', model_name_path='path/to/your/config.yaml', device=Device('cuda:0'))
```

## Model Architectures

The `myocr/modeling/` directory houses the building blocks for custom models:

*   **`architectures/`**: Defines the overall structure connecting backbones, necks, and heads. (e.g., `DBNet`, `CRNN`).
*   **`backbones/`**: Contains feature extraction networks (e.g., `ResNet`, `MobileNetV3`).
*   **`necks/`**: Includes feature fusion modules (e.g., `FPN` - Feature Pyramid Network).
*   **`heads/`**: Defines the final layers responsible for specific tasks (e.g., detection probability maps, sequence decoding).


## Available Models

### Text Detection (DBNet++)

- **DBNet++**: A state-of-the-art text detection model based on DBNet architecture
  - Input: RGB image
  - Output: Text region polygons
  - Features:
    - High accuracy for arbitrary-shaped text
    - Fast inference speed
    - Robust to various text orientations
  - Architecture:
    ```python
    Backbone: ResNet
    Neck: FPN
    Head: DBHead
    ```

### Text Recognition (CRNN)

- **CRNN**: A hybrid CNN-RNN model for text recognition
  - Input: Cropped text region
  - Output: Recognized text
  - Features:
    - Supports Chinese and English characters
    - Handles variable-length text
    - Robust to different fonts and styles
  - Architecture:
    ```python
    Backbone: CNN
    Neck: BiLSTM
    Head: CTC
    ```

### Text Classification Models

- **Text Direction Classifier**: Determines text orientation
  - Input: Text region
  - Output: Orientation angle
  - Features:
    - 0° and 180° classification
    - Helps improve recognition accuracy

## Model Performance

- coming
