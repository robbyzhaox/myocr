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

### Common Models Used

While the system is flexible, common models used in MyOCR include:

*   **Text Detection:** Often based on **DBNet** or **DBNet++**, utilizing architectures like ResNet combined with FPN necks and specialized detection heads. These models output segmentation maps indicating text presence and location.
*   **Text Recognition:** Frequently employs **CRNN (Convolutional Recurrent Neural Network)** architectures. These typically use CNN backbones (like ResNet or VGG) followed by BiLSTM layers and a CTC (Connectionist Temporal Classification) head for sequence decoding.
*   **Text Direction Classification:** Usually simpler CNN-based classifiers (e.g., adapted MobileNet or ResNet variants) trained to predict text orientation (e.g., 0 or 180 degrees).

## Model Configuration

Custom models are configured using YAML files. These files specify:

*   The overall `Architecture` (linking backbones, necks, heads).
*   Specific parameters for each component (e.g., backbone type, feature dimensions).
*   Optionally, a path to `pretrained` weights.

```yaml
# Example config.yaml (Simplified)
Architecture:
  model_type: det # or rec, cls
  backbone:
    name: MobileNetV3
    scale: 0.5
    pretrained: true
  neck:
    name: DBFPN
    out_channels: 256
  head:
    name: DBHead
    k: 50

pretrained: path/to/pretrained_weights.pth # Optional
```

## Model Conversion

The `CustomModel` class provides a `to_onnx` method to export PyTorch models defined by YAML configurations into the ONNX format for optimized inference.

```python
# Example (Conceptual)
# Assuming 'custom_model' is an instance of CustomModel
# and 'dummy_input' is a sample input tensor matching the model's expected input shape/type
custom_model.to_onnx('exported_model.onnx', dummy_input)
```

This allows leveraging the performance benefits of ONNX Runtime even for models developed using the custom PyTorch framework within MyOCR.

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
    # Architecture overview
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
    # Architecture overview
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

### Accuracy

- Text Detection:
  - ICDAR2015: 85.2% F1-score
  - Total-Text: 82.1% F1-score
  
- Text Recognition:
  - Chinese: 92.3% accuracy
  - English: 94.7% accuracy

### Speed

On NVIDIA T4 GPU:
- Detection: ~50ms per image
- Recognition: ~20ms per text region
- Classification: ~10ms per region

## Model Training

See the [Training Guide](../training/) for instructions on training custom models. 