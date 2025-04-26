# 添加新模型

MyOCR 的模块化设计允许您将新的或自定义的模型集成到系统中。该过程取决于您要添加的模型类型。

## 方式一：添加预训练的 ONNX 模型

这是最简单的方法，特别是如果您的模型适用于标准任务之一（检测、分类、识别），并且其输入/输出格式与现有的 `ParamConverter` 类兼容。

1.  **放置模型文件：**
    *   将您预训练的 `.onnx` 模型文件复制到默认模型目录 (`~/.MyOCR/models/`) 或应用程序可访问的其他位置。

2.  **更新流水线配置：**
    *   确定将使用您的模型的流水线（例如 `CommonOCRPipeline`）。
    *   编辑其对应的 YAML 配置文件（例如 `myocr/pipelines/config/common_ocr_pipeline.yaml`）。
    *   修改 `model:` 部分，使其指向您的新模型的文件名。如果模型位于默认目录中，则只需要文件名。如果位于其他位置，您可能需要调整 `myocr.config.MODEL_PATH` 或使用绝对路径（不太推荐）。

    ```yaml
    # myocr/pipelines/config/common_ocr_pipeline.yaml 中的示例
    model:
      detection: "your_new_detection_model.onnx" # 用您的模型替换默认模型
      cls_direction: "cls.onnx" # 保留默认值或替换
      recognition: "your_new_recognition_model.onnx" # 用您的模型替换默认模型
    ```

3.  **验证兼容性：**
    *   确保您的 ONNX 模型的输入和输出形状/类型与流水线在该步骤使用的 `ParamConverter`（例如，用于检测的 `TextDetectionParamConverter`）兼容。如果不兼容，您可能需要创建自定义转换器（请参阅方式三）。

## 方式二：添加自定义 PyTorch 模型（架构与权重）

如果您有在 PyTorch 中定义的自定义模型（可能使用来自 `myocr.modeling` 或外部库的组件），您可以使用 MyOCR 的自定义模型加载功能将其集成。

1.  **定义模型架构（如果是新的）：**
    *   如果您的架构尚未定义，您可能需要按照 `myocr/modeling/` 内的结构来实现其组件（例如，新的主干网络、头部）。

2.  **创建 YAML 配置：**
    *   创建一个 `.yaml` 文件，定义您的架构组件如何连接。该文件指定主干网络、颈部（可选）和头部的类，以及它们的参数。
    *   （可选）包含一个 `pretrained:` 键，指向包含整个模型训练权重的 `.pth` 文件。

    ```yaml
    # 示例： config/my_custom_detector.yaml
    Architecture:
      model_type: det
      backbone:
        name: YourCustomBackbone # myocr.modeling.backbones 下的类名
        param1: value1
      neck:
        name: YourCustomNeck
        param2: value2
      head:
        name: YourCustomHead
        param3: value3

    pretrained: /path/to/your/custom_model_weights.pth # 可选：完整的模型权重
    ```

3.  **加载自定义模型：**
    *   使用 `ModelLoader` 或 `CustomModel` 类通过其 YAML 配置加载您的模型。

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

4.  **创建预测器（使用合适的转换器）：**
    *   您可能需要一个与自定义模型的输入预处理和输出后处理需求相匹配的 `ParamConverter`。您可以重用现有的转换器（例如，如果您的输出类似，则使用 `TextDetectionParamConverter`），或者您可能需要创建继承自 `myocr.base.ParamConverter` 的自定义转换器类。

    ```python
    # 方式 A：重用现有转换器（如果兼容）
    from myocr.predictors import TextDetectionParamConverter
    predictor = custom_model.predictor(TextDetectionParamConverter(custom_model.device))

    # 方式 B：创建并使用自定义转换器
    # from my_custom_converters import MyCustomParamConverter 
    # predictor = custom_model.predictor(MyCustomParamConverter(...))
    ```

5.  **集成到流水线（可选）：**
    *   您可以直接使用您的自定义预测器，或将其集成到继承自 `myocr.base.Pipeline` 的自定义流水线类中。

## 方式三：创建自定义 `ParamConverter`

如果您的模型（ONNX 或 PyTorch）具有独特的输入要求或以现有转换器无法处理的格式生成输出，则需要创建自定义 `ParamConverter`。

1.  **继承自 `ParamConverter`：**
    *   创建一个继承自 `myocr.base.ParamConverter` 的新 Python 类。

2.  **实现 `convert_input`：**
    *   此方法接收用户提供的输入（例如 PIL 图像、`DetectedObjects`），并将其转换为模型 `forward` 或 `run` 方法所期望的精确格式（例如，具有特定形状、数据类型、归一化的 `numpy` 数组）。

3.  **实现 `convert_output`：**
    *   此方法接收模型的原始输出（例如 `numpy` 数组、张量），并将其转换为结构化的、用户友好的格式（例如 `DetectedObjects`、`RecognizedTexts` 或自定义 Pydantic 模型）。

4.  **与预测器一起使用：**
    *   从模型创建预测器时，传递自定义转换器的实例。

```python
from myocr.base import ParamConverter
from myocr.predictors.base import DetectedObjects # 或其他输入/输出类型
import numpy as np
from typing import Optional

class MyCustomConverter(ParamConverter[np.ndarray, DetectedObjects]): # 示例：输入 numpy，输出 DetectedObjects
    def __init__(self, model_device):
        super().__init__()
        self.device = model_device
        # 添加任何其他需要的参数（阈值、标签等）

    def convert_input(self, input_data: np.ndarray) -> Optional[np.ndarray]:
        # --- 添加您的自定义预处理 --- 
        # 示例：归一化、调整大小、转置、添加批次维度
        processed_input = ... 
        return processed_input

    def convert_output(self, internal_result: np.ndarray) -> Optional[DetectedObjects]:
        # --- 添加您的自定义后处理 --- 
        # 示例：解码边界框、应用 NMS、格式化结果
        formatted_output = ... 
        return formatted_output

# 用法:
# loaded_model = ... # 加载您的模型 (ONNX 或 自定义 PyTorch)
# custom_predictor = loaded_model.predictor(MyCustomConverter(loaded_model.device))
```

有关具体示例，请参阅 `myocr/predictors/` 中现有转换器的实现（例如 `text_detection_predictor.py`）。 