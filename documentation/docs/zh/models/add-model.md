# 添加新模型

MyOCR 的模块化设计允许您将新的或自定义的模型集成到系统中。该过程取决于您要添加的模型类型。


## 方式一：添加自定义 PyTorch 模型（架构与权重）

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

4.  **创建预测器（使用合适的处理器）：**
    *   您可能需要一个与自定义模型的输入预处理和输出后处理需求相匹配的 `CompositeProcessor`。您可以重用现有的处理器（例如，如果您的输出类似，则使用 `TextDetectionProcessor`），或者您可能需要创建继承自 `myocr.base.CompositeProcessor` 的自定义处理器类。

    ```python
    # 方式 A：重用现有处理器（如果兼容）
    from myocr.processors import TextDetectionProcessor
    predictor = custom_model.predictor(TextDetectionProcessor(custom_model.device))

    # 方式 B：创建并使用自定义处理器
    # from my_custom_processors import MyCustomProcessor 
    # predictor = custom_model.predictor(MyCustomProcessor(...))
    ```

5.  **集成到流水线（可选）：**
    *   您可以直接使用您的自定义预测器，或将其集成到继承自 `myocr.base.Pipeline` 的自定义流水线类中。

## 方式二：添加预训练的 ONNX 模型

这是最简单的方法，特别是如果您的模型适用于标准任务之一（检测、分类、识别），并且其输入/输出格式与现有的 `CompositeProcessor` 类兼容。

1.  **放置模型文件：**
    *   将您预训练的 `.onnx` 模型文件复制到默认模型目录 (`~/.MyOCR/models/`) 或应用程序可访问的其他位置。

2.  **加载模型**
    ```python
    from myocr.modeling.model import ModelLoader, Device

    # Load an ONNX model for CPU inference
    loader = ModelLoader()
    onnx_model = loader.load(model_format='onnx', model_name_path='path/to/your/model.onnx', device=Device('cpu'))
    ```
其它步骤同方式一

## 方式三：加载现有的 PyTorch 模型

加载预训练的 PyTorch 模型及其权重非常简单，如下所示：

```python
from myocr.modeling.model import ModelZoo
model = ModelZoo.load_model("pt", "resnet152", "cuda:0" if torch.cuda.is_available() else "cpu")
```
