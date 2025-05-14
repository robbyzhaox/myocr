# 构建自定义流水线

MyOCR 的流水线协调多个预测器来执行复杂的任务。虽然库提供了像 `CommonOCRPipeline` 和 `StructuredOutputOCRPipeline` 这样的标准流水线，但您可能需要为特定的工作流创建自定义流水线，例如：

*   使用不同的模型或预测器组合。
*   添加自定义的预处理或后处理步骤。
*   集成标准 OCR 之外的组件（例如，OCR 前的图像增强、布局分析）。
*   处理不同的输入/输出类型。

本指南解释了构建您自己的流水线的步骤。

## 1. 继承自 `base.Pipeline`

所有流水线都应继承自抽象基类 `myocr.base.Pipeline`。


## 2. 在 `__init__` 中初始化预测器

`__init__` 方法通常是您加载模型并创建流水线将使用的预测器实例的地方。

*   **加载模型：** 使用 `myocr.modeling.model.ModelLoader` 加载所需的 ONNX 或自定义 PyTorch 模型。
*   **实例化处理器：** 创建所需的 `CompositeProcessor` 类的实例（例如 `TextDetectionProcessor`、`TextRecognitionProcessor` 或自定义的处理器）。
*   **创建预测器：** 使用 `Predictor(processor)` 方法组合加载的模型和处理器。
*   **存储预测器：** 将创建的预测器实例存储为流水线类的属性（例如 `self.det_predictor`）。


## 3. 实现 `process` 方法

此方法定义了您流水线的核心逻辑。它接收输入数据（例如，图像路径、PIL 图像），按顺序调用初始化的预测器的 `predict` 方法，处理中间结果，并返回最终输出。

```python
from PIL import Image
from typing import Optional
from myocr.types import OCRResult # 导入必要的数据结构

class MyDetectionOnlyPipeline(Pipeline):
    def __init__(self, device: Device, detection_model_name: str = "dbnet++.onnx"):
        # ... (上一步的初始化代码) ...
        super().__init__()
        self.device = device
        
        det_model_path = MODEL_PATH + detection_model_name
        det_model = ModelLoader().load("onnx", det_model_path, self.device)
        det_processor = TextDetectionProcessor(det_model.device)
        self.det_predictor = Predictor(det_processor)
        logger.info(f"DetectionOnlyPipeline 使用 {detection_model_name} 在 {device.name} 上初始化完成")

    def process(self, image_path: str) -> Optional[OCRResult]:
        """处理图像文件并返回检测到的对象。"""
        # 1. 加载图像 (示例：处理路径输入)
        image = Image.open(image_path).convert("RGB")
        if image is None:
            return None
            
        # 2. 运行检测预测器
        detected_objects = self.det_predictor.predict(image)

        # 3. 返回结果
        if detected_objects:
            logger.info(f"检测成功：找到 {len(detected_objects.bounding_boxes)} 个框。")
        else:
            logger.info("检测成功：未找到文本框。")
            
        return detected_objects # 返回检测预测器的输出
```


## 4. 使用您的自定义流水线

定义后，您可以像使用内置流水线一样导入和使用您的自定义流水线。

```python
# from your_module import MyDetectionOnlyPipeline # 或 MyFullOCRPipeline
from myocr.modeling.model import Device

pipeline = MyDetectionOnlyPipeline(device=Device('cuda:0'))
results = pipeline.process('path/to/image.jpg')

if results:
    # 处理来自您的自定义流水线的结果
    print(f"找到 {len(results.bounding_boxes)} 个文本区域。")
```

请记得在您的流水线逻辑中处理模型加载或预测步骤中可能出现的错误。 