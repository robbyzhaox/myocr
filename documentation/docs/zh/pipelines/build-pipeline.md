# 构建自定义流水线

MyOCR 的流水线协调多个预测器来执行复杂的任务。虽然库提供了像 `CommonOCRPipeline` 和 `StructuredOutputOCRPipeline` 这样的标准流水线，但您可能需要为特定的工作流创建自定义流水线，例如：

*   使用不同的模型或预测器组合。
*   添加自定义的预处理或后处理步骤。
*   集成标准 OCR 之外的组件（例如，OCR 前的图像增强、布局分析）。
*   处理不同的输入/输出类型。

本指南解释了构建您自己的流水线的步骤。

## 1. 继承自 `base.Pipeline`

所有流水线都应继承自抽象基类 `myocr.base.Pipeline`。

```python
from myocr.base import Pipeline

class MyCustomPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        # 初始化逻辑在这里
        pass

    def process(self, input_data):
        # 处理逻辑在这里
        pass
```

## 2. 在 `__init__` 中初始化预测器

`__init__` 方法通常是您加载模型并创建流水线将使用的预测器实例的地方。

*   **加载模型：** 使用 `myocr.modeling.model.ModelLoader` 加载所需的 ONNX 或自定义 PyTorch 模型。
*   **实例化处理器：** 创建所需的 `CompositeProcessor` 类的实例（例如 `TextDetectionProcessor`、`TextRecognitionProcessor` 或自定义的处理器）。
*   **创建预测器：** 使用 `Predictor(processor)` 方法组合加载的模型和处理器。
*   **存储预测器：** 将创建的预测器实例存储为流水线类的属性（例如 `self.det_predictor`）。

```python
import logging
from myocr.base import Pipeline, Predictor
from myocr.modeling.model import ModelLoader, Device
from myocr.config import MODEL_PATH # 默认模型目录路径
from myocr.processors import TextDetectionProcessor
# 如果需要，导入任何自定义处理器或模型

logger = logging.getLogger(__name__)

class MyDetectionOnlyPipeline(Pipeline):
    def __init__(self, device: Device, detection_model_name: str = "dbnet++.onnx"):
        super().__init__()
        self.device = device
        # --- 加载检测模型 ---
        det_model_path = MODEL_PATH + detection_model_name
        det_model = ModelLoader().load("onnx", det_model_path, self.device)
        
        # --- 创建检测预测器 ---
        det_processor = TextDetectionProcessor(det_model.device)
        self.det_predictor = Predictor(det_processor)
        logger.info(f"DetectionOnlyPipeline 使用 {detection_model_name} 在 {device.name} 上初始化完成")
        
    def process(self, input_data):
        # 在下一步中实现
        pass
```

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

**示例：组合预测器（概念性）**

如果您需要多个步骤，可以链式调用预测器，将一个步骤的输出作为下一个步骤的输入（如果兼容）。

```python
class MyFullOCRPipeline(Pipeline):
    def __init__(self, device: Device):
        super().__init__()
        self.device = device
        # --- 加载 det, cls, rec 模型 --- (假设路径正确)
        det_model = ModelLoader().load("onnx", MODEL_PATH + "dbnet++.onnx", device)
        cls_model = ModelLoader().load("onnx", MODEL_PATH + "cls.onnx", device)
        rec_model = ModelLoader().load("onnx", MODEL_PATH + "rec.onnx", device)
        
        # --- 创建预测器 ---
        self.det_predictor = Predictor(TextDetectionProcessor(device))
        self.cls_predictor = Predictor(TextDirectionProcessor())
        self.rec_predictor = Predictor(TextRecognitionProcessor())
        logger.info(f"MyFullOCRPipeline 在 {device.name} 上初始化完成")

    def process(self, image_path: str):
        logger.debug(f"正在处理 {image_path}")
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"加载图像时出错: {e}")
            return None

        # 步骤 1: 检测
        detected = self.det_predictor.predict(image)
        if not detected or not detected.bounding_boxes:
            logger.info("未检测到文本。")
            return None
        logger.debug(f"检测到 {len(detected.bounding_boxes)} 个区域。")

        # 步骤 2: 分类
        classified = self.cls_predictor.predict(detected)
        if not classified:
            logger.warning("分类步骤失败，将在没有角度校正的情况下继续。")
            classified = detected # 如果分类失败，则使用原始检测结果
        logger.debug("分类完成。")
            
        # 步骤 3: 识别
        recognized_texts = self.rec_predictor.predict(classified)
        if not recognized_texts:
            logger.warning("识别步骤失败。")
            return None
        logger.info("识别完成。")
        
        # 如果消费者需要，添加原始图像尺寸信息
        recognized_texts.original(image.size[0], image.size[1])
        return recognized_texts # 最终结果
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

请记住在您的流水线逻辑中处理模型加载或预测步骤中可能出现的错误。 