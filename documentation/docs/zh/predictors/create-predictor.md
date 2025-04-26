# 创建自定义预测器

MyOCR 中的预测器充当已加载 `Model`（ONNX 或 PyTorch）与最终用户或流水线之间的桥梁。它们封装了必要的预处理和后处理逻辑，以便模型能够轻松地用于特定任务。

虽然 MyOCR 提供了标准预测器（通过 `TextDetectionParamConverter`、`TextRecognitionParamConverter` 等转换器），但在以下情况下，您可能需要自定义预测器：

*   您的模型需要独特的输入预处理（例如，不同的归一化、调整大小、输入格式）。
*   您的模型产生的输出需要自定义解码或格式化（例如，不同的边界框格式、专门的分类标签、现有流水线无法处理的结构化输出）。
*   您想为检测、识别或分类之外的全新任务创建预测器。

构建自定义预测器的关键是创建自定义的 **`ParamConverter`** 类。

## 1. 理解 `ParamConverter` 的作用

预测器本身是一个简单的包装器（在 `myocr.base.Predictor` 中定义）。实际工作在其关联的 `ParamConverter`（继承自 `myocr.base.ParamConverter` 的类）中进行。转换器主要有两个任务：

1.  **`convert_input(user_input)`:** 接收用户或流水线提供的数据（例如 PIL 图像），并将其转换为模型推理方法所期望的精确格式（例如，归一化的、具有批次维度的 NumPy 数组）。
2.  **`convert_output(model_output)`:** 接收模型的原始推理输出（例如，表示热力图或序列概率的 NumPy 数组），并将其转换为用户友好的、结构化的格式（例如，带有文本和分数的边界框列表，如 `DetectedObjects` 或 `RecognizedTexts`）。

## 2. 创建自定义 `ParamConverter` 类

1.  **继承:** 创建一个继承自 `myocr.base.ParamConverter` 的 Python 类。
2.  **指定类型 (可选但推荐):** 使用泛型来指示 `convert_input` 的预期输入类型和 `convert_output` 的输出类型。例如，`ParamConverter[PIL.Image.Image, DetectedObjects]` 表示它接收 PIL 图像并返回 `DetectedObjects`。
3.  **实现 `__init__`:** 初始化任何必要的参数，例如阈值、标签映射或转换期间需要的引用。
4.  **实现 `convert_input`:** 编写代码将输入数据转换为模型就绪格式。
5.  **实现 `convert_output`:** 编写代码将原始模型输出转换为所需的结构化结果。

```python
import logging
from typing import Optional, Tuple, List, Any
import numpy as np
from PIL import Image as PILImage

from myocr.base import ParamConverter
# 导入任何必要的基础结构或创建您自己的结构
from myocr.predictors.base import BoundingBox 

logger = logging.getLogger(__name__)

# --- 定义自定义输出结构 (示例) ---
class CustomResult:
    def __init__(self, label: str, score: float, details: Any):
        self.label = label
        self.score = score
        self.details = details

    def __repr__(self):
        return f"CustomResult(label='{self.label}', score={self.score:.4f}, details={self.details})"

# --- 创建自定义转换器 ---
# 示例：接收 PIL 图像，输出 CustomResult
class MyTaskConverter(ParamConverter[PILImage.Image, CustomResult]):
    def __init__(self, threshold: float = 0.5, target_size: Tuple[int, int] = (224, 224)):
        super().__init__()
        self.threshold = threshold
        self.target_size = target_size
        self.input_image_for_output = None # 如果输出转换需要，存储上下文
        logger.info(f"MyTaskConverter 初始化完成，阈值={threshold}, 目标尺寸={target_size}")

    def convert_input(self, input_data: PILImage.Image) -> Optional[np.ndarray]:
        """为一个假设的分类模型准备 PIL 图像。"""
        self.input_image_for_output = input_data # 如果需要，保存以供后续使用
        
        try:
            # 1. 调整大小
            image_resized = input_data.resize(self.target_size, PILImage.Resampling.BILINEAR)
            
            # 2. 转换为 NumPy 数组
            image_np = np.array(image_resized).astype(np.float32)
            
            # 3. 归一化 (示例：简单的 /255)
            image_np /= 255.0
            
            # 4. 如果需要，添加批次维度和通道维度 (例如 HWC -> NCHW)
            if image_np.ndim == 2: # 灰度图
                image_np = np.expand_dims(image_np, axis=-1)
            # 假设模型需要 NCHW
            image_np = np.expand_dims(image_np.transpose(2, 0, 1), axis=0) 
            
            logger.debug(f"转换后的输入图像形状: {image_np.shape}")
            return image_np.astype(np.float32)
            
        except Exception as e:
            logger.error(f"输入转换期间出错: {e}")
            return None

    def convert_output(self, internal_result: Any) -> Optional[CustomResult]:
        """处理一个假设的分类模型的原始输出。"""
        try:
            # 假设模型输出是一个包含 NumPy 分数数组的列表/元组
            scores = internal_result[0] # 示例: [[0.1, 0.8, 0.1]]
            if scores.ndim > 1: # 处理潜在的批次维度
                scores = scores[0]
                
            # 1. 找到最佳预测
            pred_index = np.argmax(scores)
            pred_score = float(scores[pred_index])
            
            logger.debug(f"原始分数: {scores}, 预测索引: {pred_index}, 分数: {pred_score}")

            # 2. 应用阈值
            if pred_score < self.threshold:
                logger.info(f"预测分数 {pred_score} 低于阈值 {self.threshold}")
                return None # 或返回默认/否定结果
                
            # 3. 将索引映射到标签 (假设存在预定义的映射)
            labels = ["猫", "狗", "其他"] # 示例标签
            pred_label = labels[pred_index] if pred_index < len(labels) else "未知"
            
            # 4. 格式化为 CustomResult
            # 包括任何额外的细节，可能使用 self.input_image_for_output
            result = CustomResult(label=pred_label, score=pred_score, details={"原始尺寸": self.input_image_for_output.size})
            
            return result

        except Exception as e:
            logger.error(f"输出转换期间出错: {e}")
            return None
```

## 3. 创建预测器实例

一旦您有了自定义转换器并加载了模型，就可以创建预测器实例。

```python
from myocr.modeling.model import ModelLoader, Device
from PIL import Image
# 假设 MyTaskConverter 如上定义

# 1. 加载您的模型 (ONNX 或 自定义 PyTorch)
model_path = "path/to/your/custom_model.onnx" # 或指向 CustomModel 的 YAML 路径
model_format = "onnx" # 或 "custom"
device = Device('cuda:0')

loader = ModelLoader()
model = loader.load(model_format, model_path, device)

# 2. 实例化您的自定义转换器
custom_converter = MyTaskConverter(threshold=0.6, target_size=(256, 256)) # 如果需要，使用自定义参数

# 3. 创建预测器
custom_predictor = model.predictor(custom_converter)

# 4. 使用预测器
input_image = Image.open("path/to/test_image.jpg").convert("RGB")
prediction_result = custom_predictor.predict(input_image) # 返回 CustomResult 或 None

if prediction_result:
    print(f"预测结果: {prediction_result}")
else:
    print("预测失败或低于阈值。")
```

## 4. 集成到流水线 (可选)

如果您的自定义预测器是更大工作流的一部分，您可以将其集成到 [自定义流水线](./../pipelines/build-pipeline.md) 中，方法是在流水线的 `__init__` 方法中初始化它，并在流水线的 `process` 方法中调用其 `predict` 方法。

通过遵循这些步骤，您可以在 MyOCR 框架内创建针对特定模型和任务的专门预测器。 