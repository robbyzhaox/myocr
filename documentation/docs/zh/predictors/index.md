# 预测器

预测器负责处理 MyOCR 中特定模型（检测、识别、分类）的推理逻辑。它们通过整合预处理和后处理步骤，弥合了原始模型输出与可用结果之间的差距。

预测器通常与一个 `Model` 对象和一个 `CompositeProcessor` 相关联。

*   **模型 (Model):** 提供核心的 `forward_internal` 方法（例如，ONNX 会话运行、PyTorch 模型前向传播）。
*   **CompositeProcessor:** 处理将输入数据转换为模型期望的格式，并将模型的原始输出转换为结构化的、有意义的格式。

## 基础组件

*   **`myocr.base.Predictor`:** 一个简单的包装器，调用 `CompositeProcessor` 的输入转换、`Model` 的前向传播以及 `CompositeProcessor` 的输出转换。
*   **`myocr.base.CompositeProcessor`:** 定义了 `preprocess` 和 `postprocess` 方法的抽象基类。
*   **`myocr.predictors.base`:** 定义了通用的数据结构，如 `BoundingBox`、`RectBoundingBox` 和 `TextRegion`，用作不同处理器的输入和输出。

## 可用的预测器和处理器

预测器是在加载的 `Model` 实例上调用 `Predictor(processor)` 方法时隐式创建的。关键组件是 `CompositeProcessor` 的实现：

### 1. 文本检测 (`TextDetectionProcessor`)

*   **文件:** `myocr/processors/text_detection_processor.py`
*   **输入:** `PIL.Image`
*   **输出:** `List[RectBoundingBox]`（包含原始图像和 `RectBoundingBox` 列表）
*   **关联模型:** 通常是 DBNet/DBNet++ ONNX 模型。

### 2. 文本方向分类 (`TextDirectionProcessor`)

*   **文件:** `myocr/processors/text_direction_processor.py`
*   **输入:** `List[RectBoundingBox]`
*   **输出:** `List[RectBoundingBox]`（每个 `RectBoundingBox` 中的 `angle` 属性已更新）
*   **关联模型:** 通常是简单的 CNN 分类器 ONNX 模型。

### 3. 文本识别 (`TextRecognitionProcessor`)

*   **文件:** `myocr/processors/text_recognition_processor.py`
*   **输入:** `List[RectBoundingBox]`（来自文本方向分类的输出）
*   **输出:** `List[TextRegion]]`
*   **关联模型:** 通常是基于 CRNN 的 ONNX 模型。

## 用法示例 (概念性)

```python
import cv2
from myocr.modeling.model import ModelLoader, Device
from myocr.processors import TextDetectionProcessor, TextDirectionProcessor, TextRecognitionProcessor

# 假设模型已加载
det_model = ModelLoader().load('onnx', 'path/to/det_model.onnx', Device('cuda:0'))
cls_model = ModelLoader().load('onnx', 'path/to/cls_model.onnx', Device('cuda:0'))
rec_model = ModelLoader().load('onnx', 'path/to/rec_model.onnx', Device('cuda:0'))

# 通过将模型与处理器关联来创建预测器
dec_predictor = Predictor(det_model, TextDetectionProcessor(det_model.device))
cls_predictor = Predictor(cls_model, TextDirectionProcessor())
rec_predictor = Predictor(rec_model, TextRecognitionProcessor())

# 加载图像
img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)

# 运行预测步骤
detected_objects = det_predictor.predict(img)
if detected_objects:
  classified_objects = cls_predictor.predict(detected_objects) # Predict 调用处理器步骤 + 模型前向传播
  recognized_texts = rec_predictor.predict(classified_objects)

  print(recognized_texts.get_content_text())
```

## 性能提示

### 批处理

```python
# 处理多个区域
results = [predictor.predict(region) for region in regions]
```

### 内存优化

```python
# 清理 GPU 内存
import torch
torch.cuda.empty_cache()
```

## 错误处理

预测器处理各种错误情况：

- 无效的输入格式
- 模型加载错误
- GPU 内存问题
- 推理错误

有关常见问题和解决方案，请参阅[故障排除指南](../faq.md)。 