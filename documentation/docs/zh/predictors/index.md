# 预测器

预测器负责处理 MyOCR 中特定模型（检测、识别、分类）的推理逻辑。它们通过整合预处理和后处理步骤，弥合了原始模型输出与可用结果之间的差距。

预测器通常与一个 `Model` 对象和一个 `CompositeProcessor` 相关联。

*   **模型 (Model):** 提供核心的 `forward_internal` 方法（例如，ONNX 会话运行、PyTorch 模型前向传播）。
*   **CompositeProcessor:** 处理将输入数据转换为模型期望的格式，并将模型的原始输出转换为结构化的、有意义的格式。

## 基础组件

*   **`myocr.base.Predictor`:** 一个简单的包装器，调用 `CompositeProcessor` 的输入转换、`Model` 的前向传播以及 `CompositeProcessor` 的输出转换。
*   **`myocr.base.CompositeProcessor`:** 定义了 `preprocess` 和 `postprocess` 方法的抽象基类。

## 可用的预测器和处理器

预测器是在加载的 `Model` 实例上调用 `Predictor(processor)` 方法时隐式创建的。关键组件是 `CompositeProcessor` 的实现：

###  文本检测 (`TextDetectionProcessor`)

*   **文件:** `myocr/processors/text_detection_processor.py`
*   **关联模型:** 通常是 DBNet/DBNet++ ONNX 模型。

###  文本方向分类 (`TextDirectionProcessor`)

*   **文件:** `myocr/processors/text_direction_processor.py`
*   **关联模型:** 通常是简单的 CNN 分类器 ONNX 模型。

###  文本识别 (`TextRecognitionProcessor`)

*   **文件:** `myocr/processors/text_recognition_processor.py`
*   **关联模型:** 通常是基于 CRNN 的 ONNX 模型。


## 性能提示

### 批处理

```python
# 处理多个区域
results = [predictor.predict(region) for region in regions]
```
