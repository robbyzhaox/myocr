# 创建自定义预测器

MyOCR 中的预测器充当已加载 `Model`（ONNX 或 PyTorch）与最终用户或流水线之间的桥梁。它们封装了必要的预处理和后处理逻辑，以便模型能够轻松地用于特定任务。

虽然 MyOCR 提供了标准预测器（通过 `TextDetectionProcessor`、`TextRecognitionProcessor` 等处理器），但在以下情况下，您可能需要自定义预测器：

*   您的模型需要独特的输入预处理（例如，不同的归一化、调整大小、输入格式）。
*   您的模型产生的输出需要自定义解码或格式化（例如，不同的边界框格式、专门的分类标签、现有流水线无法处理的结构化输出）。
*   您想为检测、识别或分类之外的全新任务创建预测器。

构建自定义预测器的关键是创建自定义的 **`CompositeProcessor`** 类。

## 1. 理解 `CompositeProcessor` 的作用

预测器本身是一个简单的包装器（在 `myocr.base.Predictor` 中定义）。实际工作在其关联的 `CompositeProcessor`（继承自 `myocr.base.CompositeProcessor` 的类）中进行。处理器主要有两个任务：

1.  **`preprocess(user_input)`:** 接收用户或流水线提供的数据（例如 PIL 图像），并将其转换为模型推理方法所期望的精确格式（例如，归一化的、具有批次维度的 NumPy 数组）。
2.  **`postprocess(model_output)`:** 接收模型的原始推理输出（例如，表示热力图或序列概率的 NumPy 数组），并将其转换为用户友好的、结构化的格式（例如，带有文本和分数的边界框列表，如 `TextRegion`）。

## 2. 创建自定义 `CompositeProcessor` 类

1.  **继承:** 创建一个继承自 `myocr.base.CompositeProcessor` 的 Python 类。
2.  **指定类型 (可选但推荐):** 使用泛型来指示 `preprocess` 的预期输入类型和 `postprocess` 的输出类型。例如，`CompositeProcessor[PIL.Image.Image, List[RectBoundingBox]]` 表示它接收 PIL 图像并返回 `List[RectBoundingBox]`。
3.  **实现 `__init__`:** 初始化任何必要的参数，例如阈值、标签映射或转换期间需要的引用。
4.  **实现 `preprocess`:** 编写代码将输入数据转换为模型就绪格式。
5.  **实现 `postprocess`:** 编写代码将原始模型输出转换为所需的结构化结果。

**注意：**具体代码请参考已有预测器

## 3. 创建预测器实例

一旦您有了自定义 `CompositeProcessor` 并加载了模型，就可以创建预测器实例。


## 4. 集成到流水线 (可选)

如果您的自定义预测器是更大工作流的一部分，您可以将其集成到 [自定义流水线](./../pipelines/build-pipeline.md) 中，方法是在流水线的 `__init__` 方法中初始化它，并在流水线的 `process` 方法中调用其 `predict` 方法。

通过这些步骤，您可以在 MyOCR 框架内创建针对特定模型和任务的预测器。 