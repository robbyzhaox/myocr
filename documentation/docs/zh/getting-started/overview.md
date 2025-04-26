# 概览

欢迎使用 MyOCR！该库提供了一个强大而灵活的框架，用于构建和部署光学字符识别 (OCR) 流水线。

## 为何选择 MyOCR？

MyOCR 的设计兼顾了生产准备和开发者体验。主要特性包括：

*   **端到端工作流:** 无缝集成文本检测、方向分类和文本识别。
*   **模块化与可扩展:** 轻松替换模型、预处理/后处理步骤（通过转换器）或整个流水线。
*   **生产优化:** 利用 ONNX Runtime 实现高性能的 CPU 和 GPU 推理。
*   **结构化数据提取:** 超越原始文本，使用 LLM 将信息提取为结构化格式（如 JSON）。
*   **开发者友好:** 提供简洁的 Python API 和预构建组件，以便快速上手。

## 核心组件

MyOCR 围绕几个关键概念构建：

### 组件图
![MyOCR 组件](../assets/images/components.png)


### 类图
![MyOCR 类图](../assets/images/myocr_class_diagram.png)

*   **模型 (Model):** 代表一个神经网络模型。MyOCR 支持加载 ONNX 模型 (`OrtModel`)、标准 PyTorch 模型 (`PyTorchModel`) 以及由 YAML 配置定义的自定义 PyTorch 模型 (`CustomModel`)。模型负责核心计算。
    *   更多详情请参阅 [模型部分](../models/index.md)。
*   **转换器 (Converter / `ParamConverter`):** 为模型准备输入数据，并将模型的原始输出处理成更易用的格式。每个预测器都使用特定的转换器。
    *   转换器详情请参阅 [预测器部分](../predictors/index.md)。
*   **预测器 (Predictor):** 结合一个 `Model` 和一个 `ParamConverter` 来执行特定的推理任务（例如文本检测）。它提供了一个用户友好的接口，接受标准输入（如 PIL 图像）并返回处理后的结果（如边界框）。
    *   可用预测器列表请参阅 [预测器部分](../predictors/index.md)。
*   **流水线 (Pipeline):** 协调多个 `Predictors` 来执行复杂的多步骤任务，如端到端 OCR。流水线为最常见的用例提供了最高级别的接口。
    *   可用流水线列表请参阅 [流水线部分](../pipelines/index.md)。

## 定制与扩展

MyOCR 的模块化设计允许轻松定制。

*   **[添加新模型](../models/add-model.md):** 了解如何通过模型加载器引入新模型。
*   **[创建自定义预测器](../predictors/create-predictor.md):** 了解如何通过 `Model` 和 `ParamConverter` 创建自定义 `Predictor`。
*   **[构建自定义流水线](../pipelines/build-pipeline.md):** 了解如何将多个 `Predictor` 组合成 `Pipeline`。
