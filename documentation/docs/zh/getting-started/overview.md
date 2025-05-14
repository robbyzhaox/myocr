# 概览

MyOCR 是一个功能强大且灵活的框架，专为帮助开发者构建和部署自定义 OCR 方案而设计，同时兼顾了生产可用性与开发体验。该库提供了一套高层次的组件架构，便于快速集成与扩展。


## 核心组件

MyOCR 围绕以下关键概念构建：

![MyOCR 组件](../assets/images/components.png)

*   **模型 (Model):** 表示一个神经网络模型。MyOCR 支持加载 ONNX 模型 (`OrtModel`)、标准 PyTorch 模型 (`PyTorchModel`) 以及通过 YAML 配置定义的自定义 PyTorch 模型 (`CustomModel`)。模型负责执行核心计算。
    *   更多详情请参阅 [模型部分](../models/index.md)。
*   **处理器 (`CompositeProcessor`):** 负责为模型准备输入数据，并将模型的原始输出转换为更易用的格式。每个预测器都配备了特定的处理器。
    *   处理器详情请参阅 [预测器部分](../predictors/index.md)。
*   **预测器 (Predictor):** 将一个 `Model` 和一个 `Processor` 结合起来执行特定的推理任务（如文本检测）。它提供了一个用户友好的接口，接受标准输入（如 PIL 图像）并返回处理后的结果（如边界框）。
    *   可用预测器列表请参阅 [预测器部分](../predictors/index.md)。
*   **流水线 (Pipeline):** 协调多个 `Predictors` 来执行复杂的多步骤任务，如端到端 OCR。流水线为最常见的用例提供了最高层次的接口。
    *   可用流水线列表请参阅 [流水线部分](../pipelines/index.md)。


**类图**
![MyOCR 类图](../assets/images/myocr_class_diagram.png)


## 定制与扩展

MyOCR 的模块化设计让定制变得简单：

*   **[添加新模型](../models/add-model.md):** 了解如何通过模型加载器引入新模型。
*   **[创建自定义预测器](../predictors/create-predictor.md):** 了解如何利用 `Model` 和 `CompositeProcessor` 创建自定义 `Predictor`。
*   **[构建自定义流水线](../pipelines/build-pipeline.md):** 了解如何将多个 `Predictor` 组合成完整的 `Pipeline`。
