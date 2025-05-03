---
hide:
  - navigation
---
# 欢迎来到 MyOCR 文档

<div align="center">
    <img width="150" alt="myocr logo" src="../assets/images/logomain.png">
</div>

**MyOCR 是一个高度可扩展和可定制的框架，用于简化生产级 OCR 系统的开发和部署。**

MyOCR 让您可以轻松训练自定义模型并将其无缝集成到您自己的 OCR 流水线中。

## 主要特性

**⚡️ 端到端 OCR 开发框架** – 专为开发者设计，用于在统一且灵活的流水线中构建和集成检测、识别和自定义 OCR 模型。

**🛠️ 模块化与可扩展** – 混合搭配组件 - 以最小的更改交换模型、预测器或输入输出处理器。

**🔌 开发者友好设计** - 清晰的 Python API、预构建的流水线和处理器，以及简单直接的训练和推理自定义。

**🚀 生产就绪性能** – 支持 ONNX runtime 实现快速 CPU/GPU 推理，支持多种部署方式。

## 快速入门

1.  **[安装](./getting-started/installation.md):** 设置 MyOCR 并下载必要的模型。
2.  **[概览](./getting-started/overview.md):** 理解核心概念（模型、预测器、流水线）以构建您的 OCR 能力。
3.  **[推理指南](./inference/local.md):** 学习如何使用 MyOCR 运行 OCR 任务。

## 核心概念

*   **[模型](./models/index.md):** 了解支持的模型类型（ONNX、PyTorch、自定义）和架构。
*   **[预测器](./predictors/index.md):** 了解模型如何与输入/输出处理器包装成 `Predictor`。
*   **[流水线](./pipelines/index.md):** 探索协调预测器以实现端到端 OCR 的高级流水线。

## 其他资源

*   **[常见问题解答](./faq.md):** 查找常见问题的答案。
*   **[更新日志](./CHANGELOG.md):** 查看最近的更新和变更。
*   **[贡献指南](./CONTRIBUTING.md):** 了解如何为项目做出贡献。
*   **[GitHub 仓库](https://github.com/robbyzhaox/myocr):** 源代码、问题和讨论。

## 许可证

MyOCR 基于 [Apache 2.0 许可证](https://github.com/robbyzhaox/myocr/blob/main/LICENSE) 开源。
