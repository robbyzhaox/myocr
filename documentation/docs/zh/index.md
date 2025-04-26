---
hide:
  - navigation
---
# 欢迎来到 MyOCR 文档

<div align="center">
    <img width="150" alt="myocr logo" src="assets/images/logomain.png">
</div>

**MyOCR 是一个 Python 库，旨在简化生产级 OCR (光学字符识别) 系统的开发和部署。**

无论您需要从文档中进行基本的文本提取，还是复杂的结构化数据提取，MyOCR 都提供了构建健壮高效流水线的工具。

## 主要特性

*   **🚀 端到端流程与模块化设计:** 通过组合模块化组件，构建完整的 OCR 工作流（检测、分类、识别）。
*   **🛠️ 可扩展:** 轻松集成自定义模型或处理逻辑。
*   **⚡ 生产就绪:** 基于 ONNX Runtime 优化，支持 CPU 与 GPU 高性能推理，为生产环境准备就绪。
*   **📊 结构化输出:** 利用 LLM 集成，将信息提取为预定义的 JSON 格式。
*   **🔌 多种使用模式:** 可作为 Python 库使用，部署为 REST API 服务，或在 Docker 中运行。

## 快速入门

1.  **[安装](./getting-started/installation.md):** 安装 MyOCR 并下载必要的模型。
2.  **[概览](./getting-started/overview.md):** 理解核心概念（模型、预测器、流水线）。
3.  **[推理指南](./inference/local.md):** 学习如何使用该库运行 OCR 任务。

## 核心概念深入了解

*   **[模型](./models/index.md):** 了解支持的模型类型（ONNX、PyTorch、自定义）和架构。
*   **[预测器](./predictors/index.md):** 理解预测器如何包装模型及其输入输出处理逻辑以执行特定任务。
*   **[流水线](./pipelines/index.md):** 探索协调多个预测器以实现端到端 OCR 的高级流水线。

## 部署选项

除了将 MyOCR 作为 Python 库使用（请参阅 [本地推理指南](./inference/local.md)），您还可以通过以下方式部署它：

*   **作为 REST API:**
    *   该服务使用 `gunicorn` 运行（如 Docker 镜像中所配置），通常监听端口 8000。
    *   对于本地开发/测试，您可以运行 `python main.py`，但这可能会使用不同的端口（请检查 `main.py`）。
    *   端点：`GET /ping`、`POST /ocr`（基本 OCR）、`POST /ocr-json`（结构化 OCR）。
    *   提供一个独立的 UI 界面：[doc-insight-ui](https://github.com/robbyzhaox/doc-insight-ui)
*   **使用 Docker:**
    *   使用 `Dockerfile-infer-CPU` 或 `Dockerfile-infer-GPU` 构建 CPU 或 GPU 特定镜像。
    *   使用辅助脚本进行构建（请替换 `[cpu|gpu]` 并确保 `VERSION` 正确）：
        ```bash
        # 首先，确定版本
        VERSION=$(python -c 'import myocr.version; print(myocr.version.VERSION)')
        # 构建所需的镜像 (cpu 或 gpu)
        bash scripts/build_docker_image.sh [cpu|gpu]
        ```
    *   运行示例（请将 `[cpu|gpu]` 和 `$VERSION` 替换为实际值）：
        ```bash
        # GPU 镜像版本 0.1.0 示例
        docker run -d -p 8000:8000 myocr:gpu-0.1.0
        # CPU 镜像版本 0.1.0 示例
        docker run -d -p 8000:8000 myocr:cpu-0.1.0
        ```
    *   容器内的服务运行在端口 8000。

## 其他资源

*   **[常见问题解答](./faq.md):** 查找常见问题的答案。
*   **[更新日志](./CHANGELOG.md):** 查看最近的更新和变更。
*   **[贡献指南](./CONTRIBUTING.md):** 了解如何为项目做出贡献。
*   **[GitHub 仓库](https://github.com/robbyzhaox/myocr):** 源代码、问题和讨论。

## 许可证

MyOCR 基于 [Apache 2.0 许可证](https://github.com/robbyzhaox/myocr/blob/main/LICENSE) 开源。
