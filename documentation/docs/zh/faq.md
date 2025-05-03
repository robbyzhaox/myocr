# FAQ - 常见问题解答

### 问：MyOCR 默认在哪里查找模型？

**答：** 默认路径在 `myocr/config.py` 中配置 (`MODEL_PATH`)，在 Linux/macOS 上通常解析为 `~/.MyOCR/models/`。流水线配置文件 (`myocr/pipelines/config/*.yaml`) 中引用的模型文件名是相对于此目录的。如果您将模型存储在其他位置，可以更改 `MODEL_PATH` 或在 YAML 配置中使用绝对路径。

### 问：如何在 CPU 和 GPU 推理之间切换？

**答：** 在初始化流水线或模型时，传递一个 `myocr.modeling.model` 中的 `Device` 对象。

*   对于 GPU（假设 CUDA 已设置好）：`Device('cuda:0')`（针对第一个 GPU）。

*   对于 CPU：`Device('cpu')`。

请确保安装了正确的 `onnxruntime` 包（`onnxruntime` 用于 CPU，`onnxruntime-gpu` 用于 GPU），并且在使用 GPU 时安装了兼容的 CUDA 驱动程序。

### 问：`StructuredOutputOCRPipeline` 无法工作或出现错误。

**答：** 此流水线依赖于外部的大型语言模型 (LLM)。

1.  **检查配置：** 确保 `myocr/pipelines/config/structured_output_pipeline.yaml` 文件中为您选择的 LLM 提供商（例如 OpenAI、Ollama、本地服务器）设置了正确的 `model`、`base_url` 和 `api_key`。

2.  **API 密钥：** 确保 API 密钥已正确指定（直接在 YAML 中，或通过 YAML 指向的环境变量，如 `OPENAI_API_KEY`）。

3.  **连接性：** 验证您的环境可以访问为 LLM API 指定的 `base_url`。

4.  **Schema：** 确保在初始化期间传递的 Pydantic `json_schema` 是有效的，并且字段描述能够有效指导 LLM。

### 问：Predictor（预测器）和 Pipeline（流水线）有什么区别？

**答：**

*   **预测器：** 较低级别的组件，封装单个 `Model` 及其特定的预处理和后处理逻辑（在 `CompositeProcessor` 中定义）。它处理一个特定任务（例如文本检测）。

*   **流水线：** 较高级别的组件，协调多个 `Predictors` 以执行完整的工作流（例如端到端 OCR，结合检测、分类和识别）。流水线为常用任务提供主要的用户界面。

### 问：如何使用我自己的自定义模型？

**答：**

*   **ONNX 模型：** 将您的 `.onnx` 文件放入模型目录，并更新相关的流水线配置 YAML 文件 (`myocr/pipelines/config/*.yaml`)，使其指向您的模型文件名。请参阅 [概览](./getting-started/overview.md#replacing-or-adding-new-models-onnx) 部分。

*   **自定义 PyTorch 模型：** 使用 `myocr/modeling/` 中的组件（主干网络、颈部、头部）定义您的模型架构，并创建一个指定该架构的 YAML 配置文件。使用 `ModelLoader().load(model_format='custom', ...)` 加载它，或创建一个自定义流水线/预测器。有关 `CustomModel` 和 YAML 配置的详细信息，请参阅 [模型文档](./models/index.md)。
