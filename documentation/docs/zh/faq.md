# FAQ - 常见问题解答

### 问：MyOCR 默认在哪里查找模型？

**答：** 默认路径在 `myocr/config.py` 中的 `MODEL_PATH` 变量中配置，在 Linux/macOS 系统上通常解析为 `~/.MyOCR/models/` 目录。流水线配置文件 (`myocr/pipelines/config/*.yaml`) 中引用的模型文件名都是相对于此目录的。如果您想将模型存储在其他位置，可以更改 `MODEL_PATH` 值，或在 YAML 配置中使用绝对路径。

### 问：如何在 CPU 和 GPU 推理之间切换？

**答：** 在初始化流水线或模型时，传入一个来自 `myocr.modeling.model` 的 `Device` 对象即可。

*   使用 GPU（前提是已正确配置 CUDA）：`Device('cuda:0')`（对应第一个 GPU）。

*   使用 CPU：`Device('cpu')`。

请确保安装了正确版本的 `onnxruntime` 包（CPU 版本用 `onnxruntime`，GPU 版本用 `onnxruntime-gpu`），并且在使用 GPU 时安装了兼容的 CUDA 驱动程序。

### 问：`StructuredOutputOCRPipeline` 不工作或出现错误，该怎么办？

**答：** 这个流水线依赖于外部大型语言模型 (LLM)，请检查以下几点：

1.  **配置检查：** 确保 `myocr/pipelines/config/structured_output_pipeline.yaml` 文件中已为您选择的 LLM 提供商（如 OpenAI、Ollama 或本地服务器）正确配置了 `model`、`base_url` 和 `api_key` 参数。

2.  **API 密钥：** 确保 API 密钥已正确设置（可以直接在 YAML 中指定，或通过 YAML 指向的环境变量，如 `OPENAI_API_KEY`）。

3.  **连接检查：** 验证您的环境能够访问 LLM API 的 `base_url`。

4.  **Schema 验证：** 确保初始化时传入的 Pydantic `json_schema` 有效，并且各字段描述能充分指导 LLM 理解您的需求。

### 问：Predictor（预测器）和 Pipeline（流水线）有什么区别？

**答：**

*   **预测器：** 底层组件，它封装了单个 `Model` 及其特定的预处理和后处理逻辑（在 `CompositeProcessor` 中定义）。预测器专注于处理单一任务（如文本检测）。

*   **流水线：** 高层组件，它协调多个 `Predictors` 来完成一个完整的工作流（如结合检测、分类和识别的端到端 OCR）。流水线为用户提供了处理常见任务的主要接口。

### 问：如何使用自己的自定义模型？

**答：**

*   **ONNX 模型：** 将您的 `.onnx` 文件放入模型目录，然后更新相关的流水线配置 YAML 文件 (`myocr/pipelines/config/*.yaml`)，使其指向您的模型文件。详情请参阅 [概览](./getting-started/overview.md#替换或添加新模型-onnx) 部分。

*   **自定义 PyTorch 模型：** 使用 `myocr/modeling/` 目录中的组件（主干网络、颈部、头部模块）定义您的模型架构，并创建一个描述该架构的 YAML 配置文件。然后使用 `ModelLoader().load(model_format='custom', ...)` 加载它，或创建一个自定义流水线/预测器。关于 `CustomModel` 和 YAML 配置的详细信息，请参阅 [模型文档](./models/index.md)。
