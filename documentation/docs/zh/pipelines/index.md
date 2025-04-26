# 流水线 (Pipelines)

MyOCR 流水线协调多个组件（预测器、模型）来执行端到端的 OCR 任务。它们为处理图像或文档提供了一个高级接口。

## 可用流水线

### 1. `CommonOCRPipeline`

*   **文件:** `myocr/pipelines/common_ocr_pipeline.py`
*   **用途:** 执行标准 OCR：文本检测、可选的文本方向分类和文本识别。

**初始化:**

```python
from myocr.pipelines import CommonOCRPipeline
from myocr.modeling.model import Device

# 初始化用于 GPU (或 'cpu') 的流水线
pipeline = CommonOCRPipeline(device=Device('cuda:0'))
```

**配置:**

流水线从 `myocr/pipelines/config/common_ocr_pipeline.yaml` 加载配置。该文件指定了用于检测、分类和识别的 ONNX 模型的路径，这些路径是相对于 `myocr.config` 中定义的 `MODEL_PATH`（默认为 `~/.MyOCR/models/`）的。

```yaml
# 示例: myocr/pipelines/config/common_ocr_pipeline.yaml
model:
  detection: "dbnet++.onnx" # 默认检测模型
  cls_direction: "cls.onnx"   # 默认分类模型
  recognition: "rec.onnx"   # 默认识别模型
```

**处理:**

`process` 方法接收一个图像文件的路径。

```python
image_path = 'path/to/your/image.png'
ocr_results = pipeline.process(image_path)

if ocr_results:
    # 访问合并后的识别文本
    print("--- 完整文本 ---")
    print(ocr_results.get_content_text())
    print("-----------------")

    # 访问单个文本项及其属性
    print("--- 文本项 ---")
    for text_item in ocr_results.text_items:
        box = text_item.bounding_box
        print(f"  文本: \"{text_item.text}\"")
        print(f"  置信度: {text_item.confidence:.4f}")
        # 边界框详情 (左, 上, 右, 下, 角度信息, 检测分数)
        print(f"  框坐标 (L,T,R,B): ({box.left}, {box.top}, {box.right}, {box.bottom})") 
        if hasattr(box, 'angle') and box.angle:
             print(f"  角度: {box.angle[0]} 度, 置信度: {box.angle[1]:.4f}")
        print(f"  检测分数: {box.score:.4f}")
        print("---")
else:
    print(f"在 {image_path} 中未找到文本")
```

**工作流程:**

1.  加载图像。
2.  使用 `TextDetectionPredictor` 查找文本区域。
3.  使用 `TextDirectionPredictor` 对检测到的区域进行方向分类，并在内部旋转裁剪区域。
4.  使用 `TextRecognitionPredictor` 识别每个定向区域内的文本。
5.  返回一个 `RecognizedTexts` 对象，其中包含 `TextItem` 对象的列表（每个对象包含文本、置信度和边界框信息）。

### 2. `StructuredOutputOCRPipeline`

*   **文件:** `myocr/pipelines/structured_output_pipeline.py`
*   **用途:** 扩展 `CommonOCRPipeline`，增加了一个步骤，使用 LLM 通过 `OpenAiChatExtractor` 从识别的文本中提取结构化信息 (JSON)。

**初始化:**

需要一个设备和一个定义所需 JSON 输出模式的 Pydantic 模型。您可以使用预定义的模式或创建自己的模式。

```python
from myocr.pipelines import StructuredOutputOCRPipeline
from myocr.pipelines.response_format import InvoiceModel # 使用预定义的模型
from myocr.modeling.model import Device

# 使用预定义的 InvoiceModel schema 初始化流水线
pipeline = StructuredOutputOCRPipeline(device=Device('cuda:0'), json_schema=InvoiceModel)

# --- 或者，使用自定义 Pydantic 模型 ---
# from pydantic import BaseModel, Field
# class CustomSchema(BaseModel):
#     field_a: str = Field(description="描述 A")
#     field_b: int = Field(description="描述 B")
# pipeline = StructuredOutputOCRPipeline(device=Device('cuda:0'), json_schema=CustomSchema)
```

**配置:**

此流水线从 `myocr/pipelines/config/structured_output_pipeline.yaml` 加载其特定配置，其中包括 `OpenAiChatExtractor` 的设置（LLM 模型名称、API 基础 URL、API 密钥）。

```yaml
# 示例: myocr/pipelines/config/structured_output_pipeline.yaml
chat_bot:
  model: "gpt-4o" # 或其他兼容模型，如 qwen, ollama 模型
  base_url: "https://api.openai.com/v1" # 或您的本地 LLM 服务器端点
  api_key: "YOUR_API_KEY" # 或环境变量引用
```
**重要提示:** 确保为所选的 LLM 提供商（OpenAI、本地 Ollama 等）正确配置了 `api_key` 和 `base_url`。

**处理:**

`process` 方法接收一个图像路径。

```python
image_path = 'path/to/your/invoice.pdf' # 或其他图像格式
structured_data = pipeline.process(image_path)

if structured_data:
    # 结果是在初始化时传递的 Pydantic 模型的实例 (例如 InvoiceModel)
    print("--- 提取的数据 (JSON) ---")
    print(structured_data.model_dump_json(indent=2))
    print("---------------------------")

    # 直接访问提取的字段 (如果使用 InvoiceModel)
    # print(f"发票号码: {structured_data.invoiceNumber}")
    # print(f"总金额: {structured_data.totalAmount}")
else:
     print(f"无法从 {image_path} 提取结构化数据")

```

**工作流程:**

1.  使用继承的 `CommonOCRPipeline.process` 方法执行标准 OCR 以获取原始识别文本 (`RecognizedTexts` 对象)。
2.  如果找到文本，则将合并的文本内容 (`get_content_text()`) 传递给 `OpenAiChatExtractor`。
3.  提取器与配置的 LLM 交互，提供文本和所需的 `json_schema` (Pydantic 模型) 作为指令。
4.  LLM 尝试提取相关信息并根据模式对其进行格式化。
5.  返回填充了提取数据的所提供 Pydantic 模型的实例，如果提取失败则返回 `None`。

## 定制

可以通过以下方式定制流水线：

*   修改 `myocr/pipelines/config/` 中的 `.yaml` 配置文件以使用不同的模型或更改提取器设置。
*   创建继承自 `Pipeline`（在 `myocr/base.py` 中定义）或现有流水线的新流水线类。
*   集成不同的预测器或提取器。 