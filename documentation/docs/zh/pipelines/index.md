# 流水线

MyOCR 流水线协调多个组件（预测器、模型）来执行端到端的 OCR 任务。它们为处理图像或文档提供了一个高级接口。

## 可用流水线

###  `CommonOCRPipeline`

定义在 `myocr/pipelines/common_ocr_pipeline.py` 中。

此流水线执行标准 OCR：文本检测、可选的文本方向分类和文本识别。

**初始化:**

```python
from myocr.pipelines import CommonOCRPipeline
from myocr.modeling.model import Device

# 初始化用于 GPU (或 'cpu') 的流水线
pipeline = CommonOCRPipeline(device=Device('cuda:0'))
```

**配置:**

流水线从 `myocr/pipelines/config/common_ocr_pipeline.yaml` 加载配置。该文件指定了用于检测、分类和识别的 ONNX 模型的路径，这些路径是相对于 `myocr.config` 中定义的 `MODEL_PATH` 的。

```yaml
# 示例: myocr/pipelines/config/common_ocr_pipeline.yaml
model:
  detection: "dbnet++.onnx"
  cls_direction: "cls.onnx"
  recognition: "rec.onnx"
```

**处理:**

`__call__` 方法接收一个图像文件的路径。

```python
image_path = 'path/to/your/image.png'
ocr_results = pipeline(image_path)

if ocr_results:
    # 访问识别的文本和边界框
    print(ocr_results)
```

**工作流程:**

1.  加载图像。
2.  使用 `TextDetectionPredictor` 查找文本区域。
3.  使用 `TextDirectionPredictor` 对检测到的区域进行方向分类。
4.  使用 `TextRecognitionPredictor` 识别每个定向区域内的文本。
5.  返回一个结果对象，其中包含边界框、文本和可能的置信度分数（详细信息取决于 `Predictor` 的实现）。

###  `StructuredOutputOCRPipeline`

定义在 `myocr/pipelines/structured_output_pipeline.py` 中。

此流水线通过添加一个步骤来扩展 `CommonOCRPipeline`，该步骤使用大型语言模型（LLM）通过 `OpenAiChatExtractor` 从识别的文本中提取结构化信息（例如 JSON）。

**初始化:**

需要一个设备和一个定义所需 JSON 输出模式的 Pydantic 模型。

```python
from myocr.pipelines import StructuredOutputOCRPipeline
from myocr.modeling.model import Device
from pydantic import BaseModel, Field

# 定义您所需的输出结构
class InvoiceData(BaseModel):
    invoice_number: str = Field(description="发票号码")
    total_amount: float = Field(description="应付总额")
    due_date: str = Field(description="付款截止日期")

# 初始化流水线
pipeline = StructuredOutputOCRPipeline(device=Device('cuda:0'), json_schema=InvoiceData)
```

**配置:**

此流水线从 `myocr/pipelines/config/structured_output_pipeline.yaml` 加载其特定配置，其中包括 `OpenAiChatExtractor` 的设置（LLM 模型名称、API 基础 URL、API 密钥）。

```yaml
# 示例: myocr/pipelines/config/structured_output_pipeline.yaml
chat_bot:
  model: "gpt-4o"
  base_url: "https://api.openai.com/v1"
  api_key: "YOUR_API_KEY"
```

**处理:**

`__call__` 方法接收一个图像路径。

```python
image_path = 'path/to/your/invoice.pdf'
structured_data = pipeline(image_path)

if structured_data:
    print(structured_data)
```

**工作流程:**

1.  使用继承的 `CommonOCRPipeline` 执行标准 OCR 以获取原始识别文本。
2.  如果找到文本，则将文本内容传递给 `OpenAiChatExtractor`。
3.  提取器与配置的 LLM 交互，提供文本和所需的 `json_schema`（Pydantic 模型）作为指令。
4.  LLM 尝试提取相关信息并根据模式对其进行格式化。
5.  返回填充了提取数据的所提供 Pydantic 模型的实例。


## 性能优化

### 批处理

```python
# 处理多个图像
results = [pipeline(img_path) for img_path in image_paths]
```

### GPU 加速

```python
# 使用 GPU 进行更快的处理
pipeline = CommonOCRPipeline("cuda:0")
```

### 内存管理

```python
# 清理 GPU 内存
import torch
torch.cuda.empty_cache()
```

## 错误处理

流水线处理各种错误情况：

- 无效的图像格式
- 缺失的模型文件
- GPU 内存不足
- 无效的配置

有关常见问题和解决方案，请参阅[故障排除指南](../faq.md)。 