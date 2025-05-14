# 本地推理

本节描述如何使用 MyOCR 库执行推理，主要通过使用预定义的流水线。

## 使用流水线进行推理

执行端到端 OCR 的推荐方法是通过 `myocr.pipelines` 中提供的流水线类。这些流水线处理必要模型的加载以及检测、分类和识别步骤的编排。

### 使用 `CommonOCRPipeline` 进行标准 OCR

此流水线适用于通用 OCR 任务，目标是从图像中提取所有文本及其位置。

```python
from myocr.pipelines import CommonOCRPipeline

# Initialize common OCR pipeline (using GPU)
pipeline = CommonOCRPipeline("cuda:0")  # Use "cpu" for CPU mode

# Perform OCR recognition on an image
result = pipeline("path/to/your/image.jpg")
print(result)
```

### 使用 `StructuredOutputOCRPipeline` 进行结构化数据提取

当您需要从文档中提取特定信息并根据预定义的 schema 将其格式化为 JSON 时，请使用此流水线。

```yaml
chat_bot:
  model: qwen2.5:14b
  base_url: http://127.0.0.1:11434/v1
  api_key: 'key'

```

```python
from pydantic import BaseModel, Field
from myocr.pipelines import StructuredOutputOCRPipeline

# Define output data model, refer to:
from myocr.pipelines.response_format import InvoiceModel

# Initialize structured OCR pipeline
pipeline = StructuredOutputOCRPipeline("cuda:0", InvoiceModel)

# Process image and get structured data
result = pipeline("path/to/invoice.jpg")
print(result.to_dict())
```

## 直接使用预测器（高级）

虽然推荐使用流水线，但如果您需要对过程进行更精细的控制（例如，仅使用检测，或提供预处理输入），您可以直接使用单个预测器。有关初始化和使用每个预测器/处理器对的详细信息，请参阅**预测器**部分文档。

## 性能注意事项

*   **设备选择：** 使用支持 CUDA 的 GPU（`Device('cuda:0')`）比 CPU（`Device('cpu')`）显著加快推理速度。确保您安装了必要的驱动程序和 ONNX Runtime GPU 构建。
*   **模型选择：** 流水线 YAML 文件中配置的特定 ONNX 模型会影响性能和准确性。
*   **批处理：** 虽然当前的流水线示例处理单个图像，但预测器通常在内部处理批输入（例如，在识别中同时处理所有检测到的框）。对于处理大量图像，如果需要，考虑在应用程序级别进行并行执行或批处理。
