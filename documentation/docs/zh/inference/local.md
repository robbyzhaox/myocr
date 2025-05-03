# 本地推理

本节描述如何使用 MyOCR 库执行推理，主要通过使用预定义的流水线。

## 使用流水线进行推理

执行端到端 OCR 的推荐方法是通过 `myocr.pipelines` 中提供的流水线类。这些流水线处理必要模型的加载以及检测、分类和识别步骤的编排。

### 使用 `CommonOCRPipeline` 进行标准 OCR

此流水线适用于通用 OCR 任务，目标是从图像中提取所有文本及其位置。

```python
import logging
from myocr.pipelines import CommonOCRPipeline
from myocr.modeling.model import Device
from PIL import Image

# 配置日志记录（可选）
logging.basicConfig(level=logging.INFO)

# --- 1. 初始化 ---
# 指定设备（'cuda:0' 代表 GPU 0，'cpu' 代表 CPU）
device = Device('cuda:0')

# 初始化流水线
# 这将加载 myocr/pipelines/config/common_ocr_pipeline.yaml 中指定的默认模型
ocr_pipeline = CommonOCRPipeline(device=device)

# --- 2. 处理 ---
image_path = 'path/to/your/document.jpg' # 或 .png, .tif 等

# 处理图像加载、检测、分类和识别
results = ocr_pipeline(image_path)

# --- 3. 处理结果 ---
if results:
    # 获取合并的文本内容
    full_text = results.get_content_text()
    print("--- 完整识别文本 ---")
    print(full_text)
    print("-----------------------------")

    # 访问单个文本框及其属性
    print("--- 单个文本框 ---")
    for text_item in results.text_items:
        box = text_item.bounding_box
        print(f"文本: \"{text_item.text}\"")
        print(f"  置信度: {text_item.confidence:.4f}")
        print(f"  框坐标 (L,T,R,B): ({box.left}, {box.top}, {box.right}, {box.bottom})") 
        print(f"  检测分数: {box.score:.4f}")
        print("---")
else:
    logging.warning(f"在图像 {image_path} 中未检测到文本")

```

### 使用 `StructuredOutputOCRPipeline` 进行结构化数据提取

当您需要从文档中提取特定信息并根据预定义的 schema 将其格式化为 JSON 时，请使用此流水线。

```python
import logging
from myocr.pipelines import StructuredOutputOCRPipeline
from myocr.modeling.model import Device
from pydantic import BaseModel, Field
import os

# 配置日志记录（可选）
logging.basicConfig(level=logging.INFO)

# --- 1. 定义输出 Schema ---
# 使用 Pydantic 定义您想要提取的信息结构
class ReceiptInfo(BaseModel):
    store_name: Optional[str] = Field(None, description="商店或供应商名称")
    total_amount: Optional[float] = Field(None, description="最终支付的总金额")
    transaction_date: Optional[str] = Field(None, description="交易日期 (YYYY-MM-DD)")
    items: List[str] = Field([], description="提到的购买商品列表")

# --- 2. 初始化 ---
device = Device('cuda:0')

# 确保设置了 OpenAI API 密钥（或直接提供）
# 假设密钥在环境变量 OPENAI_API_KEY 中
# 流水线配置文件 (structured_output_pipeline.yaml) 应该指向此密钥
if "OPENAI_API_KEY" not in os.environ:
  logging.warning("未设置 OPENAI_API_KEY 环境变量。LLM 提取可能会失败。")
  # 您可能需要在配置 yaml 中手动设置 api_key 或修改流水线代码

# 使用设备和所需的输出 schema 初始化流水线
structured_pipeline = StructuredOutputOCRPipeline(device=device, json_schema=ReceiptInfo)

# --- 3. 处理 ---
image_path = 'path/to/your/receipt.png'

# 此过程涉及 OCR 后跟基于 LLM 的提取
extracted_data = structured_pipeline(image_path)

# --- 4. 处理结果 ---
if extracted_data:
    print("--- 提取的信息 (JSON) ---")
    print(extracted_data.model_dump_json(indent=2))
    print("-------------------------------------")
else:
    # 如果 OCR 未找到文本或 LLM 无法提取符合 schema 的数据，可能会发生这种情况
    logging.warning(f"无法从图像 {image_path} 中提取结构化数据")

```

## 直接使用预测器（高级）

虽然推荐使用流水线，但如果您需要对过程进行更精细的控制（例如，仅使用检测，或提供预处理输入），您可以直接使用单个预测器。有关初始化和使用每个预测器/处理器对的详细信息，请参阅**预测器**部分文档。

## 性能注意事项

*   **设备选择：** 使用支持 CUDA 的 GPU（`Device('cuda:0')`）比 CPU（`Device('cpu')`）显著加快推理速度。确保您安装了必要的驱动程序和 ONNX Runtime GPU 构建。
*   **模型选择：** 流水线 YAML 文件中配置的特定 ONNX 模型会影响性能和准确性。
*   **批处理：** 虽然当前的流水线示例处理单个图像，但预测器通常在内部处理批输入（例如，在识别中同时处理所有检测到的框）。对于处理大量图像，如果需要，考虑在应用程序级别进行并行执行或批处理。
