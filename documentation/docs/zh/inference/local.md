# 推理指南

本节演示如何使用 MyOCR 库执行推理，主要侧重于高级流水线接口。

## 使用流水线进行推理

执行端到端 OCR 的推荐方法是通过 `myocr.pipelines` 中提供的流水线类。这些流水线处理模型加载以及检测、分类（可选）和识别步骤的编排。

### 使用 `CommonOCRPipeline` 进行标准 OCR

对于需要从图像中提取所有文本及其位置的通用 OCR 任务，请使用此流水线。

```python
import logging
from myocr.pipelines import CommonOCRPipeline
from myocr.modeling.model import Device
from PIL import Image

# 配置日志记录 (可选，但有助于调试)
logging.basicConfig(level=logging.INFO)

# --- 1. 初始化 ---
# 指定设备 ('cuda:0' 代表 GPU 0, 'cpu' 代表 CPU)
try:
    device = Device('cuda:0')
    # 初始化流水线
    # 这会根据 myocr/pipelines/config/common_ocr_pipeline.yaml 
    # 从 ~/.MyOCR/models/ 加载默认模型
    ocr_pipeline = CommonOCRPipeline(device=device)
    logging.info(f"CommonOCRPipeline 在 {device.name} 上成功初始化。")
except Exception as e:
    logging.error(f"初始化流水线时出错: {e}。请确保已下载模型。")
    exit()

# --- 2. 处理 ---
image_path = 'path/to/your/document.jpg' # 或 .png, .tif 等

try:
    logging.info(f"正在处理图像: {image_path}")
    # process 方法处理所有步骤：图像加载、检测、分类、识别
    results = ocr_pipeline.process(image_path) # 返回 RecognizedTexts 对象或 None
except FileNotFoundError:
    logging.error(f"未找到图像文件: {image_path}")
    exit()
except Exception as e:
    logging.error(f"OCR 处理期间出错: {e}")
    exit()

# --- 3. 处理结果 ---
if results:
    logging.info("OCR 处理完成。")
    # 获取合并的文本内容
    full_text = results.get_content_text()
    print("--- 完整识别文本 ---")
    print(full_text)
    print("-----------------------------")

    # 访问单个文本框及其属性
    print("--- 单个文本框 ---")
    for text_item in results.text_items:
        box = text_item.bounding_box
        print(f"  文本: \"{text_item.text}\"")
        print(f"  置信度: {text_item.confidence:.4f}")
        print(f"  框坐标 (L,T,R,B): ({box.left}, {box.top}, {box.right}, {box.bottom})")
        if hasattr(box, 'angle') and box.angle:
            print(f"  角度: {box.angle[0]} 度, 置信度: {box.angle[1]:.4f}")
        print(f"  检测分数: {box.score:.4f}")
        print("---")
else:
    logging.warning(f"在图像 {image_path} 中未检测到文本。")

```

### 使用 `StructuredOutputOCRPipeline` 进行结构化数据提取

当您需要从文档中提取特定信息并根据预定义的 Pydantic schema 将其格式化为结构化格式（如 JSON）时，请使用此流水线。

```python
import logging
import os
from myocr.pipelines import StructuredOutputOCRPipeline
from myocr.modeling.model import Device
from myocr.pipelines.response_format import InvoiceModel # 使用预定义的 schema
# from pydantic import BaseModel, Field # 仅在定义自定义 schema 时需要

# 配置日志
logging.basicConfig(level=logging.INFO)

# --- 1. 选择 Schema ---
# 本例中我们将使用预定义的 InvoiceModel。
# 如果需要，请在此处定义您的自定义 Pydantic 模型。
# class MyCustomSchema(BaseModel):
#    field_1: str = Field(description="...")

# --- 2. 初始化 ---
device = Device('cuda:0')

# 确保 LLM 配置正确 (API 密钥, 基础 URL)
# 检查 myocr/pipelines/config/structured_output_pipeline.yaml 和环境变量
if "OPENAI_API_KEY" not in os.environ:
    logging.warning("未设置 OPENAI_API_KEY 环境变量。LLM 提取可能会失败或使用默认值。")

try:
    # 使用设备和选定的 schema (InvoiceModel) 初始化流水线
    structured_pipeline = StructuredOutputOCRPipeline(device=device, json_schema=InvoiceModel)
    logging.info(f"StructuredOutputOCRPipeline 在 {device.name} 上成功初始化。")
except Exception as e:
    logging.error(f"初始化结构化流水线时出错: {e}")
    exit()

# --- 3. 处理 ---
image_path = 'path/to/your/invoice.png'

try:
    logging.info(f"正在处理图像以进行结构化提取: {image_path}")
    # 此过程涉及 OCR，然后是基于 LLM 的提取
    extracted_data = structured_pipeline.process(image_path) # 返回 InvoiceModel 实例或 None
except FileNotFoundError:
    logging.error(f"未找到图像文件: {image_path}")
    exit()
except Exception as e:
    logging.error(f"结构化 OCR 处理期间出错: {e}")
    exit()

# --- 4. 处理结果 ---
if extracted_data:
    logging.info("结构化数据提取完成。")
    # 结果是一个 Pydantic 模型实例 (本例中为 InvoiceModel)
    print("--- 提取的信息 (JSON) ---")
    print(extracted_data.model_dump_json(indent=2))
    print("-------------------------------------")

    # 使用属性访问直接访问字段
    print(f"提取的发票号码: {extracted_data.invoiceNumber}")
    print(f"提取的总金额: {extracted_data.totalAmount}")
else:
    logging.warning(f"无法从图像 {image_path} 中提取结构化数据。")

```

## 直接使用预测器 (高级)

虽然流水线提供了便利，但您可以直接使用单个预测器以获得更精细的控制（例如，仅使用检测、自定义预处理）。有关初始化和使用预测器/转换器对的详细信息，请参阅 **[预测器文档](../predictors/index.md)**。

## 替代推理方法

除了在 Python 脚本中将 MyOCR 作为库使用外，您还可以通过以下方式运行推理：

*   **REST API:** 启动内置的 Flask 服务器 (`python main.py`) 并向 `/ocr` 或 `/ocr-json` 等端点发送请求。有关详细信息，请参阅主 [文档主页](../index.md#deployment-options)。
*   **Docker:** 使用提供的 Dockerfile（`Dockerfile-infer-CPU`、`Dockerfile-infer-GPU`）或辅助脚本（`scripts/build_docker_image.sh`）构建并运行服务的容器化版本。有关详细信息，请参阅主 [文档主页](../index.md#deployment-options)。

## 性能注意事项

*   **设备:** GPU (`Device('cuda:0')`) 比 CPU (`Device('cpu')`) 快得多。确保安装了 CUDA 驱动程序和 ONNX Runtime GPU 包。
*   **模型:** 流水线 YAML 文件中配置的特定 ONNX 模型会影响性能和准确性。默认模型提供了良好的平衡。
*   **批处理:** 对于处理大量图像，如果性能至关重要，请考虑在应用程序级别实现批处理，尽管底层的预测器在某种程度上内部处理批处理（例如，在识别中同时处理多个检测到的框）。 