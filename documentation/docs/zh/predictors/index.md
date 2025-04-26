# 预测器 (Predictors)

预测器负责处理 MyOCR 中特定模型（检测、识别、分类）的推理逻辑。它们通过整合预处理和后处理步骤，弥合了原始模型输出与可用结果之间的差距。

预测器通常与一个 `Model` 对象和一个 `ParamConverter` 相关联。

*   **模型 (Model):** 提供核心的 `forward_internal` 方法（例如，ONNX 会话运行、PyTorch 模型前向传播）。
*   **参数转换器 (ParamConverter):** 处理将输入数据转换为模型期望的格式，并将模型的原始输出转换为结构化的、有意义的格式。

## 基础组件

*   **`myocr.base.Predictor`:** 一个简单的包装器，调用 `ParamConverter` 的输入转换、`Model` 的前向传播以及 `ParamConverter` 的输出转换。
*   **`myocr.base.ParamConverter`:** 定义了 `convert_input` 和 `convert_output` 方法的抽象基类。
*   **`myocr.predictors.base`:** 定义了通用的数据结构，如 `BoundingBox`、`RectBoundingBox`、`DetectedObjects`、`TextItem` 和 `RecognizedTexts`，用作不同转换器的输入和输出。

## 可用的预测器和转换器

预测器是在加载的 `Model` 实例上调用 `.predictor(converter)` 方法时隐式创建的。关键组件是 `ParamConverter` 的实现：

### 1. 文本检测 (`TextDetectionParamConverter`)

*   **文件:** `myocr/predictors/text_detection_predictor.py`
*   **输入:** `PIL.Image`
*   **输出:** `DetectedObjects`（包含原始图像和 `RectBoundingBox` 列表）
*   **关联模型:** 通常是 DBNet/DBNet++ ONNX 模型（例如 `dbnet++.onnx`）。
*   **预处理 (`convert_input`):**
    *   调整图像尺寸，使宽高能被 32 整除。
    *   归一化像素值（减去均值，除以标准差）。
    *   将通道转置为 CHW 格式。
    *   添加批次维度。
*   **后处理 (`convert_output`):**
    *   接收模型的原始概率图。
    *   应用阈值 (0.3) 创建二值图。
    *   在二值图中查找轮廓。
    *   根据长度过滤轮廓。
    *   计算最小面积旋转矩形 (`cv2.minAreaRect`)。
    *   根据最小边长过滤矩形。
    *   根据轮廓内的平均概率计算置信度分数。
    *   根据置信度分数 (>= 0.3) 进行过滤。
    *   扩展边界框多边形（`_unclip` 函数，比例 2.3）。
    *   计算扩展框的最小面积矩形。
    *   再次根据最小边长进行过滤。
    *   将最终框坐标缩放回原始图像尺寸。
    *   创建 `RectBoundingBox` 对象。
    *   按从上到下、从左到右的顺序对框进行排序。
    *   将结果包装在 `DetectedObjects` 容器中。

### 2. 文本方向分类 (`TextDirectionParamConverter`)

*   **文件:** `myocr/predictors/text_direction_predictor.py`
*   **输入:** `DetectedObjects`
*   **输出:** `DetectedObjects`（每个 `RectBoundingBox` 中的 `angle` 属性已更新）
*   **关联模型:** 通常是简单的 CNN 分类器 ONNX 模型（例如 `cls.onnx`）。
*   **预处理 (`convert_input`):**
    *   迭代输入 `DetectedObjects` 中的边界框。
    *   使用 `myocr.util.crop_rectangle` 从原始图像裁剪每个文本区域（目标高度 48）。
    *   将裁剪后的图像存储在 `RectBoundingBox` 对象中 (`set_croped_img`)。
    *   归一化裁剪后的图像像素 (`(/ 255.0 - 0.5) / 0.5`)。
    *   确保为 3 通道（如果是灰度图则扩展维度）。
    *   将批次在水平方向上填充到批次中的最大宽度。
    *   将图像堆叠成批处理张量 (BCHW)。
*   **后处理 (`convert_output`):**
    *   接收模型的原始分类 logits/概率。
    *   查找每个框的最大概率索引 (0 或 1)。
    *   将索引映射到角度 (0 或 180 度)。
    *   计算置信度分数（预测类别的概率）。
    *   更新输入 `DetectedObjects` 中相应 `RectBoundingBox` 的 `.angle` 属性（作为一个 `(角度, 置信度)` 元组）。
    *   返回修改后的 `DetectedObjects`。

### 3. 文本识别 (`TextRecognitionParamConverter`)

*   **文件:** `myocr/predictors/text_recognition_predictor.py`
*   **输入:** `DetectedObjects`（来自文本方向分类的输出）
*   **输出:** `RecognizedTexts`（包含 `TextItem` 列表）
*   **关联模型:** 通常是基于 CRNN 的 ONNX 模型（例如 `rec.onnx`）。
*   **预处理 (`convert_input`):**
    *   检索存储在每个 `RectBoundingBox` 中的预裁剪图像 (`get_croped_img`)。
    *   如果 `angle` 属性指示为 180 度，则旋转图像。
    *   归一化像素值 (`(/ 255.0 - 0.5) / 0.5`)。
    *   确保为 3 通道。
    *   将批次在水平方向上填充到最大宽度。
    *   将图像堆叠成批处理张量 (BCHW)。
*   **后处理 (`convert_output`):**
    *   接收模型的原始序列输出 (时间步数, 批次大小, 类别数)。
    *   转置为 (批次大小, 时间步数, 类别数)。
    *   迭代批次中的每个项目：
        *   应用 Softmax 获取概率。
        *   计算字符置信度（每个时间步的最大概率）。
        *   计算整体文本置信度（字符置信度的平均值）。
        *   执行 CTC 解码：获取字符索引（每个时间步的 argmax），并使用 `myocr.util.LabelTranslator`（使用大型中英文字符集初始化）解码序列，移除空白符和重复项。
        *   创建一个包含解码文本、置信度和原始 `RectBoundingBox` 的 `TextItem`。
    *   将所有 `TextItem` 收集到一个 `RecognizedTexts` 对象中。

## 用法示例 (概念性)

此示例展示了如何*可以*单独使用预测器。实际上，使用 `Pipeline`（如 `CommonOCRPipeline`）通常更简单，因为它处理了这种编排。

```python
from myocr.modeling.model import ModelLoader, Device
from myocr.predictors import TextDetectionParamConverter, TextRecognitionParamConverter, TextDirectionParamConverter
from PIL import Image

# 假设模型已加载 (用实际路径或配置中的名称替换)
# 确保模型已下载到 ~/.MyOCR/models/ 或调整路径
det_model_path = "dbnet++.onnx"
cls_model_path = "cls.onnx"
rec_model_path = "rec.onnx"

device = Device('cuda:0') # 或 Device('cpu')

det_model = ModelLoader().load('onnx', det_model_path, device)
cls_model = ModelLoader().load('onnx', cls_model_path, device)
rec_model = ModelLoader().load('onnx', rec_model_path, device)

# 通过将模型与转换器关联来创建预测器
det_predictor = det_model.predictor(TextDetectionParamConverter(det_model.device))
cls_predictor = cls_model.predictor(TextDirectionParamConverter())
rec_predictor = rec_model.predictor(TextRecognitionParamConverter())

# 加载图像
img = Image.open('path/to/image.png').convert("RGB")

# --- 手动运行预测步骤 ---

# 1. 检测文本区域
detected_objects = det_predictor.predict(img) # 输出: DetectedObjects | None

if detected_objects:
    print(f"检测到 {len(detected_objects.bounding_boxes)} 个框。")

    # 2. 分类文本方向
    # 输出: DetectedObjects (角度已更新) | None
    classified_objects = cls_predictor.predict(detected_objects)

    if classified_objects:
        # 3. 识别文本
        # 输出: RecognizedTexts | None
        recognized_texts = rec_predictor.predict(classified_objects)

        if recognized_texts:
            print("--- 识别的文本 ---")
            print(recognized_texts.get_content_text())
            print("---------------------")
        else:
            print("识别步骤失败。")
    else:
        print("分类步骤失败。")
else:
    print("未检测到文本。") 