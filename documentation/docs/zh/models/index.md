# 模型

本节提供有关 MyOCR 项目中用于文本检测、识别和方向分类等任务的深度学习模型的详细信息。

## 模型加载与管理

MyOCR 利用 `myocr/modeling/model.py` 中定义的灵活模型加载系统。它支持加载不同格式的模型：

*   **ONNX (`OrtModel`):** 使用 ONNX Runtime (`onnxruntime`) 加载并运行优化后的模型。由于性能优势，这通常是推理的首选格式。
*   **PyTorch (`PyTorchModel`):** 加载标准的 PyTorch 模型，可能利用来自 `torchvision` 等库的预定义架构。
*   **自定义 PyTorch (`CustomModel`):** 加载通过 YAML 配置文件定义的自定义 PyTorch 模型。这些配置使用 `myocr/modeling/` 中定义的组件来指定模型的架构，包括主干网络 (backbones)、颈部 (necks) 和头部 (heads)。

`ModelLoader` 类充当工厂，根据指定的格式 (`onnx`, `pt`, `custom`) 实例化正确的模型类型。

```python
# 示例 (概念性)
from myocr.modeling.model import ModelLoader, Device

# 加载用于 CPU 推理的 ONNX 模型
loader = ModelLoader()
onnx_model = loader.load(model_format='onnx', model_name_path='path/to/your/model.onnx', device=Device('cpu'))

# 加载由 YAML 定义的用于 GPU 推理的自定义模型
custom_model = loader.load(model_format='custom', model_name_path='path/to/your/config.yaml', device=Device('cuda:0'))
```

## 模型架构

`myocr/modeling/` 目录包含了构建自定义模型的基石：

*   **`architectures/`**: 定义连接主干网络、颈部和头部的整体结构（例如 `DBNet`, `CRNN`）。
*   **`backbones/`**: 包含特征提取网络（例如 `ResNet`, `MobileNetV3`）。
*   **`necks/`**: 包括特征融合模块（例如 `FPN` - 特征金字塔网络）。
*   **`heads/`**: 定义负责特定任务的最终层（例如，检测概率图、序列解码）。


## 可用模型

### 文本检测 (DBNet++)

- **DBNet++**: 基于 DBNet 架构的最先进的文本检测模型
  - 输入：RGB 图像
  - 输出：文本区域多边形
  - 特点：
    - 对任意形状文本的高精度
    - 快速推理速度
    - 对各种文本方向的鲁棒性
  - 架构：
    ```python
    Backbone: ResNet
    Neck: FPN
    Head: DBHead
    ```

### 文本识别 (CRNN)

- **CRNN**: 用于文本识别的混合 CNN-RNN 模型
  - 输入：裁剪的文本区域
  - 输出：识别的文本
  - 特点：
    - 支持中文和英文字符
    - 处理可变长度文本
    - 对不同字体和样式的鲁棒性
  - 架构：
    ```python
    Backbone: CNN
    Neck: BiLSTM
    Head: CTC
    ```

### 文本分类模型

- **文本方向分类器**: 确定文本方向
  - 输入：文本区域
  - 输出：方向角度
  - 特点：
    - 0° 和 180° 分类
    - 帮助提高识别准确性

## 模型性能

- 待更新