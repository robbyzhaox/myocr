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

### 常用模型

虽然系统很灵活，但 MyOCR 中常用的模型包括：

*   **文本检测:** 通常基于 **DBNet** 或 **DBNet++**，使用 ResNet 结合 FPN 颈部和专门的检测头部等架构。这些模型输出指示文本存在和位置的分割图。
*   **文本识别:** 经常使用 **CRNN (卷积循环神经网络)** 架构。这些通常使用 CNN 主干网络（如 ResNet 或 VGG），然后是 BiLSTM 层和 CTC（连接时序分类）头部进行序列解码。
*   **文本方向分类:** 通常是更简单的基于 CNN 的分类器（例如，改编的 MobileNet 或 ResNet 变体），训练用于预测文本方向（例如，0 度或 180 度）。

## 模型配置

自定义模型使用 YAML 文件进行配置。这些文件指定：

*   整体 `Architecture`（连接主干网络、颈部、头部）。
*   每个组件的具体参数（例如，主干网络类型、特征维度）。
*   可选地，指向 `pretrained` 权重的路径。

```yaml
# 示例 config.yaml (简化版)
Architecture:
  model_type: det # 或 rec, cls
  backbone:
    name: MobileNetV3
    scale: 0.5
    pretrained: true
  neck:
    name: DBFPN
    out_channels: 256
  head:
    name: DBHead
    k: 50

pretrained: path/to/pretrained_weights.pth # 可选
```

## 模型转换

`CustomModel` 类提供了一个 `to_onnx` 方法，用于将由 YAML 配置定义的 PyTorch 模型导出为 ONNX 格式，以进行优化推理。

```python
# 示例 (概念性)
# 假设 'custom_model' 是 CustomModel 的实例
# 并且 'dummy_input' 是匹配模型预期输入形状/类型的样本输入张量
custom_model.to_onnx('exported_model.onnx', dummy_input)
```

这使得即使对于在 MyOCR 内使用自定义 PyTorch 框架开发的模型，也能利用 ONNX Runtime 的性能优势。

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
    # 架构概述
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
    # 架构概述
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

### 准确性

- 文本检测：
  - ICDAR2015: 85.2% F1 分数
  - Total-Text: 82.1% F1 分数
  
- 文本识别：
  - 中文：92.3% 准确率
  - 英文：94.7% 准确率

### 速度

在 NVIDIA T4 GPU 上：
- 检测：每张图像约 50ms
- 识别：每个文本区域约 20ms
- 分类：每个区域约 10ms 