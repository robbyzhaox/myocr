# 模型

本节提供有关 MyOCR 项目中用于文本检测、识别和方向分类等任务的深度学习模型的详细信息。

## 模型加载与管理

MyOCR 利用 `myocr/modeling/model.py` 中定义的灵活模型加载系统。它支持加载不同格式的模型：

*   **ONNX (`OrtModel`):** 使用 ONNX Runtime (`onnxruntime`) 加载并运行优化后的模型。由于性能优势，这通常是推理的首选格式，也是默认流水线使用的主要格式。
*   **PyTorch (`PyTorchModel`):** 加载标准的 PyTorch 模型，可能利用来自 `torchvision` 等库的预定义架构。
*   **自定义 PyTorch (`CustomModel`):** 加载通过 YAML 配置文件定义的自定义 PyTorch 模型。这些配置使用 `myocr/modeling/` 中定义的组件来指定模型的架构，包括主干网络 (backbones)、颈部 (necks) 和头部 (heads)。

`ModelLoader` 类充当工厂，根据指定的格式 (`onnx`, `pt`, `custom`) 实例化正确的模型类型。

```python
# 示例 (概念性 - 流水线通常会自动处理)
from myocr.modeling.model import ModelLoader, Device

# 加载用于 CPU 推理的 ONNX 模型
loader = ModelLoader()
onnx_model = loader.load(model_format='onnx', model_name_path='path/to/your/model.onnx', device=Device('cpu'))

# 加载由 YAML 定义的用于 GPU 推理的自定义模型
custom_model = loader.load(model_format='custom', model_name_path='path/to/your/config.yaml', device=Device('cuda:0'))
```

**注意:** 虽然您可以手动加载模型，但 `Pipeline` 类通常会根据其各自的配置文件（例如 `myocr/pipelines/config/common_ocr_pipeline.yaml`）自动处理所需模型的加载。

## 模型架构

`myocr/modeling/` 目录包含了构建自定义 PyTorch 模型的基石：

*   **`architectures/`**: 定义连接主干网络、颈部和头部的整体结构（例如 `DBNet`, `CRNN`）。
*   **`backbones/`**: 包含特征提取网络（例如 `ResNet`, `MobileNetV3`）。
*   **`necks/`**: 包括特征融合模块（例如 `FPN` - 特征金字塔网络）。
*   **`heads/`**: 定义负责特定任务的最终层（例如，检测概率图、序列解码）。

### 默认模型 (ONNX)

默认流水线依赖于安装过程中下载到 `~/.MyOCR/models/` 的预训练 ONNX 模型：

*   **文本检测 (`dbnet++.onnx`):** 基于 **DBNet++** 架构。输出指示文本存在和位置的分割图。
*   **文本识别 (`rec.onnx`):** 可能采用 **CRNN (卷积循环神经网络)** 架构或类似的序列到序列模型。使用 CTC 解码处理可变长度文本。
*   **文本方向分类 (`cls.onnx`):** 一个轻量级的**基于 CNN 的分类器**，用于预测文本方向（0 度或 180 度）。

## 配置 (自定义模型)

自定义 PyTorch 模型使用 YAML 文件进行配置。这些文件指定：

*   整体 `Architecture`（连接主干网络、颈部、头部）。
*   每个组件的具体参数（例如，主干网络类型、特征维度）。
*   可选地，指向 `pretrained` 权重的路径。

```yaml
# 示例 config.yaml (说明性 - 结构可能有所不同)
Architecture:
  model_type: det # 或 rec, cls
  backbone:
    name: MobileNetV3
    scale: 0.5
    pretrained: true # 如果适用，指向主干网络权重的路径
  neck:
    name: DBFPN # 示例特征金字塔网络
    out_channels: 256
  head:
    name: DBHead # 示例检测头
    k: 50

pretrained: path/to/full_model_weights.pth # 可选：加载整个组合模型的权重
```

## 模型转换

`CustomModel` 类提供了一个 `to_onnx` 方法，用于将由 YAML 配置定义的 PyTorch 模型导出为 ONNX 格式，以进行优化推理。

```python
# 示例 (概念性)
# 假设 'custom_model' 是从 YAML 配置加载的 CustomModel 实例
# 并且 'dummy_input' 是匹配模型预期输入形状/类型的样本输入张量
custom_model.to_onnx('exported_model.onnx', dummy_input)
```

这使得即使对于在 MyOCR 内使用自定义 PyTorch 框架开发的模型，也能利用 ONNX Runtime 的性能优势。 