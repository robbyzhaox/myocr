<div align="center">
    <h1 align="center">MyOCR - 高级OCR方案构建框架</h1>
    <img width="200" alt="myocr logo" src="https://raw.githubusercontent.com/robbyzhaox/myocr/refs/heads/main/documentation/docs/assets/images/logomain.png">

[![Docs](https://img.shields.io/badge/Docs-online-brightgreen)](https://robbyzhaox.github.io/myocr/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-model-yellow?logo=huggingface&logoColor=white&labelColor=ffcc00)](https://huggingface.co/spaces/robbyzhaox/myocr)
[![Docker](https://img.shields.io/docker/pulls/robbyzhaox/myocr?logo=docker&label=Docker%20Pulls)](https://hub.docker.com/r/robbyzhaox/myocr)
[![PyPI](https://img.shields.io/pypi/v/myocr-kit?logo=pypi&label=Pypi)](https://pypi.org/project/myocr-kit/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

[English](./README.md) | 简体中文
</div>

MyOCR是一个高度可扩展和定制化的OCR系统构建框架。可以用于轻松训练、整合深度学习模型，构建适用于实际应用场景的自定义OCR方案。

尝试在线演示 [HuggingFace](https://huggingface.co/spaces/robbyzhaox/myocr) 或 [魔搭社区](https://modelscope.cn/studios/robbyzhao/myocr/summary)

## **🌟 核心特性**:

**⚡️ 端到端OCR开发框架** – 专为开发者设计，可在统一灵活的流程中构建和集成检测、识别及自定义OCR模型。

**🛠️ 模块化与可扩展性** – 可混合搭配组件，只需最小改动即可替换模型、预测器或输入输出处理器。

**🔌 对开发者友好** - 简洁的Python API、预定义的模型和组件，以及便捷的训练和推理定制选项。

**🚀 生产级性能** – 支持ONNX运行时以实现快速CPU/GPU推理，支持多种部署方式。

## 📣 更新
- **2025.05.17 MyOCR v0.1.1发布**


## 🛠️ 安装

### 📦 系统要求
- Python 3.11+
- CUDA: 如果需要 GPU 加速，推荐使用 12.6 或更高版本。也支持仅 CPU 模式。
- 操作系统: 支持 Linux、macOS 或 Windows。

### 📥 安装依赖

```bash
# 从GitHub克隆代码
git clone https://github.com/robbyzhaox/myocr.git
cd myocr

# 在开始下面步骤前，可以先创建虚拟环境

# 安装依赖
pip install -e .

# 安装开发环境
pip install -e ".[dev]"

# 下载预训练模型权重到models目录
# Linux, macOS
mkdir -p ~/.MyOCR/models/
# 对于Windows, 可以在当前路径创建models目录
从以下链接下载权重: https://drive.google.com/drive/folders/1RXppgx4XA_pBX9Ll4HFgWyhECh5JtHnY
# 备用下载链接: https://pan.baidu.com/s/122p9zqepWfbEmZPKqkzGBA?pwd=yq6j
```

## 🚀 快速开始

### 🖥️ 本地推理

#### 基础OCR识别

```python
from myocr.pipelines import CommonOCRPipeline

# 初始化通用OCR流水线（使用GPU）
pipeline = CommonOCRPipeline("cuda:0")  # 使用"cpu"进行CPU模式

# 对图像执行OCR识别
result = pipeline("path/to/your/image.jpg")
print(result)
```

#### 结构化OCR输出（示例：发票信息提取）

在myocr.pipelines.config.structured_output_pipeline.yaml中配置chat_bot
```yaml
chat_bot:
  model: qwen2.5:14b
  base_url: http://127.0.0.1:11434/v1
  api_key: 'key'
```
**注意:** chat bot 当前支持:
- Ollama API
- OpenAI API


```python
from pydantic import BaseModel, Field
from myocr.pipelines import StructuredOutputOCRPipeline

# 定义输出数据模型，参考 main.py 中的 InvoiceModel 定义

# 初始化结构化OCR流水线
pipeline = StructuredOutputOCRPipeline("cuda:0", InvoiceModel)

# 处理图像并获取结构化数据
result = pipeline("path/to/invoice.jpg")
print(result.to_dict())
```

### 🐳 Docker部署

该框架提供Docker部署支持，可以使用以下命令构建和运行：

#### 运行Docker容器

```bash
docker run -d -p 8000:8000 robbyzhaox/myocr:latest

# 如果要使用StructuredOutputOCRPipline或者/ocr-json api，请修改环境变量后使用下列命令
docker run -d \
  -p 8000:8000 \
  -e CHAT_BOT_MODEL="qwen2.5:14b" \
  -e CHAT_BOT_BASEURL="http://127.0.0.1:11434/v1" \
  -e CHAT_BOT_APIKEY="key" \
  robbyzhaox/myocr:latest
```

#### 访问API（Docker）

```bash
IMAGE_PATH="your_image.jpg"

BASE64_IMAGE=$(base64 -w 0 "$IMAGE_PATH")  # Linux
#BASE64_IMAGE=$(base64 -i "$IMAGE_PATH" | tr -d '\n') # macOS

curl -X POST \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"${BASE64_IMAGE}\"}" \
  http://localhost:8000/ocr

```

### 🔗 使用REST API

该框架提供了一个简单的Flask API服务，可通过HTTP接口调用：

```bash
# 启动服务，默认端口：5000
python main.py 
```

API端点：
- `GET /ping`：检查服务是否正常运行
- `POST /ocr`：基础OCR识别
- `POST /ocr-json`：结构化OCR输出

我们还为这些端点提供了UI界面，请参考[doc-insight-ui](https://github.com/robbyzhaox/doc-insight-ui)


## 🎖 贡献指南

我们欢迎任何形式的贡献，包括但不限于：

- 提交错误报告
- 添加新功能
- 改进文档
- 优化性能

## 📄 许可证

本项目在Apache 2.0许可证下开源，详情请参阅[LICENSE](LICENSE)文件。 