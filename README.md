<div align="center">
    <h1 align="center">MyOCR - Advanced OCR Pipeline Builder</h1>
    <img width="200" alt="myocr logo" src="https://raw.githubusercontent.com/robbyzhaox/myocr/refs/heads/main/documentation/docs/assets/images/logomain.png">

[![Docs](https://img.shields.io/badge/Docs-online-brightgreen)](https://robbyzhaox.github.io/myocr/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-model-yellow?logo=huggingface&logoColor=white&labelColor=ffcc00)](https://huggingface.co/spaces/robbyzhaox/myocr)
[![Docker](https://img.shields.io/docker/pulls/robbyzhaox/myocr?logo=docker&label=Docker%20Pulls)](https://hub.docker.com/r/robbyzhaox/myocr)
[![PyPI](https://img.shields.io/pypi/v/myocr-kit?logo=pypi&label=Pypi)](https://pypi.org/project/myocr-kit/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

English | [简体中文](./README_zh.md)
</div>

MyOCR is a highly extensible and customizable framework for building OCR systems. Engineers can easily train, integrate deep learning models into custom OCR pipelines for real-world applications.

Try the online demo on 
[HuggingFace](https://huggingface.co/spaces/robbyzhaox/myocr) or [ModelScope](https://modelscope.cn/studios/robbyzhao/myocr/summary)

## **🌟 Key Features**:

**⚡️ End-to-End OCR Development Framework** – Designed for developers to build and integrate detection, recognition, and custom OCR models in a unified and flexible pipeline.

**🛠️ Modular & Extensible** – Mix and match components - swap models, predictors, or input output processors with minimal changes.

**🔌 Developer-Friendly by Design** - Clean Python APIs, prebuilt pipelines and processors, and straightforward customization for training and inference.

**🚀 Production-Ready Performance** – ONNX runtime support for fast CPU/GPU inference, support various ways of deployment.

## 📣 Updates
- **🔥2025.05.17 MyOCR v0.1.1 released**


## 🛠️ Installation

### 📦 Requirements
- Python 3.11+
- CUDA: Version 12.6 or higher is recommended for GPU acceleration. CPU-only mode is also supported.
- Operating System: Linux, macOS, or Windows.

### 📥  Install Dependencies

```bash
# Clone the code from GitHub
git clone https://github.com/robbyzhaox/myocr.git
cd myocr

# You can create your own venv before the following steps
# Install dependencies
pip install -e .

# Development environment installation
pip install -e ".[dev]"

# Download pre-trained model weights to models
# for Linux, macOS
mkdir -p ~/.MyOCR/models/
# for Windows, the "models" directory can be created in the current path
Download weights from: https://drive.google.com/drive/folders/1RXppgx4XA_pBX9Ll4HFgWyhECh5JtHnY
# Alternative download link: https://pan.baidu.com/s/122p9zqepWfbEmZPKqkzGBA?pwd=yq6j
```

## 🚀 Quick Start

### 🖥️ Local Inference

#### Basic OCR Recognition

```python
from myocr.pipelines import CommonOCRPipeline

# Initialize common OCR pipeline (using GPU)
pipeline = CommonOCRPipeline("cuda:0")  # Use "cpu" for CPU mode

# Perform OCR recognition on an image
result = pipeline("path/to/your/image.jpg")
print(result)
```

#### Structured OCR Output (Example: Invoice Information Extraction)

config chat_bot in myocr.pipelines.config.structured_output_pipeline.yaml
```yaml
chat_bot:
  model: qwen2.5:14b
  base_url: http://127.0.0.1:11434/v1
  api_key: 'key'
```
**Note:** chat bot currently support:
- Ollama API
- OpenAI API


```python
from pydantic import BaseModel, Field
from myocr.pipelines import StructuredOutputOCRPipeline

# Define output data model, refer to InvoiceModel in main.py

# Initialize structured OCR pipeline
pipeline = StructuredOutputOCRPipeline("cuda:0", InvoiceModel)

# Process image and get structured data
result = pipeline("path/to/invoice.jpg")
print(result.to_dict())
```

### 🐳 Docker Deployment

The framework provides support for Docker deployment, which can be built and run using the following commands:

#### Run the Docker Container

```bash
docker run -d -p 8000:8000 robbyzhaox/myocr:latest

# set the environment variables like following with -e option of docker run if you want use the StructuredOutputOCRPipline
docker run -d \
  -p 8000:8000 \
  -e CHAT_BOT_MODEL="qwen2.5:14b" \
  -e CHAT_BOT_BASEURL="http://127.0.0.1:11434/v1" \
  -e CHAT_BOT_APIKEY="key" \
  robbyzhaox/myocr:latest
```

#### Accessing API Endpoints (Docker)

```bash
IMAGE_PATH="your_image.jpg"

BASE64_IMAGE=$(base64 -w 0 "$IMAGE_PATH")  # Linux
#BASE64_IMAGE=$(base64 -i "$IMAGE_PATH" | tr -d '\n') # macOS

curl -X POST \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"${BASE64_IMAGE}\"}" \
  http://localhost:8000/ocr

```

### 🔗 Using Rest API

The framework provides a simple Flask API service that can be called via HTTP interface:

```bash
# Start the service default port: 5000
python main.py 
```

API endpoints:
- `GET /ping`: Check if the service is running properly
- `POST /ocr`: Basic OCR recognition
- `POST /ocr-json`: Structured OCR output

We also have a UI for these endpoints, please refer to [doc-insight-ui](https://github.com/robbyzhaox/doc-insight-ui)


## 🎖 Contribution Guidelines

We welcome any form of contribution, including but not limited to:

- Submitting bug reports
- Adding new features
- Improving documentation
- Optimizing performance

## 📄 License

This project is open-sourced under the Apache 2.0 License, see the [LICENSE](LICENSE) file for details.
