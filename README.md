<div align="center">
    <h1 align="center">MyOCR - Advanced OCR Pipeline Builder</h1>
    <img width="200" alt="myocr logo" src="https://raw.githubusercontent.com/robbyzhaox/myocr/refs/heads/main/documentation/docs/assets/images/logomain.png">

[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen)](https://robbyzhaox.github.io/myocr/)
</div>


MyOCR is a Python package designed to streamline the development of production-ready OCR systems. Engineers can easily train, customize, and deploy deep learning models into high-performance OCR pipelines for real-world applications.

## **🌟 Key Features**:

**⚡️ End-to-End OCR Workflow** – Seamlessly integrate detection, recognition, and various models.

**🛠️ Modular & Extensible**– Mix and match components (swap models, processors, or input output converters).

**🚀 Optimized for Production** – ONNX runtime support for high-speed CPU/GPU inference.

**📊 Smart Structured Outputs** – Convert raw OCR results into organized formats (e.g., invoices, forms).

**🔌 Developer-Centric – Clean** Python APIs, prebuilt pipelines, and easy custom training.

## 📣 Updates
- **🔥2025.04.24 release MyOCR alpha version**:
    - Release image detection, class, recognition models
    - All components can work together


## 🛠️ Installation

### 💻 Requirements
- Python 3.11+
- CUDA 12.6+ (Recommended for GPU acceleration, but CPU mode is also supported)

### ⬇️  Install Dependencies

```bash
# Clone the code from GitHub
git clone https://github.com/robbyzhaox/myocr.git
cd myocr

# Install dependencies
pip install -e .

# Development environment installation
pip install -e ".[dev]"

# Download pre-trained model weights
mkdir -p ~/.MyOCR/models/
curl -fsSL "https://drive.google.com/file/d/1b5I8Do4ODU9xE_dinDGZMraq4GDgHPH9/view?usp=drive_link" -o ~/.MyOCR/models/dbnet++.onnx
curl -fsSL "https://drive.google.com/file/d/1MSF7ArwmRjM4anDiMnqhlzj1GE_J7gnX/view?usp=drive_link" -o ~/.MyOCR/models/rec.onnx
curl -fsSL "https://drive.google.com/file/d/1TCu3vAXNVmPBY2KtoEBTGOE6tpma0puX/view?usp=drive_link" -o ~/.MyOCR/models/cls.onnx
```

## ⏭️  Quick Start

### 📝  Local Ieference

#### 📜 Basic OCR Recognition

```python
from myocr.pipelines import CommonOCRPipeline

# Initialize common OCR pipeline (using GPU)
pipeline = CommonOCRPipeline("cuda:0")  # Use "cpu" for CPU mode

# Perform OCR recognition on an image
result = pipeline("path/to/your/image.jpg")
print(result)
```

#### 📝  Structured OCR Output (Example: Invoice Information Extraction)

config chat_bot in myocr.pipelines.config.structured_output_pipeline.yaml
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

### 🌐 Using Rest API

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


### 🚗 Docker Deployment

The framework provides support for Docker deployment, which can be built and run using the following commands:

#### 📦 Automated Build Script

The easiest way to build and run a Docker container is to use the provided script:

```bash
# Make the script executable
chmod +x scripts/build_docker_image.sh

# Run the script
./scripts/build_docker_image.sh
```

This script will:
- Stop and remove any existing MyOCR containers
- Clean up existing Docker images
- Copy models from your local configuration
- Build a new GPU-enabled Docker image
- Start a container with the service exposed on port 8000

#### 📦 Manual Docker Commands

If you need more control or customization:

```bash
# CPU version
docker build -f Dockerfile-infer-CPU -t myocr:cpu .

# GPU version
docker build -f Dockerfile-infer-GPU -t myocr:gpu .

# Run container
docker run -d -p 8000:8000 myocr:gpu
```

## 🎖 Contribution Guidelines

We welcome any form of contribution, including but not limited to:

- Submitting bug reports
- Adding new features
- Improving documentation
- Optimizing performance

### 📦 Development Utilities

MyOCR includes several Makefile commands to help with development:

```bash
# Format code (runs isort, black, and ruff fix)
make run-format

# Run code quality checks (isort, black, ruff, mypy, pytest)
make run-checks

# Preview documentation in local
cd documentation
mkdocs serve -a 127.0.0.1:8001
```

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for more detailed contribution guidelines.

## 📄 License

This project is open-sourced under the Apache 2.0 License, see the [LICENSE](LICENSE) file for details.
