# MyOCR - Advanced OCR Custom Builder

## Project Overview

MyOCR is a modular OCR recognition framework designed for secondary development, providing a complete pipeline from text detection and recognition to structured information extraction. The framework features:

- **Modular Design**: Each component can be independently replaced or upgraded
- **High-Performance Model Support**: Based on ONNX format models, supporting both CPU and GPU inference
- **Structured Output**: Ability to convert OCR recognition results into structured data (e.g., invoice information)
- **Easy to Extend**: Provides clear interfaces for developers to extend functionality based on their needs

## Installation Guide

### Requirements
- Python 3.11+
- CUDA (Recommended for GPU acceleration, but CPU mode is also supported)

### Installation Method

```bash
# Clone the code from GitHub
git clone https://github.com/robbyzhaox/myocr.git
cd myocr

# Install dependencies
pip install -e .

# Development environment installation
pip install -e ".[dev]"

# Download pre-trained models
mkdir -p ~/.MyOCR/models/
curl -fsSL "https://drive.google.com/file/d/1b5I8Do4ODU9xE_dinDGZMraq4GDgHPH9/view?usp=drive_link" -o ~/.MyOCR/models/dbnet++.onnx
curl -fsSL "https://drive.google.com/file/d/1MSF7ArwmRjM4anDiMnqhlzj1GE_J7gnX/view?usp=drive_link" -o ~/.MyOCR/models/rec.onnx
curl -fsSL "https://drive.google.com/file/d/1TCu3vAXNVmPBY2KtoEBTGOE6tpma0puX/view?usp=drive_link" -o ~/.MyOCR/models/cls.onnx
```

## Quick Start

### Basic OCR Recognition

```python
from myocr.pipelines.common_ocr_pipeline import CommonOCRPipeline

# Initialize OCR pipeline (using GPU)
pipeline = CommonOCRPipeline("cuda:0")  # Use "cpu" for CPU mode

# Perform OCR recognition on an image
result = pipeline("path/to/your/image.jpg")
print(result)
```

### Structured OCR Output (Example: Invoice Information Extraction)

```python
from pydantic import BaseModel, Field
from myocr.pipelines.structured_output_pipeline import StructuredOutputOCRPipeline

# Define output data model
class InvoiceItem(BaseModel):
    name: str = Field(description="Item name in the invoice")
    price: float = Field(description="Item unit price")
    number: str = Field(description="Item quantity")
    tax: str = Field(description="Item tax amount")

class InvoiceModel(BaseModel):
    invoiceNumber: str = Field(description="Invoice number")
    invoiceDate: str = Field(description="Invoice date")
    invoiceItems: list[InvoiceItem] = Field(description="List of items in the invoice")
    totalAmount: float = Field(description="Total amount of the invoice")
    
    def to_dict(self):
        self.__dict__["invoiceItems"] = [item.__dict__ for item in self.invoiceItems]
        return self.__dict__

# Initialize structured OCR pipeline
pipeline = StructuredOutputOCRPipeline("cuda:0", InvoiceModel)

# Process image and get structured data
result = pipeline("path/to/invoice.jpg")
print(result.to_dict())
```

### Using HTTP API Service

The framework provides a simple Flask API service that can be called via HTTP interface:

```bash
# Start the service
python main.py
```

API endpoints:
- `GET /ping`: Check if the service is running properly
- `POST /ocr`: Basic OCR recognition
- `POST /ocr-json`: Structured OCR output

We also have a UI for these endpoints, please refer to [text](https://github.com/robbyzhaox/doc-insight-ui)

## Architecture Design

The MyOCR framework consists of the following components:

### Core Modules
1. **pipelines**: OCR processing pipelines that encapsulate the complete processing flow
   - `CommonOCRPipeline`: Basic OCR processing
   - `StructuredOutputOCRPipeline`: OCR processing with structured output
   
2. **predictors**: OCR predictor components
   - `TextDetectionPredictor`: Text detection predictor
   - `TextRecognitionPredictor`: Text recognition predictor
   
3. **modeling**: Model loading and management
   - `ModelZoo`: Model management and loading

4. **extractor**: Structured information extraction
   - `OpenAiChatExtractor`: Using LLM for structured information extraction

## Customization and Extension

### Adding New Structured Output Models

1. Define your data model using Pydantic:

```python
from pydantic import BaseModel, Field

class CustomModel(BaseModel):
    field1: str = Field(description="Description of field1")
    field2: int = Field(description="Description of field2")
    # Add more fields...
```

2. Create a new pipeline with your model:

```python
from myocr.pipelines.structured_output_pipeline import StructuredOutputOCRPipeline

pipeline = StructuredOutputOCRPipeline("cuda:0", CustomModel)
```

### Replacing or Adding New Models

1. Place ONNX format model files in the `myocr/models` directory
2. Modify the configuration file to use the new models:

```yaml
model:
  detection: "path/to/your/detection_model.onnx"
  recognition: "path/to/your/recognition_model.onnx"
```

## Docker Deployment

The framework provides support for Docker deployment, which can be built and run using the following commands:

### Automated Build Script

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

### Manual Docker Commands

If you need more control or customization:

```bash
# CPU version
docker build -f Dockerfile-infer-CPU -t myocr:cpu .

# GPU version
docker build -f Dockerfile-infer-GPU -t myocr:gpu .

# Run container
docker run -d -p 5000:5000 myocr:gpu
```

## Contribution Guidelines

We welcome any form of contribution, including but not limited to:

- Submitting bug reports
- Adding new features
- Improving documentation
- Optimizing performance

### Development Utilities

MyOCR includes several Makefile commands to help with development:

```bash
# Format code (runs isort, black, and ruff fix)
make run-format

# Run code quality checks (isort, black, ruff, mypy, pytest)
make run-checks

# Build documentation
make docs
```

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for more detailed contribution guidelines.

## License

This project is open-sourced under the Apache 2.0 License, see the [LICENSE](LICENSE) file for details.
