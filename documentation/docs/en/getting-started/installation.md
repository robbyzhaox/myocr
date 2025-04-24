
### Requirements
- Python 3.11+
- CUDA 12.6+ (Recommended for GPU acceleration, but CPU mode is also supported)

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