# Pipelines

MyOCR pipelines orchestrate multiple components (predictors, models) to perform end-to-end OCR tasks. They provide a high-level interface for processing images or documents.

## Available Pipelines

### 1. `CommonOCRPipeline`

Defined in `myocr/pipelines/common_ocr_pipeline.py`.

This pipeline performs standard OCR: text detection, optional text direction classification, and text recognition.

**Initialization:**

```python
from myocr.pipelines import CommonOCRPipeline
from myocr.modeling.model import Device

# Initialize pipeline for GPU (or 'cpu')
pipeline = CommonOCRPipeline(device=Device('cuda:0'))
```

**Configuration:**

The pipeline loads configuration from `myocr/pipelines/config/common_ocr_pipeline.yaml`. This file specifies the paths to the ONNX models used for detection, classification, and recognition relative to the `MODEL_PATH` defined in `myocr.config`.

```yaml
# Example: myocr/pipelines/config/common_ocr_pipeline.yaml
model:
  detection: "dbnet++.onnx"
  cls_direction: "cls.onnx"
  recognition: "rec.onnx"
```

**Processing:**

The `__call__` method takes the path to an image file.

```python
image_path = 'path/to/your/image.png'
ocr_results = pipeline(image_path)

if ocr_results:
    # Access recognized text and bounding boxes
    print(ocr_results)
```

**Workflow:**

1.  Loads the image.
2.  Uses `TextDetectionPredictor` to find text regions.
3.  Uses `TextDirectionPredictor` to classify the orientation of detected regions.
4.  Uses `TextRecognitionPredictor` to recognize the text within each oriented region.
5.  Returns a result object containing bounding boxes, text, and potentially confidence scores (details depend on the `Predictor` implementation).

### 2. `StructuredOutputOCRPipeline`

Defined in `myocr/pipelines/structured_output_pipeline.py`.

This pipeline extends `CommonOCRPipeline` by adding a step to extract structured information (e.g., JSON) from the recognized text using a large language model (LLM) via the `OpenAiChatExtractor`.

**Initialization:**

Requires a device and a Pydantic model defining the desired JSON output schema.

```python
from myocr.pipelines import StructuredOutputOCRPipeline
from myocr.modeling.model import Device
from pydantic import BaseModel, Field

# Define your desired output structure
class InvoiceData(BaseModel):
    invoice_number: str = Field(description="The invoice number")
    total_amount: float = Field(description="The total amount due")
    due_date: str = Field(description="The payment due date")

# Initialize pipeline
pipeline = StructuredOutputOCRPipeline(device=Device('cuda:0'), json_schema=InvoiceData)
```

**Configuration:**

This pipeline loads its specific configuration from `myocr/pipelines/config/structured_output_pipeline.yaml`, which includes settings for the `OpenAiChatExtractor` (LLM model name, API base URL, API key).

```yaml
# Example: myocr/pipelines/config/structured_output_pipeline.yaml
chat_bot:
  model: "gpt-4o"
  base_url: "https://api.openai.com/v1"
  api_key: "YOUR_API_KEY"
```

**Processing:**

The `__call__` method takes an image path.

```python
image_path = 'path/to/your/invoice.pdf'
structured_data = pipeline(image_path)

if structured_data:
    print(structured_data)
```

**Workflow:**

1.  Performs standard OCR using the inherited `CommonOCRPipeline` to get the raw recognized text.
2.  If text is found, it passes the text content to the `OpenAiChatExtractor`.
3.  The extractor interacts with the configured LLM, providing the text and the desired `json_schema` (Pydantic model) as instructions.
4.  The LLM attempts to extract the relevant information and format it according to the schema.
5.  Returns an instance of the provided Pydantic model populated with the extracted data.

## Customization

Pipelines can be customized by:

*   Modifying the `.yaml` configuration files to use different models.
*   Creating new pipeline classes that inherit from `Pipeline` or existing pipelines.
*   Integrating different predictors or extractors.

## Performance Optimization

### Batch Processing

```python
# Process multiple images
results = [pipeline(img_path) for img_path in image_paths]
```

### GPU Acceleration

```python
# Use GPU for faster processing
pipeline = CommonOCRPipeline("cuda:0")
```

### Memory Management

```python
# Clear GPU memory
import torch
torch.cuda.empty_cache()
```

## Error Handling

Pipelines handle various error cases:

- Invalid image format
- Missing model files
- GPU out of memory
- Invalid configuration

See the [Troubleshooting Guide](../faq.md) for common issues and solutions. 