# Local Inference

This section describes how to perform inference using the MyOCR library, primarily by utilizing the pre-defined pipelines.

## Using Pipelines for Inference

The recommended way to perform end-to-end OCR is through the pipeline classes provided in `myocr.pipelines`. These pipelines handle the loading of necessary models and the orchestration of detection, classification, and recognition steps.

### Standard OCR with `CommonOCRPipeline`

This pipeline is suitable for general OCR tasks where the goal is to extract all text and its location from an image.

```python
from myocr.pipelines import CommonOCRPipeline

# Initialize common OCR pipeline (using GPU)
pipeline = CommonOCRPipeline("cuda:0")  # Use "cpu" for CPU mode

# Perform OCR recognition on an image
result = pipeline("path/to/your/image.jpg")
print(result)
```

### Structured Data Extraction with `StructuredOutputOCRPipeline`

This pipeline is used when you need to extract specific information from a document and format it as JSON, based on a predefined schema.

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


## Direct Predictor Usage (Advanced)

While pipelines are recommended, you can use individual predictors directly if you need more granular control over the process (e.g., using only detection, or providing pre-processed inputs). Refer to the **Predictors** section documentation for details on initializing and using each predictor/processor pair.

## Performance Considerations

*   **Device Selection:** Using a CUDA-enabled GPU (`Device('cuda:0')`) significantly speeds up inference compared to CPU (`Device('cpu')`). Ensure you have the necessary drivers and ONNX Runtime GPU build installed.
*   **Model Choice:** The specific ONNX models configured in the pipeline YAML files impact performance and accuracy.
*   **Batch Processing:** While the current pipeline examples process single images, predictors often handle batch inputs internally (e.g., processing all detected boxes simultaneously in recognition). For processing many images, consider parallel execution or batching at the application level if needed. 