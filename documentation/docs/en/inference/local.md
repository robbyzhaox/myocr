# Inference

This section describes how to perform inference using the MyOCR library, primarily by utilizing the pre-defined pipelines.

## Using Pipelines for Inference

The recommended way to perform end-to-end OCR is through the pipeline classes provided in `myocr.pipelines`. These pipelines handle the loading of necessary models and the orchestration of detection, classification (optional), and recognition steps.

### Standard OCR with `CommonOCRPipeline`

This pipeline is suitable for general OCR tasks where the goal is to extract all text and its location from an image.

```python
import logging
from myocr.pipelines import CommonOCRPipeline
from myocr.modeling.model import Device
from PIL import Image

# Configure logging (optional)
logging.basicConfig(level=logging.INFO)

# --- 1. Initialization ---
# Specify the device ('cuda:0' for GPU 0, 'cpu' for CPU)
device = Device('cuda:0')

# Initialize the pipeline
# This will load the default models specified in 
# myocr/pipelines/config/common_ocr_pipeline.yaml
try:
    ocr_pipeline = CommonOCRPipeline(device=device)
    logging.info("CommonOCRPipeline initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing pipeline: {e}")
    exit()

# --- 2. Processing ---
image_path = 'path/to/your/document.jpg' # Or .png, .tif, etc.

try:
    logging.info(f"Processing image: {image_path}")
    # The process method handles image loading, detection, classification, and recognition
    results = ocr_pipeline.process(image_path)
except FileNotFoundError:
    logging.error(f"Image file not found: {image_path}")
    exit()
except Exception as e:
    logging.error(f"Error during OCR processing: {e}")
    exit()

# --- 3. Handling Results ---
if results:
    logging.info("OCR processing completed.")
    # Get combined text content
    full_text = results.get_content_text()
    print("--- Full Recognized Text ---")
    print(full_text)
    print("-----------------------------")

    # Access individual text boxes and their properties
    print("--- Individual Boxes ---")
    for text_item in results.text_items:
        box = text_item.bounding_box
        print(f"Text: \"{text_item.text}\"")
        print(f"  Confidence: {text_item.confidence:.4f}")
        # BoundingBox details (left, top, right, bottom, angle, score from detection)
        print(f"  Box Coords (L,T,R,B): ({box.left}, {box.top}, {box.right}, {box.bottom})") 
        print(f"  Angle: {box.angle[0]} degrees, Confidence: {box.angle[1]:.4f}") # Angle is tuple (pred_angle, confidence)
        print(f"  Detection Score: {box.score:.4f}")
        print("---")
else:
    logging.warning(f"No text detected in image: {image_path}")

```

### Structured Data Extraction with `StructuredOutputOCRPipeline`

This pipeline is used when you need to extract specific information from a document and format it as JSON, based on a predefined schema.

```python
import logging
from myocr.pipelines import StructuredOutputOCRPipeline
from myocr.modeling.model import Device
from pydantic import BaseModel, Field
import os

# Configure logging (optional)
logging.basicConfig(level=logging.INFO)

# --- 1. Define Output Schema ---
# Use Pydantic to define the structure of the information you want to extract.
class ReceiptInfo(BaseModel):
    store_name: Optional[str] = Field(None, description="Name of the store or vendor")
    total_amount: Optional[float] = Field(None, description="The final total amount paid")
    transaction_date: Optional[str] = Field(None, description="Date of the transaction (YYYY-MM-DD)")
    items: List[str] = Field([], description="List of purchased items mentioned")

# --- 2. Initialization ---
device = Device('cuda:0')

# Ensure your OpenAI API Key is set (or provide directly)
# Assumes the key is in an environment variable OPENAI_API_KEY
# The pipeline config yaml (structured_output_pipeline.yaml) should point to this key
if "OPENAI_API_KEY" not in os.environ:
  logging.warning("OPENAI_API_KEY environment variable not set. LLM extraction might fail.")
  # You might need to manually set the api_key in the config yaml or modify the pipeline code

try:
    # Initialize pipeline with the device and the desired output schema
    structured_pipeline = StructuredOutputOCRPipeline(device=device, json_schema=ReceiptInfo)
    logging.info("StructuredOutputOCRPipeline initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing structured pipeline: {e}")
    exit()

# --- 3. Processing ---
image_path = 'path/to/your/receipt.png'

try:
    logging.info(f"Processing image for structured extraction: {image_path}")
    # This process involves OCR followed by LLM-based extraction
    extracted_data = structured_pipeline.process(image_path)
except FileNotFoundError:
    logging.error(f"Image file not found: {image_path}")
    exit()
except Exception as e:
    # This could be an OCR error or an error during LLM interaction (e.g., API key issue, network error)
    logging.error(f"Error during structured OCR processing: {e}")
    exit()

# --- 4. Handling Results ---
if extracted_data:
    logging.info("Structured data extraction completed.")
    # The result is a Pydantic model instance
    print("--- Extracted Information (JSON) ---")
    print(extracted_data.model_dump_json(indent=2))
    print("-------------------------------------")

    # You can access fields directly
    # print(f"Store: {extracted_data.store_name}")
    # print(f"Total: {extracted_data.total_amount}")
else:
    # This might happen if OCR found no text or the LLM failed to extract data matching the schema
    logging.warning(f"Could not extract structured data from image: {image_path}")

```

## Direct Predictor Usage (Advanced)

While pipelines are recommended, you can use individual predictors directly if you need more granular control over the process (e.g., using only detection, or providing pre-processed inputs). Refer to the **Predictors** section documentation for details on initializing and using each predictor/converter pair.

## Performance Considerations

*   **Device Selection:** Using a CUDA-enabled GPU (`Device('cuda:0')`) significantly speeds up inference compared to CPU (`Device('cpu')`). Ensure you have the necessary drivers and ONNX Runtime GPU build installed.
*   **Model Choice:** The specific ONNX models configured in the pipeline YAML files impact performance and accuracy.
*   **Batch Processing:** While the current pipeline examples process single images, predictors often handle batch inputs internally (e.g., processing all detected boxes simultaneously in recognition). For processing many images, consider parallel execution or batching at the application level if needed. 