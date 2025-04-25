# Overview

Welcome to MyOCR! This library provides a powerful and flexible framework for building and deploying Optical Character Recognition (OCR) pipelines.

## Why MyOCR?

MyOCR is designed with production readiness and developer experience in mind. Key features include:

*   **End-to-End Workflow:** Seamlessly integrates text detection, direction classification, and text recognition.
*   **Modular & Extensible:** Easily swap models, pre/post-processing steps (via Converters), or entire pipelines.
*   **Optimized for Production:** Leverages ONNX Runtime for high-performance CPU and GPU inference.
*   **Structured Data Extraction:** Go beyond raw text with pipelines that extract information into structured formats (like JSON) using LLMs.
*   **Developer-Friendly:** Offers clean Python APIs and pre-built components to get started quickly.

## Core Components

MyOCR is built around several key concepts:

### Components Diagram
![MyOCR Components](../assets/images/components.png)


### Class Diagram
![MyOCR Class](../assets/images/myocr_class_diagram.png)

*   **Model:** Represents a neural network model. MyOCR supports loading ONNX models (`OrtModel`), standard PyTorch models (`PyTorchModel`), and custom PyTorch models defined by YAML configurations (`CustomModel`). Models handle the core computation.
    *   See the [Models Section](../../models/index.md) for more details.
*   **Converter (`ParamConverter`):** Prepares input data for a model and processes the model's raw output into a more usable format. Each predictor uses a specific converter.
    *   See the [Predictors Section](../../predictors/index.md) for converter specifics.
*   **Predictor:** Combines a `Model` and a `ParamConverter` to perform a specific inference task (e.g., text detection). It provides a user-friendly interface, accepting standard inputs (like PIL Images) and returning processed results (like bounding boxes).
    *   See the [Predictors Section](../../predictors/index.md) for available predictors.
*   **Pipeline:** Orchestrates multiple `Predictors` to perform complex, multi-step tasks like end-to-end OCR. Pipelines offer the highest-level interface for most common use cases.
    *   See the [Pipelines Section](../../pipelines/index.md) for available pipelines.

## Customization and Extension

MyOCR's modular design allows for easy customization.

### Adding New Structured Output Schemas

The `StructuredOutputOCRPipeline` uses Pydantic models to define the desired output format. You can easily create your own:

1.  Define your data model using Pydantic:

    ```python
    from pydantic import BaseModel, Field
    from typing import List, Optional # Optional import if needed

    class CustomDataSchema(BaseModel):
        customer_name: Optional[str] = Field(None, description="The name of the customer")
        order_id: str = Field(..., description="The unique order identifier") # Use ... for required fields
        # Add more fields with descriptions...
    ```

2.  Pass your model when creating the pipeline:

    ```python
    from myocr.pipelines import StructuredOutputOCRPipeline
    from myocr.modeling.model import Device
    # from your_module import CustomDataSchema # Import your defined model

    # Assuming CustomDataSchema is defined as above
    pipeline = StructuredOutputOCRPipeline(device=Device("cuda:0"), json_schema=CustomDataSchema)
    ```

### Replacing or Adding New Models (ONNX)

If you have your own ONNX models for detection, classification, or recognition:

1.  **Place Model Files:** Copy your `.onnx` model files to the default model directory (`~/.MyOCR/models/`) or another location.

2.  **Update Configuration:** Modify the relevant pipeline configuration YAML file (e.g., `myocr/pipelines/config/common_ocr_pipeline.yaml`) to point to your new model files. Use paths relative to the main model directory specified in `myocr.config.MODEL_PATH` (which defaults to `~/.MyOCR/models/`).

    ```yaml
    # Example modification in myocr/pipelines/config/common_ocr_pipeline.yaml
    model:
      detection: "your_custom_detection_model.onnx" # Assumes file is in ~/.MyOCR/models/
      cls_direction: "your_custom_cls_model.onnx"
      recognition: "path/relative/to/model_dir/your_rec_model.onnx" # Example if in a subdirectory
    ```

    Refer to the specific pipeline documentation for details on its configuration file.
   