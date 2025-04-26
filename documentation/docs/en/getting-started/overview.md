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
    *   See the [Models Section](../models/index.md) for more details.
*   **Converter (`ParamConverter`):** Prepares input data for a model and processes the model's raw output into a more usable format. Each predictor uses a specific converter.
    *   See the [Predictors Section](../predictors/index.md) for converter specifics.
*   **Predictor:** Combines a `Model` and a `ParamConverter` to perform a specific inference task (e.g., text detection). It provides a user-friendly interface, accepting standard inputs (like PIL Images) and returning processed results (like bounding boxes).
    *   See the [Predictors Section](../predictors/index.md) for available predictors.
*   **Pipeline:** Orchestrates multiple `Predictors` to perform complex, multi-step tasks like end-to-end OCR. Pipelines offer the highest-level interface for most common use cases.
    *   See the [Pipelines Section](../pipelines/index.md) for available pipelines.


## Customization and Extension

MyOCR's modular design allows for easy customization.

*   **[Adding New Models](../models/new-model.md):** Learn about the ways to introduce a new model by the model loader.
*   **[Creating Custom Predictors](../predictors/new-predictor.md):** Learn about how to create a custom `Predictor` by `Model` and `ParamConverter`.
*   **[Building Custom Pipelines](../pipelines/build-pipeline.md):** Learn how to Orchestrates multiple `Predictor`'s to `Pipeline`