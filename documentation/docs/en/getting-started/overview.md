# Overview

MyOCR provides a powerful and flexible framework for building and deploying your own OCR pipelines. This library is designed with production readiness and developer experience in mind, it offers a high-level component architecture for easy integration and extension.


## Core Components

MyOCR is built around several key concepts:

![MyOCR Components](../assets/images/components.png)

*   **Model:** Represents a neural network model. MyOCR supports loading ONNX models (`OrtModel`), standard PyTorch models (`PyTorchModel`), and custom PyTorch models defined by YAML configurations (`CustomModel`). Models handle the core computation.
    *   See the [Models Section](../models/index.md) for more details.
*   **Processor (`CompositeProcessor`):** Prepares input data for a model and processes the model's raw output into a more usable format. Each predictor uses a specific processor.
    *   See the [Predictors Section](../predictors/index.md) for processor specifics.
*   **Predictor:** Combines a `Model` and a `Processor` to perform a specific inference task (e.g., text detection). It provides a user-friendly interface, accepting standard inputs (like PIL Images) and returning processed results (like bounding boxes).
    *   See the [Predictors Section](../predictors/index.md) for available predictors.
*   **Pipeline:** Orchestrates multiple `Predictors` to perform complex, multi-step tasks like end-to-end OCR. Pipelines offer the highest-level interface for most common use cases.
    *   See the [Pipelines Section](../pipelines/index.md) for available pipelines.


**Class Diagram**
![MyOCR Class](../assets/images/myocr_class_diagram.png)


## Customization and Extension

MyOCR's modular design allows for easy customization:

*   **[Adding New Models](../models/add-model.md):** Learn about the ways to introduce a new model by the model loader.
*   **[Creating Custom Predictors](../predictors/create-predictor.md):** Learn about how to create a custom `Predictor` by `Model` and `CompositeProcessor`.
*   **[Building Custom Pipelines](../pipelines/build-pipeline.md):** Learn how to Orchestrates multiple `Predictor`'s to `Pipeline`