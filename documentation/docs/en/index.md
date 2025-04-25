# Welcome to MyOCR Documentation

<div align="center">
    <img width="150" alt="myocr logo" src="assets/images/logomain.png">
</div>

**MyOCR is a Python library designed to streamline the development and deployment of production-ready OCR systems.**

Whether you need basic text extraction or complex structured data extraction from documents, MyOCR provides the tools to build robust and efficient pipelines.

## Key Features

*   **🚀 End-to-End & Modular:** Build complete OCR workflows (Detection, Classification, Recognition, etc.) by combining modular components.
*   **🛠️ Extensible:** Easily integrate custom models or processing logic.
*   **⚡ Production Ready:** Optimized for speed with ONNX Runtime support for CPU and GPU inference.
*   **📊 Structured Output:** Extract information into predefined JSON formats using LLM integration.
*   **🔌 Multiple Usage Modes:** Use as a Python library, deploy as a REST API service, or run in Docker.

## Getting Started

1.  **[Installation](./getting-started/installation.md):** Set up MyOCR and download necessary models.
2.  **[Overview](./getting-started/overview.md):** Understand the core concepts (Models, Predictors, Pipelines).
3.  **[Inference Guide](./inference/local.md):** Learn how to run OCR tasks using the library.

## Core Concepts Deep Dive

*   **[Models](./models/model-list.md):** Learn about the supported model types (ONNX, PyTorch, Custom) and architectures.
*   **[Predictors](./predictors/predictor-list.md):** Understand how models are wrapped with pre/post-processing for specific tasks.
*   **[Pipelines](./pipelines/pipelines-list.md):** Explore the high-level pipelines that orchestrate predictors for end-to-end OCR.

## Deployment Options

Beyond using MyOCR as a Python library (see [Inference Guide](./inference/local.md)), you can also deploy it:

*   **As a REST API:**
    *   Start the built-in Flask server: `python main.py` (runs on port 5000 by default).
    *   Endpoints: `GET /ping`, `POST /ocr` (basic OCR), `POST /ocr-json` (structured OCR).
    *   A separate UI is available: [doc-insight-ui](https://github.com/robbyzhaox/doc-insight-ui)
*   **Using Docker:**
    *   Build CPU/GPU images using `Dockerfile-infer-CPU` or `Dockerfile-infer-GPU`.
    *   Use the helper script: `scripts/build_docker_image.sh` for easy setup.
    *   Example run: `docker run -d -p 8000:8000 myocr:gpu` (exposes service on port 8000).

## Additional Resources

*   **[FAQ](./faq.md):** Find answers to common questions.
*   **[Changelog](./CHANGELOG.md):** See recent updates and changes.
*   **[Contributing Guidelines](./CONTRIBUTING.md):** Learn how to contribute to the project.
*   **[GitHub Repository](https://github.com/robbyzhaox/myocr):** Source code, issues, and discussions.

## License

MyOCR is open-sourced under the [Apache 2.0 License](https://github.com/robbyzhaox/myocr/blob/main/LICENSE).


