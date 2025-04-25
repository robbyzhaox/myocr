# Installation

This guide covers the necessary steps to install MyOCR and its dependencies.

## Requirements

*   **Python:** Version 3.11 or higher is required.
*   **CUDA:** Version 12.6 or higher is recommended for GPU acceleration. CPU-only mode is also supported.
*   **Operating System:** Linux, macOS, or Windows.

## Installation Steps

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/robbyzhaox/myocr.git
    cd myocr
    ```

2.  **Install Dependencies:**

    *   **For standard usage:**
        ```bash
        # Installs the package and required dependencies
        pip install -e .
        ```
    *   **For development (including testing, linting, etc.):**
        ```bash
        # Installs standard dependencies plus development tools
        pip install -e ".[dev]"
        ```

3.  **Download Pre-trained Models:**

    MyOCR relies on pre-trained models for its default pipelines. These need to be downloaded manually.

    ```bash
    # Create the default model directory if it doesn't exist
    # On Linux/macOS:
    mkdir -p ~/.MyOCR/models/
    # On Windows (using Git Bash or similar):
    # mkdir -p ~/AppData/Local/MyOCR/models/
    # Note: Adjust the Windows path if needed based on your environment.

    # Download models (ensure curl is installed)
    # Detection Model (DBNet++)
    curl -L "https://drive.google.com/uc?export=download&id=1b5I8Do4ODU9xE_dinDGZMraq4GDgHPH9" -o ~/.MyOCR/models/dbnet++.onnx
    # Recognition Model (CRNN-like)
    curl -L "https://drive.google.com/uc?export=download&id=1MSF7ArwmRjM4anDiMnqhlzj1GE_J7gnX" -o ~/.MyOCR/models/rec.onnx
    # Classification Model (Angle)
    curl -L "https://drive.google.com/uc?export=download&id=1TCu3vAXNVmPBY2KtoEBTGOE6tpma0puX" -o ~/.MyOCR/models/cls.onnx
    ```

    *   **Note:** The default location where MyOCR expects models is `~/.MyOCR/models/`. This path is defined in `myocr/config.py`. You can modify this configuration or place models elsewhere if needed, but you would need to adjust the paths in the pipeline configuration files (`myocr/pipelines/config/*.yaml`).
    *   The `curl` commands above use Google Drive links. Ensure you can download from these links in your environment. You might need to adjust the commands or download the files manually if `curl` has issues with redirects or permissions.

## Next Steps

Once installation is complete and models are downloaded, you can proceed to:

*   [Overview](overview.md): Get a high-level understanding of the library.
*   [Inference Guide](../../inference/index.md): See examples of how to run OCR tasks.