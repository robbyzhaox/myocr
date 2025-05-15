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

        # Installs dependencies for documentation
        pip install -e ".[docs]"
        ```

3.  **Download Pre-trained Models:**

    MyOCR relies on pre-trained models for its default pipelines. These need to be downloaded manually.

    ```bash
    # Create the default model directory if it doesn't exist
    # On Linux/macOS:
    mkdir -p ~/.MyOCR/models/
    # On Windows (using Git Bash or similar):
    # mkdir -p ~/AppData/Local/MyOCR/models/
    # or just create the models directory in current path
    # Note: Adjust the Windows path if needed based on your environment.

    # Download weights from following links to models directory: 
    # https://drive.google.com/drive/folders/1RXppgx4XA_pBX9Ll4HFgWyhECh5JtHnY
    # or
    # https://pan.baidu.com/s/122p9zqepWfbEmZPKqkzGBA?pwd=yq6j
    ```

    *   **Note:** The default location where MyOCR expects models is `~/.MyOCR/models/`. This path is defined in `myocr/config.py`. You can modify this configuration or place models elsewhere if needed, but you would need to adjust the paths in the pipeline configuration files (`myocr/pipelines/config/*.yaml`).

## Next Steps

Once installation is complete and models are downloaded, you can proceed to:

*   [Inference Guide](../inference/local.md): See examples of how to run OCR tasks.