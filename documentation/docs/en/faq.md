# FAQ - Frequently Asked Questions

### Q: I'm having trouble downloading the models using the `curl` commands.

**A:** The download links point to Google Drive. 
1.  Ensure `curl` is installed and can access external URLs, especially handling redirects (`-L` flag should be used, which is included in the example).
2.  You might encounter permission issues or download quotas with Google Drive. Try accessing the link directly in your browser to download the `.onnx` files (`dbnet++.onnx`, `rec.onnx`, `cls.onnx`).
3.  Once downloaded, manually place these files into the default model directory: `~/.MyOCR/models/` (create the directory if it doesn't exist). Adjust the path for Windows if necessary (e.g., `~/AppData/Local/MyOCR/models/`).

### Q: Where does MyOCR look for models by default?

**A:** The default path is configured in `myocr/config.py` (`MODEL_PATH`) and usually resolves to `~/.MyOCR/models/` on Linux/macOS. Pipeline configuration files (`myocr/pipelines/config/*.yaml`) reference model filenames relative to this directory. You can change `MODEL_PATH` or use absolute paths in the YAML configuration if you store models elsewhere.

### Q: How do I switch between CPU and GPU inference?

**A:** When initializing pipelines or models, pass a `Device` object from `myocr.modeling.model`. 
*   For GPU (assuming CUDA is set up): `Device('cuda:0')` (for the first GPU).
*   For CPU: `Device('cpu')`.
Ensure you have the correct `onnxruntime` package installed (`onnxruntime` for CPU, `onnxruntime-gpu` for GPU) and compatible CUDA drivers for GPU usage.

### Q: The `StructuredOutputOCRPipeline` isn't working or gives errors.

**A:** This pipeline relies on an external Large Language Model (LLM).
1.  **Check Configuration:** Ensure the `myocr/pipelines/config/structured_output_pipeline.yaml` file has the correct `model`, `base_url`, and `api_key` for your chosen LLM provider (e.g., OpenAI, Ollama, a local server).
2.  **API Key:** Make sure the API key is correctly specified (either directly in the YAML or via an environment variable if the YAML points to one, like `OPENAI_API_KEY`).
3.  **Connectivity:** Verify that your environment can reach the `base_url` specified for the LLM API.
4.  **Schema:** Ensure the Pydantic `json_schema` passed during initialization is valid and the descriptions guide the LLM effectively.

### Q: What's the difference between a Predictor and a Pipeline?

**A:**
*   **Predictor:** A lower-level component that wraps a single `Model` with its specific pre-processing (`convert_input`) and post-processing (`convert_output`) logic (defined in a `ParamConverter`). It handles one specific task (e.g., text detection).
*   **Pipeline:** A higher-level component that orchestrates multiple `Predictors` to perform a complete workflow (e.g., end-to-end OCR combining detection, classification, and recognition). Pipelines provide the main user-facing interface for common tasks.

### Q: How can I use my own custom models?

**A:**
*   **ONNX Models:** Place your `.onnx` file in the model directory and update the relevant pipeline configuration YAML file (`myocr/pipelines/config/*.yaml`) to point to your model's filename. See the [Overview](./getting-started/overview.md#replacing-or-adding-new-models-onnx) section.
*   **Custom PyTorch Models:** Define your model architecture using components from `myocr/modeling/` (backbones, necks, heads) and create a YAML configuration file specifying the architecture. Load it using `ModelLoader().load(model_format='custom', ...)` or create a custom pipeline/predictor. See the [Models Documentation](./models/index.md) for details on `CustomModel` and YAML configuration.

### Q: I see errors related to `pyclipper` or `shapely` during detection.

**A:** These are dependencies used for geometric operations (like expanding bounding boxes) in the `TextDetectionParamConverter`. Ensure they were installed correctly along with MyOCR. You might need to install their binary dependencies separately depending on your OS (e.g., `libgeos-dev` on Debian/Ubuntu for Shapely). Check the installation logs or try reinstalling `pip install -e .`.
