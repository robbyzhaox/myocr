# Inference via REST API

MyOCR provides a built-in RESTful API service based on Flask, allowing you to perform OCR tasks via HTTP requests. This is useful for integrating MyOCR into web applications, microservices, or accessing it from different programming languages.

You can run this API service directly or deploy it using Docker.

## Option 1: Running the Flask API Directly

This method runs the API service directly on your host machine.

**1. Prerequisites:**

*   Ensure you have completed the [Installation steps](../getting-started/installation.md), including installing dependencies and downloading models.
*   Make sure you are in the root directory of the `myocr` project.

**2. Start the Server:**

```bash
# Start the Flask development server
# It will typically run on http://127.0.0.1:5000 by default
python main.py 
```

*   The server uses the models and configurations defined within the project.
*   By default, it might use the device (CPU/GPU) configured in the underlying pipeline settings or attempt auto-detection. Check the server logs for details.

**3. API Endpoints:**

*   **`GET /ping`**: Checks if the service is running. Returns a simple confirmation.
    ```bash
    curl http://127.0.0.1:5000/ping
    ```
*   **`POST /ocr`**: Performs basic OCR on an uploaded image.
    *   **Request:** Send a `POST` request with the image file included as `multipart/form-data`. The file part should be named `file`.
    ```bash
    curl -X POST -F "file=@/path/to/your/image.jpg" http://127.0.0.1:5000/ocr 
    ```
    *   **Response:** Returns a JSON object containing the recognized text and bounding box information (similar to the output of `CommonOCRPipeline`).
*   **`POST /ocr-json`**: Performs OCR and extracts structured information based on a schema.
    *   **Request:** Send a `POST` request with the image file (`file`) and the desired JSON schema (`schema_json`) as `multipart/form-data`.
        *   `schema_json`: A JSON string representing the Pydantic model schema (including descriptions for fields).
    ```bash
    # Example using the pre-defined InvoiceModel schema (get schema first if needed)
    # NOTE: Generating the correct schema_json might require a helper script or knowing the exact format expected by the API.
    # This example assumes schema_json contains the JSON representation of InvoiceModel.schema()
    SCHEMA='{...}' # Replace with actual JSON schema string

    curl -X POST \
      -F "file=@/path/to/your/invoice.png" \
      -F "schema_json=$SCHEMA" \
      http://127.0.0.1:5000/ocr-json
    ```
    *   **Response:** Returns a JSON object matching the provided schema, populated with the extracted data.

**4. Optional UI:**

A separate Streamlit-based UI is available for interacting with these endpoints: [doc-insight-ui](https://github.com/robbyzhaox/doc-insight-ui).

## Option 2: Deploying with Docker

Docker provides a containerized environment for running the API service, ensuring consistency across different machines.

**1. Prerequisites:**

*   [Docker](https://docs.docker.com/get-docker/) installed.
*   For GPU support: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.
*   Ensure models are downloaded to the default location (`~/.MyOCR/models/`) on the **host machine** before building the image, as the Docker build process might copy them.

**2. Build the Docker Image:**

Choose the appropriate Dockerfile:

*   **For GPU Inference:**
    ```bash
    docker build -f Dockerfile-infer-GPU -t myocr:gpu .
    ```
*   **For CPU Inference:**
    ```bash
    docker build -f Dockerfile-infer-CPU -t myocr:cpu .
    ```

**3. Run the Docker Container:**

*   **GPU Version:** Expose the container's port (usually 8000 or 5000 internally, check the Dockerfile) to a host port (e.g., 8000). Requires the `--gpus all` flag.
    ```bash
    # Map host port 8000 to container port 8000
    docker run -d --gpus all -p 8000:8000 --name myocr-service myocr:gpu
    ```
*   **CPU Version:**
    ```bash
    # Map host port 8000 to container port 8000
    docker run -d -p 8000:8000 --name myocr-service myocr:cpu
    ```
*   **Note:** The internal port the application listens on inside the container might vary (check `EXPOSE` in the Dockerfile or the `CMD` instruction). The `-p` flag maps `HOST_PORT:CONTAINER_PORT`.

**4. Using the Helper Script (Easier Setup):**

The project includes a script to simplify building and running the GPU version:

```bash
# Make executable (if needed)
chmod +x scripts/build_docker_image.sh

# Run the script (stops old containers, cleans images, builds, runs)
./scripts/build_docker_image.sh
```
This script typically maps port 8000 on the host to the container's service port.

**5. Accessing API Endpoints (Docker):**

Once the container is running, access the API endpoints using the **host machine's IP/hostname** and the **mapped host port** (e.g., 8000 in the examples above):

```bash
# Example Ping (assuming port 8000 is mapped)
curl http://localhost:8000/ping 

# Example Basic OCR (assuming port 8000 is mapped)
curl -X POST -F "file=@/path/to/your/image.jpg" http://localhost:8000/ocr
```

Remember to replace `/path/to/your/image.jpg` with the actual path on the machine where you are running the `curl` command.