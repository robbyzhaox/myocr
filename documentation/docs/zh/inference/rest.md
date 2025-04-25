# 通过 REST API 进行推理

MyOCR 提供了一个基于 Flask 的内置 RESTful API 服务，允许您通过 HTTP 请求执行 OCR 任务。这对于将 MyOCR 集成到 Web 应用程序、微服务或从不同的编程语言访问它非常有用。

您可以直接运行此 API 服务，也可以使用 Docker 进行部署。

## 方式一：直接运行 Flask API

此方法直接在您的主机上运行 API 服务。

**1. 先决条件：**

*   确保您已完成 [安装步骤](../getting-started/installation.md)，包括安装依赖项和下载模型。
*   确保您位于 `myocr` 项目的根目录中。

**2. 启动服务器：**

```bash
# 启动 Flask 开发服务器
# 默认情况下，它通常在 http://127.0.0.1:5000 上运行
python main.py 
```

*   服务器使用项目中定义的模型和配置。
*   默认情况下，它可能会使用底层流水线设置中配置的设备 (CPU/GPU)，或尝试自动检测。请检查服务器日志以获取详细信息。

**3. API 端点：**

*   **`GET /ping`**：检查服务是否正在运行。返回简单的确认信息。
    ```bash
    curl http://127.0.0.1:5000/ping
    ```
*   **`POST /ocr`**：对上传的图像执行基本 OCR。
    *   **请求：** 发送一个 `POST` 请求，并将图像文件作为 `multipart/form-data` 包含在内。文件部分应命名为 `file`。
    ```bash
    curl -X POST -F "file=@/path/to/your/image.jpg" http://127.0.0.1:5000/ocr 
    ```
    *   **响应：** 返回一个 JSON 对象，其中包含识别的文本和边界框信息（类似于 `CommonOCRPipeline` 的输出）。
*   **`POST /ocr-json`**：执行 OCR 并根据 schema 提取结构化信息。
    *   **请求：** 发送一个 `POST` 请求，其中包含图像文件 (`file`) 和所需的 JSON schema (`schema_json`) 作为 `multipart/form-data`。
        *   `schema_json`：表示 Pydantic 模型 schema 的 JSON 字符串（包括字段描述）。
    ```bash
    # 使用预定义的 InvoiceModel schema 的示例 (如果需要，先获取 schema)
    # 注意：生成正确的 schema_json 可能需要辅助脚本或了解 API 期望的确切格式。
    # 此示例假定 schema_json 包含 InvoiceModel.schema() 的 JSON 表示
    SCHEMA='{...}' # 替换为实际的 JSON schema 字符串

    curl -X POST \
      -F "file=@/path/to/your/invoice.png" \
      -F "schema_json=$SCHEMA" \
      http://127.0.0.1:5000/ocr-json
    ```
    *   **响应：** 返回与提供的 schema 匹配的 JSON 对象，其中填充了提取的数据。

**4. 可选 UI：**

有一个单独的基于 Streamlit 的 UI 可用于与这些端点交互：[doc-insight-ui](https://github.com/robbyzhaox/doc-insight-ui)。

## 方式二：使用 Docker 部署

Docker 提供了一个容器化环境来运行 API 服务，确保了跨不同机器的一致性。

**1. 先决条件：**

*   已安装 [Docker](https://docs.docker.com/get-docker/)。
*   对于 GPU 支持：已安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。
*   在构建镜像之前，请确保模型已下载到**主机**上的默认位置 (`~/.MyOCR/models/`)，因为 Docker 构建过程可能会复制它们。

**2. 构建 Docker 镜像：**

选择合适的 Dockerfile：

*   **用于 GPU 推理：**
    ```bash
    docker build -f Dockerfile-infer-GPU -t myocr:gpu .
    ```
*   **用于 CPU 推理：**
    ```bash
    docker build -f Dockerfile-infer-CPU -t myocr:cpu .
    ```

**3. 运行 Docker 容器：**

*   **GPU 版本：** 将容器的端口（通常是 8000 或 5000，请检查 Dockerfile）暴露给主机端口（例如 8000）。需要 `--gpus all` 标志。
    ```bash
    # 将主机端口 8000 映射到容器端口 8000
    docker run -d --gpus all -p 8000:8000 --name myocr-service myocr:gpu
    ```
*   **CPU 版本：**
    ```bash
    # 将主机端口 8000 映射到容器端口 8000
    docker run -d -p 8000:8000 --name myocr-service myocr:cpu
    ```
*   **注意：** 应用程序在容器内部监听的端口可能会有所不同（检查 Dockerfile 中的 `EXPOSE` 或 `CMD` 指令）。`-p` 标志映射 `主机端口:容器端口`。

**4. 使用辅助脚本（更简单的设置）：**

项目包含一个脚本来简化构建和运行 GPU 版本的过程：

```bash
# 如果需要，使其可执行
chmod +x scripts/build_docker_image.sh

# 运行脚本 (停止旧容器、清理镜像、构建、运行)
./scripts/build_docker_image.sh
```
此脚本通常将主机上的端口 8000 映射到容器的服务端口。

**5. 访问 API 端点 (Docker)：**

容器运行后，使用**主机的 IP/主机名**和**映射的主机端口**（例如，上面示例中的 8000）访问 API 端点：

```bash
# Ping 示例 (假设映射了端口 8000)
curl http://localhost:8000/ping 

# 基本 OCR 示例 (假设映射了端口 8000)
curl -X POST -F "file=@/path/to/your/image.jpg" http://localhost:8000/ocr
```

请记住将 `/path/to/your/image.jpg` 替换为您运行 `curl` 命令的机器上的实际路径。 