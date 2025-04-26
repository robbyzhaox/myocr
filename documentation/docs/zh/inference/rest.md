# 通过 REST API 进行推理

MyOCR 提供了一个内置的 RESTful API 服务，允许您通过 HTTP 请求执行 OCR 任务。这对于将 MyOCR 集成到 Web 应用程序、微服务或从不同的编程语言访问它非常有用。

您可以直接运行此 API 服务进行开发，也可以使用 Docker 进行生产部署。

## 方式一：直接运行 (用于开发)

此方法直接在您的主机上运行 API 服务，通常适用于本地开发和测试。

**1. 先决条件：**

*   确保您已完成 [安装步骤](../getting-started/installation.md)，包括安装依赖项和下载模型。
*   确保您位于 `myocr` 项目的根目录中。

**2. 启动服务器：**

```bash
# 使用 python 启动服务器 (请检查 main.py 以了解确切的主机/端口)
# 这可能会使用开发服务器 (例如 Flask 的默认服务器) 和端口 (例如 5000)。
python main.py 
```

*   服务器使用项目中定义的模型和配置。
*   端口和主机取决于 `main.py` 的配置方式。
*   **注意：** 对于生产部署，建议使用 Docker 和 Gunicorn (方式二)。

**3. API 端点 (使用示例端口 5000 - 如果需要请调整)：**

*   **`GET /ping`**：检查服务是否正在运行。
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
    *   **响应：** 返回一个符合提供的schema的JSON对象，其中填充了提取的数据。

**4. 可选 UI：**

有一个单独的基于 Streamlit 的 UI 可用于与这些端点交互：[doc-insight-ui](https://github.com/robbyzhaox/doc-insight-ui)。

## 方式二：使用 Docker 部署 (生产环境推荐)

Docker 提供了一个容器化环境来运行 API 服务，确保了一致性并利用 Gunicorn 提高性能。

**1. 先决条件：**

*   已安装 [Docker](https://docs.docker.com/get-docker/)。
*   对于 GPU 支持：已安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。
*   在构建镜像之前，请确保模型已下载到**主机**上的默认位置 (`~/.MyOCR/models/`)，因为 Docker 构建过程可能会复制它们。

**2. 使用辅助脚本构建 Docker 镜像：**

推荐使用提供的脚本来构建镜像。它会处理带有正确版本的标签。

```bash
# 确保脚本可执行
chmod +x scripts/build_docker_image.sh

# 确定应用程序版本
VERSION=$(python -c 'import myocr.version; print(myocr.version.VERSION)')

# 构建所需的镜像 (将 [cpu|gpu] 替换为 'cpu' 或 'gpu')
bash scripts/build_docker_image.sh [cpu|gpu]

# 示例：为当前版本构建 CPU 镜像
# bash scripts/build_docker_image.sh cpu 

# 脚本将输出最终的镜像标签 (例如 myocr:cpu-0.1.0)
```

**3. 运行 Docker 容器：**

使用构建脚本生成的镜像标签 (例如 `myocr:cpu-X.Y.Z` 或 `myocr:gpu-X.Y.Z`)。容器内的服务运行在端口 8000。

*   **GPU 版本 (将 $IMAGE_TAG 替换为实际标签)：**
    ```bash
    # 示例: docker run -d --gpus all -p 8000:8000 --name myocr-service myocr:gpu-0.1.0
    docker run -d --gpus all -p 8000:8000 --name myocr-service $IMAGE_TAG
    ```
*   **CPU 版本 (将 $IMAGE_TAG 替换为实际标签)：**
    ```bash
    # 示例: docker run -d -p 8000:8000 --name myocr-service myocr:cpu-0.1.0
    docker run -d -p 8000:8000 --name myocr-service $IMAGE_TAG
    ```
*   `-p 8000:8000` 标志将您主机上的端口 8000 映射到容器内的端口 8000。

**4. 访问 API 端点 (Docker)：**

容器运行后，使用主机的 IP/主机名 (或 `localhost`) 和映射的主机端口 (示例中为 8000) 访问 API 端点：

```bash
# Ping 示例
curl http://localhost:8000/ping 

# 基本 OCR 示例
curl -X POST -F "file=@/path/to/your/image.jpg" http://localhost:8000/ocr

# 结构化 OCR 示例 (将 $SCHEMA 替换为实际的 JSON schema 字符串)
SCHEMA='{...}' 
curl -X POST \
  -F "file=@/path/to/your/invoice.png" \
  -F "schema_json=$SCHEMA" \
  http://localhost:8000/ocr-json
```

请记住将 `/path/to/your/image.jpg` 替换为您运行 `curl` 命令的机器上的实际路径。 