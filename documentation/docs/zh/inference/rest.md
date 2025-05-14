# 通过 REST API 进行推理

MyOCR 提供了内置的 RESTful API 服务，让您能够通过 HTTP 请求执行 OCR 任务。这对于将 MyOCR 集成到网页应用、微服务或其他编程语言中特别有用。

您可以直接运行此 API 服务进行开发，或者使用 Docker 进行生产环境部署。

## 方式一：直接运行（适用于开发环境）

此方法会直接在您的主机上运行 API 服务，通常适合本地开发和测试。

**1. 前提条件：**

*   确保您已完成 [安装步骤](../getting-started/installation.md)，包括安装依赖项和下载模型。
*   确保您位于 `myocr` 项目的根目录。

**2. 启动服务器：**

```bash
# 使用 python 启动服务器（请查看 main.py 了解确切的主机/端口配置）
# 这可能会使用开发服务器（如 Flask 的默认服务器）和默认端口（如 5000）
python main.py 
```

*   服务器使用项目中定义的模型和配置。
*   端口和主机取决于 `main.py` 的配置方式。
*   **注意：** 对于生产环境部署，建议使用 Docker 和 Gunicorn（方式二）。

**3. API 端点（示例使用端口 5000 - 请根据需要调整）：**

*   **`GET /ping`**：检查服务是否正在运行。
    ```bash
    curl http://127.0.0.1:5000/ping
    ```
*   **`POST /ocr`**：对上传的图像执行基本 OCR。
    *   **请求：** 发送一个 `POST` 请求，图像以 base64 编码字符串形式提供。
    ```bash
    curl -X POST \
        -H "Content-Type: application/json" \
        -d '{"image": "BASE64_IMAGE"}'' \
        http://127.0.0.1:5000/ocr
    ```

    *   **响应：** 返回包含识别文本和边界框信息的 JSON 对象（类似于 `CommonOCRPipeline` 的输出）。

*   **`POST /ocr-json`**：执行 OCR 并根据预定义的 schema 提取结构化信息。
    *   **请求：** 发送一个 `POST` 请求，包含图像的 base64 字符串。
        
    ```bash
    curl -X POST \
        -H "Content-Type: application/json" \
        -d '{"image": "BASE64_IMAGE"}'' \
        http://127.0.0.1:5000/ocr-json
    ```

    *   **响应：** 返回一个符合提供 schema 的 JSON 对象，其中包含从图像中提取的数据。

**4. 可选用户界面：**

有一个基于 Next.js 的独立用户界面可用于与这些 API 端点交互：[doc-insight-ui](https://github.com/robbyzhaox/doc-insight-ui)。

## 方式二：使用 Docker 部署（推荐用于生产环境）

Docker 提供了一个容器化环境来运行 API 服务，确保了一致性并利用 Gunicorn 提高性能。

**1. 前提条件：**

*   已安装 [Docker](https://docs.docker.com/get-docker/)。
*   对于 GPU 支持：已安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。
*   在构建镜像前，确保已将模型下载到**主机**上的默认位置 (`~/.MyOCR/models/`)，因为 Docker 构建过程可能会复制这些文件。

**2. 使用辅助脚本构建 Docker 镜像：**

推荐使用提供的脚本来构建镜像，它能自动处理正确的版本标签。

```bash
# 确保脚本具有执行权限
chmod +x scripts/build_docker_image.sh

# 获取应用程序版本
VERSION=$(python -c 'import myocr.version; print(myocr.version.VERSION)')

# 构建所需镜像（将 [cpu|gpu] 替换为 'cpu' 或 'gpu'）
bash scripts/build_docker_image.sh [cpu|gpu]

# 示例：为当前版本构建 CPU 镜像
# bash scripts/build_docker_image.sh cpu 

# 脚本会输出最终的镜像标签（例如 myocr:cpu-0.1.0）
```

**3. 运行 Docker 容器：**

使用构建脚本生成的镜像标签（如 `myocr:cpu-X.Y.Z` 或 `myocr:gpu-X.Y.Z`）。容器内的服务运行在端口 8000 上。

*   **GPU 版本（将 $IMAGE_TAG 替换为实际标签）：**
    ```bash
    # 示例: docker run -d --gpus all -p 8000:8000 --name myocr-service myocr:gpu-0.1.0
    docker run -d --gpus all -p 8000:8000 --name myocr-service $IMAGE_TAG
    ```
*   **CPU 版本（将 $IMAGE_TAG 替换为实际标签）：**
    ```bash
    # 示例: docker run -d -p 8000:8000 --name myocr-service myocr:cpu-0.1.0
    docker run -d -p 8000:8000 --name myocr-service $IMAGE_TAG
    ```
*   `-p 8000:8000` 参数将主机上的端口 8000 映射到容器内的端口 8000。

**4. 访问 API 端点（Docker）：**

容器运行后，使用主机的 IP/主机名（或 `localhost`）和映射的端口（示例中为 8000）访问 API 端点：

```bash
# Ping 测试
curl http://localhost:8000/ping 

# 图像 base64 编码
IMAGE_PATH="your_image.jpg"

BASE64_IMAGE=$(base64 -w 0 "$IMAGE_PATH")  # Linux
#BASE64_IMAGE=$(base64 -i "$IMAGE_PATH" | tr -d '\n') # macOS

# 基本 OCR 示例
curl -X POST \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"${BASE64_IMAGE}\"}" \
  http://localhost:8000/ocr

# 结构化 OCR 示例
curl -X POST \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"${BASE64_IMAGE}\"}" \
  http://localhost:8000/ocr-json
```

注意：请将 `your_image.jpg` 替换为您运行 `curl` 命令的机器上实际图像文件的路径。 