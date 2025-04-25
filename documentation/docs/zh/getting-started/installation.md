# 安装指南

本指南涵盖了安装 MyOCR 及其依赖项所需的步骤。

## 系统要求

*   **Python:** 需要 3.11 或更高版本。
*   **CUDA:** 推荐使用 12.6 或更高版本以进行 GPU 加速。也支持仅 CPU 模式。
*   **操作系统:** Linux、macOS 或 Windows。

## 安装步骤

1.  **克隆仓库:**

    ```bash
    git clone https://github.com/robbyzhaox/myocr.git
    cd myocr
    ```

2.  **安装依赖:**

    *   **标准用法:**
        ```bash
        # 安装软件包和必需的依赖项
        pip install -e .
        ```
    *   **用于开发 (包括测试、代码检查等):**
        ```bash
        # 安装标准依赖项外加开发工具
        pip install -e ".[dev]"
        ```

3.  **下载预训练模型:**

    MyOCR 的默认流水线依赖于预训练模型。这些模型需要手动下载。

    ```bash
    # 如果默认模型目录不存在，则创建它
    # 在 Linux/macOS 上:
    mkdir -p ~/.MyOCR/models/
    # 在 Windows 上 (使用 Git Bash 或类似工具):
    # mkdir -p ~/AppData/Local/MyOCR/models/
    # 注意: 如果需要，请根据您的环境调整 Windows 路径。

    # 下载模型 (确保已安装 curl)
    # 检测模型 (DBNet++)
    curl -L "https://drive.google.com/uc?export=download&id=1b5I8Do4ODU9xE_dinDGZMraq4GDgHPH9" -o ~/.MyOCR/models/dbnet++.onnx
    # 识别模型 (类 CRNN)
    curl -L "https://drive.google.com/uc?export=download&id=1MSF7ArwmRjM4anDiMnqhlzj1GE_J7gnX" -o ~/.MyOCR/models/rec.onnx
    # 分类模型 (角度)
    curl -L "https://drive.google.com/uc?export=download&id=1TCu3vAXNVmPBY2KtoEBTGOE6tpma0puX" -o ~/.MyOCR/models/cls.onnx
    ```

    *   **注意:** MyOCR 默认查找模型的位置是 `~/.MyOCR/models/`。此路径在 `myocr/config.py` 中定义。如果需要，您可以修改此配置或将模型放置在其他地方，但您需要调整流水线配置文件 (`myocr/pipelines/config/*.yaml`) 中的路径。
    *   上面的 `curl` 命令使用了 Google Drive 链接。请确保您的环境可以从这些链接下载。如果 `curl` 处理重定向或权限时出现问题，您可能需要调整命令或手动下载文件。

## 后续步骤

安装完成并下载模型后，您可以继续：

*   [概览](overview.md): 对库有一个高层次的理解。
*   [推理指南](../../inference/index.md): 查看如何运行 OCR 任务的示例。
