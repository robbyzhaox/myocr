# 安装指南

本指南介绍了安装 MyOCR 及其依赖项的必要步骤。

## 系统要求

*   **Python:** 需要 3.11 或更高版本。
*   **CUDA:** 如果需要 GPU 加速，推荐使用 12.6 或更高版本。也支持仅 CPU 模式。
*   **操作系统:** 支持 Linux、macOS 或 Windows。

## 安装步骤

1.  **克隆代码仓库:**

    ```bash
    git clone https://github.com/robbyzhaox/myocr.git
    cd myocr
    ```

2.  **安装依赖项:**

    *   **标准安装:**
        ```bash
        # 安装软件包及其必需的依赖项
        pip install -e .
        ```
    *   **开发环境安装 (包括测试、代码检查等工具):**
        ```bash
        # 安装标准依赖项及开发工具
        pip install -e ".[dev]"

        # 安装文档相关依赖项
        pip install -e ".[docs]"
        ```

3.  **下载预训练模型:**

    MyOCR 的默认流水线依赖于预训练模型，需要手动下载这些模型。

    ```bash
    # 如果默认模型目录不存在，创建该目录
    # 在 Linux/macOS 系统上:
    mkdir -p ~/.MyOCR/models/
    # 在 Windows 系统上 (使用 Git Bash 或类似工具):
    # mkdir -p ~/AppData/Local/MyOCR/models/
    # 或者直接在当前路径创建models目录
    # 注意: 如需要，请根据您的环境调整 Windows 路径。

    # 从以下链接下载模型到models目录
    # https://drive.google.com/drive/folders/1RXppgx4XA_pBX9Ll4HFgWyhECh5JtHnY
    # 或者
    # https://pan.baidu.com/s/122p9zqepWfbEmZPKqkzGBA?pwd=yq6j
    ```

    *   **注意:** MyOCR 默认从 `~/.MyOCR/models/` 目录查找模型。此路径在 `myocr/config.py` 中定义。如果需要，您可以修改此配置或将模型放在其他位置，但这样您需要相应地调整流水线配置文件 (`myocr/pipelines/config/*.yaml`) 中的路径。

## 后续步骤

完成安装并下载模型后，您可以继续：

*   [推理指南](../inference/local.md): 查看使用 MyOCR 运行 OCR 任务的示例。
