# 为 MyOCR 做出贡献

感谢您有兴趣为 MyOCR 做出贡献！本文档提供了为该项目贡献的指南和说明。

## 行为准则

参与本项目即表示您同意为每个人维护一个尊重和包容的环境。请在沟通中保持友善、体贴和建设性。

## 开始贡献

1.  **Fork 仓库**: 在 GitHub 上创建您自己的仓库分支 (Fork)。
2.  **克隆您的分支**: 
    ```bash
    git clone https://github.com/your-username/myocr.git
    cd myocr
    ```
3.  **添加上游仓库**: 
    ```bash
    git remote add upstream https://github.com/robbyzhaox/myocr.git
    ```
4.  **创建分支**: 为您的工作创建一个新分支。
    ```bash
    git checkout -b feature/your-feature-name
    ```

## 开发环境设置

1.  **安装依赖**: 
    ```bash
    pip install -e ".[dev]"
    ```
    这将在开发模式下安装软件包及其所有开发依赖项。

2.  **设置 pre-commit 钩子** (可选但推荐): 
    ```bash
    pre-commit install
    ```

## Pull Request (PR) 流程

1.  **保持变更集中**: 每个 PR 应针对一个特定的功能、错误修复或改进。
2.  **更新文档**: 确保更新文档以反映您的更改。
3.  **编写测试**: 为您所做的更改添加或更新测试。
4.  **在本地运行测试**: 在提交 PR 之前，请确保所有测试都通过。
    ```bash
    pytest
    ```
5.  **提交 PR**: 将您的更改推送到您的分支，并针对主仓库创建一个 PR。
    ```bash
    git push origin feature/your-feature-name
    ```
6.  **PR 描述**: 提供清晰的更改描述，并引用任何相关的问题 (Issue)。
7.  **代码审查**: 积极响应代码审查意见，并进行必要的调整。

##编码规范

我们使用多种工具来强制执行编码规范。确保您的代码符合这些标准的最简单方法是使用提供的 Makefile 命令：

### 使用 Makefile

```bash
# 格式化所有代码 (运行 isort, black, 和 ruff fix)
make run-format

# 运行代码质量检查 (isort, black, ruff, mypy, pytest)
make run-checks
```

### 单独的工具

如果您喜欢单独运行这些工具：

1.  **Black**: 用于代码格式化
    ```bash
    black .
    ```

2.  **isort**: 用于导入排序
    ```bash
    isort .
    ```

3.  **Ruff**: 用于代码检查 (Linting)
    ```bash
    ruff check .
    ```

4.  **mypy**: 用于类型检查
    ```bash
    mypy myocr
    ```

这些工具的配置位于 `pyproject.toml` 文件中。

## 测试指南

1.  **编写单元测试**: 为新功能和错误修复编写全面的测试。
2.  **测试覆盖率**: 目标是为所有新代码实现高测试覆盖率。
3.  **测试目录结构**: 
    - 将测试放在 `tests/` 目录下
    - 遵循与源代码相同的目录结构

## 文档

良好的文档对项目至关重要：

1.  **文档字符串 (Docstrings)**: 为所有公共类和函数添加文档字符串。
2.  **示例用法**: 在适当的文档字符串中包含示例用法。
3.  **README 更新**: 如果您添加了主要功能或更改了功能，请更新 README。
4.  **API 文档**: 对于重要的补充内容，请考虑更新 API 文档。

### 构建文档

您可以使用以下命令在本地构建文档：

```bash
make docs
```

此命令将生成 HTML 文档并启动本地服务器以供查看。

## 问题报告

在创建新问题 (Issue) 之前：

1.  **检查现有问题**: 确保该问题尚未被报告。
2.  **提供信息**: 包含有关问题的详细信息：
    - 重现步骤
    - 预期行为
    - 实际行为
    - 环境（操作系统、Python 版本等）
    - 日志或错误消息
3.  **使用模板**: 如果可用，请使用仓库中提供的问题模板。

## 添加新功能

在提出新功能时：

1.  **先讨论**: 对于主要功能，请在实施之前先创建一个 Issue 进行讨论。
2.  **模块化方法**: 在设计新功能时，请牢记模块化架构。
3.  **流水线集成**: 确保新组件能与现有流水线结构良好集成。
4.  **模型兼容性**: 如果添加新模型，请确保它们可以使用现有的 ModelZoo 系统加载。

## Docker 开发

我们提供了用于构建 CPU 和 GPU 推理镜像的 Dockerfile，以及一个简化构建过程的实用脚本。

### 使用构建脚本 (推荐)

`scripts/build_docker_image.sh` 脚本可自动构建带有版本的 Docker 镜像：

```bash
# 如果脚本尚未具有执行权限，请添加
chmod +x scripts/build_docker_image.sh

# 确定应用程序版本
VERSION=$(python -c 'import myocr.version; print(myocr.version.VERSION)')

# 运行脚本以构建 CPU 或 GPU 镜像
# 将 [cpu|gpu] 替换为您的目标环境
bash scripts/build_docker_image.sh [cpu|gpu]

# 示例：构建 CPU 镜像
# bash scripts/build_docker_image.sh cpu
```

此脚本会：
1. 停止并删除任何基于正在构建的*特定版本标签*（例如 `myocr:cpu-X.Y.Z`）的现有容器。
2. 删除该特定标签的现有 Docker 镜像。
3. 将模型从 `~/.MyOCR/models/` 复制到构建上下文中。
4. 使用相应的 Dockerfile（`Dockerfile-infer-CPU` 或 `Dockerfile-infer-GPU`）构建新的 Docker 镜像。
5. 使用版本号标记镜像（例如 `myocr:cpu-X.Y.Z` 或 `myocr:gpu-X.Y.Z`）。

**注意：** 此脚本仅*构建*镜像。有关运行容器的说明，请参阅 [REST API 推理指南](./inference/rest.md#方式二使用-docker-部署-生产环境推荐)。

### 手动 Docker 构建

如果您倾向于手动构建 Docker 镜像：

```bash
# 首先，将模型复制到构建上下文
mkdir -p models
cp -r ~/.MyOCR/models/* ./models/

# 确定版本
VERSION=$(python -c 'import myocr.version; print(myocr.version.VERSION)')

# 构建 GPU 版本 (示例标签)
docker build -f Dockerfile-infer-GPU -t myocr:gpu-$VERSION .

# 构建 CPU 版本 (示例标签)
docker build -f Dockerfile-infer-CPU -t myocr:cpu-$VERSION .

# 清理复制的模型
rm -rf ./models
```

请记住将 `$VERSION` 替换为实际的版本号。然后，您可以按照推理文档中的说明运行构建好的镜像。

## 许可证

为 MyOCR 做出贡献，即表示您同意您的贡献将根据项目的 Apache 2.0 许可证进行许可。

---

感谢您为 MyOCR 做出贡献！您的努力使这个项目变得更好。 