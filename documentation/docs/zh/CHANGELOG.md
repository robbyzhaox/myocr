# 更新日志

本项目所有重要的变更都将记录在此文件中。

该格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)，
并且本项目遵循 [语义化版本](https://semver.org/spec/v2.0.0.html)。

## 未发布

### 添加

- 引入布局检测
- 引入表格检测
- 测试 OCR 精度

## [v0.1.0-beta](https://github.com/robbyzhaox/myocr/releases/tag/v0.1.0-beta) - 2025-05-12

### 添加

- 统一 OCR 结果的数据结构

### 更改

- 重构 CommonOCRPipeline 以使用新的 OCRResult 类型
- 优化 CommonOCRPipeline 和 HTTP 端点的代码

## [v0.1.0-alpha.4](https://github.com/robbyzhaox/myocr/releases/tag/v0.1.0-alpha.4) - 2025-05-08

### 修复
- 修复识别文本的置信度

## [v0.1.0-alpha.3](https://github.com/robbyzhaox/myocr/releases/tag/v0.1.0-alpha.3) - 2025-05-07

### 修复
- 修复构建 Docker 镜像的工作流

## [v0.1.0-alpha.2](https://github.com/robbyzhaox/myocr/releases/tag/v0.1.0-alpha.2) - 2025-05-07

### 添加

- 添加发布 Docker 镜像的工作流
- 更新 readme
- 添加手动发布文档的配置

### 修复
- 修复黑色空格的字符解码

## [v0.1.0-alpha.1](https://github.com/robbyzhaox/myocr/releases/tag/v0.1.0-alpha.1) - 2025-05-06

### 添加
- 在 releash.sh 中添加版本检查
- 在 readme 中添加演示 URL

### 修复

- 通过将日志配置移出 myocr 包来修复日志问题

## [v0.1.0-alpha](https://github.com/robbyzhaox/myocr/releases/tag/v0.1.0-alpha) - 2025-05-04 