# 训练自定义模型

MyOCR 允许使用其基于 PyTorch 的建模组件 (`myocr.modeling`) 来训练自定义模型。核心思想是在标准的 PyTorch 训练循环中利用从 YAML 配置加载的 `CustomModel` 类。

**免责声明:** 本指南概述了通用方法。项目中包含的 `myocr/training/` 目录可能包含为 MyOCR 量身定制的特定训练脚本、实用程序、损失函数或数据集处理程序。在从头开始编写训练循环之前，强烈建议先探索 `myocr/training/` 的内容，以了解框架特定的实现和辅助工具。

## 1. 准备数据

*   **数据集:** 您需要一个适用于您任务的带标签数据集（例如，用于 OCR 的带有边界框和转录文本的图像）。
*   **PyTorch Dataset 类:** 创建一个自定义的 `torch.utils.data.Dataset` 类来加载您的图像和标签，并执行必要的初始转换。
*   **DataLoader:** 使用 `torch.utils.data.DataLoader` 来创建用于训练和验证的数据批次。

## 2. 配置模型架构 (YAML)

*   在 YAML 配置文件（例如 `config/my_trainable_model.yaml`）中定义您要训练的模型的架构。
*   您可以从头开始训练，也可以为特定组件加载预训练权重（例如，在 YAML 的 `backbone` 部分指定的预训练主干网络）。


## 3. 设置训练循环

*   **加载模型:** 使用 `ModelLoader` 从 YAML 配置加载您的 `CustomModel`。
*   **定义损失:** 为您的任务选择或实现一个合适的损失函数（例如，用于识别的 `torch.nn.CTCLoss`，基于 DBNet 原理的用于检测的自定义损失）。检查 `myocr/modeling/` 或 `myocr/training/` 中可能预定义的损失函数。
*   **定义优化器:** 选择一个 PyTorch 优化器（例如 `torch.optim.Adam`, `SGD`）。
*   **训练设备:** 设置设备（CPU 或 GPU）。

```python
import torch
import torch.optim as optim
from myocr.modeling.model import ModelLoader, Device

# --- 配置 ---
MODEL_CONFIG_PATH = 'config/my_trainable_model.yaml'
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
OUTPUT_DIR = "./trained_models"

# --- 设置 ---
device = Device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 加载自定义模型结构
loader = ModelLoader()
model = loader.load(model_format='custom', model_name_path=MODEL_CONFIG_PATH, device=device)

# 定义损失函数 (CTC 示例)
# criterion = torch.nn.CTCLoss(blank=0).to(device.name) 
# 或查找/实现您的特定损失
criterion = ... 

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 可选: 学习率调度器
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

## 4. 运行训练循环

*   迭代周期 (epochs) 和批次 (batches)。
*   将模型设置为训练模式 (`model.train()`)。
*   执行前向传播，计算损失，执行反向传播，并更新优化器。
*   包含一个使用 `model.eval()` 和 `torch.no_grad()` 的验证循环来监控性能。
*   定期保存模型检查点（例如，根据验证损失保存性能最佳的模型）。

```python
import os

print(f"在 {device.name} 上开始训练...")
trainer = Trainer(model,[], nn.CrossEntropyLoss(), optimizer=Adam(model.parameters(), lr=0.001), num_epochs=50, batch_size = 64)
trainer.fit(train_dataset, val_dataset)

print('训练完成')

# 保存最终模型
final_model_path = os.path.join(OUTPUT_DIR, "final_model.pth")
torch.save(model.loaded_model.state_dict(), final_model_path)
print(f"已将最终模型保存到 {final_model_path}")
```

## 5. 训练后

*   **评估:** 将您保存的权重（`.pth` 文件）加载到 `CustomModel` 中（可能通过在 YAML 配置中将 `pretrained` 键设置为您保存的路径）并运行评估。
*   **ONNX 导出 (可选):** 为了优化推理，您可以使用 `CustomModel` 的 `to_onnx` 方法将训练好的 PyTorch 模型转换为 ONNX 格式。

```python
# 加载训练好的模型（假设 YAML 通过 'pretrained' 键指向保存的 .pth 文件）
# trained_model = loader.load('custom', MODEL_CONFIG_PATH, device)

# --- 或者在加载架构后手动加载 state dict --- 
model_for_export = loader.load('custom', MODEL_CONFIG_PATH, device)
model_for_export.loaded_model.load_state_dict(torch.load(best_model_path, map_location=device.name))
model_for_export.eval()

# 创建一个具有正确形状和类型的虚拟输入
dummy_input = torch.randn(1, 3, 640, 640).to(device.name) # 根据需要调整形状

onnx_output_path = os.path.join(OUTPUT_DIR, "trained_model.onnx")

model_for_export.to_onnx(onnx_output_path, dummy_input)
print(f"已将模型导出到 {onnx_output_path}")
```
*   然后，您可以按照[添加新模型](./add-model.md)中的步骤使用这个导出的 ONNX 模型。 