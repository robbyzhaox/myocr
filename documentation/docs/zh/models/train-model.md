# 训练自定义模型

MyOCR 允许使用其基于 PyTorch 的建模组件 (`myocr.modeling`) 来训练自定义模型。核心思想是在标准的 PyTorch 训练循环中利用从 YAML 配置加载的 `CustomModel` 类。

**免责声明:** 本指南概述了通用方法。项目中包含的 `myocr/training/` 目录可能包含为 MyOCR 量身定制的特定训练脚本、实用程序、损失函数或数据集处理程序。在从头开始编写训练循环之前，强烈建议先探索 `myocr/training/` 的内容，以了解框架特定的实现和辅助工具。

## 1. 准备数据

*   **数据集:** 您需要一个适用于您任务的带标签数据集（例如，用于 OCR 的带有边界框和转录文本的图像）。
*   **PyTorch Dataset 类:** 创建一个自定义的 `torch.utils.data.Dataset` 类来加载您的图像和标签，并执行必要的初始转换。
*   **DataLoader:** 使用 `torch.utils.data.DataLoader` 来创建用于训练和验证的数据批次。

```python
# 概念性示例：自定义数据集
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class YourOCRDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image) # 应用数据增强/预处理
            
        # 以您的模型/损失函数期望的格式返回图像和标签
        return image, label 

# --- 创建数据集和数据加载器 ---
train_transform = ... # 定义训练转换（增强、张量转换、归一化）
val_transform = ...   # 定义验证转换（张量转换、归一化）

train_dataset = YourOCRDataset(train_image_paths, train_labels, transform=train_transform)
val_dataset = YourOCRDataset(val_image_paths, val_labels, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
```

## 2. 配置模型架构 (YAML)

*   在 YAML 配置文件（例如 `config/my_trainable_model.yaml`）中定义您要训练的模型的架构。
*   指定来自 `myocr.modeling` 或您自定义实现的主干网络、颈部（可选）和头部组件。
*   您可以从头开始训练，也可以为特定组件加载预训练权重（例如，在 YAML 的 `backbone` 部分指定的预训练主干网络）。

```yaml
# 示例: config/my_trainable_model.yaml
Architecture:
  model_type: rec # 或 det, cls
  backbone:
    name: ResNet # 示例
    layers: 34
    pretrained: true # 为主干网络加载 ImageNet 权重
  neck:
    name: FPN # 示例
    out_channels: 256
  head:
    name: CTCHead # 用于识别的示例
    num_classes: 9000 # 您的字符集中的类数 + 空白符

# 可选: 加载整个组合模型的权重 (例如，用于微调)
# pretrained: /path/to/your/full_model_weights.pth 
```

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
best_val_loss = float('inf')

os.makedirs(OUTPUT_DIR, exist_ok=True)

for epoch in range(NUM_EPOCHS):
    # --- 训练阶段 ---
    model.train() # 将模型设置为训练模式
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device.name)
        # --- 根据您的模型和损失要求格式化标签和输入 ---
        # 例如，对于 CTC Loss，标签需要目标长度
        formatted_labels = ... 
        target_lengths = ... 
        input_lengths = ... # CTC 通常需要
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs) # 模型返回用于训练的原始输出
        
        # 计算损失 (根据您的标准调整)
        # loss = criterion(outputs.log_softmax(2), formatted_labels, input_lengths, target_lengths)
        loss = ... # 根据您的具体设置计算损失
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99: # 每 100 个批次打印一次进度
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] 训练损失: {running_loss / 100:.4f}')
            running_loss = 0.0

    # --- 验证阶段 ---
    model.eval() # 将模型设置为评估模式
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device.name)
            # --- 格式化标签和输入 ---
            formatted_labels = ...
            target_lengths = ...
            input_lengths = ...
            
            outputs = model(inputs)
            # loss = criterion(outputs.log_softmax(2), formatted_labels, input_lengths, target_lengths)
            loss = ... # 计算验证损失
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    print(f'Epoch {epoch + 1} - 验证损失: {avg_val_loss:.4f}')

    # --- 保存最佳模型 ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_path = os.path.join(OUTPUT_DIR, f"best_model_epoch_{epoch+1}.pth")
        # 仅保存模型 state_dict
        torch.save(model.loaded_model.state_dict(), best_model_path) 
        print(f"已将新的最佳模型保存到 {best_model_path}")

    # 可选: 更新学习率调度器
    # if scheduler: scheduler.step()

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
# 加载训练好的模型 (假设 YAML 通过 'pretrained' 键指向保存的 .pth 文件)
# trained_model = loader.load('custom', MODEL_CONFIG_PATH, device)

# --- 或者在加载架构后手动加载 state dict ---
model_for_export = loader.load('custom', MODEL_CONFIG_PATH, device)
model_for_export.loaded_model.load_state_dict(torch.load(best_model_path, map_location=device.name))
model_for_export.eval()

# 创建具有正确形状和类型的虚拟输入
dummy_input = torch.randn(1, 3, 640, 640).to(device.name) # 根据需要调整形状

onnx_output_path = os.path.join(OUTPUT_DIR, "trained_model.onnx")

model_for_export.to_onnx(onnx_output_path, dummy_input)
print(f"已将模型导出到 {onnx_output_path}")
```
*   然后，您可以按照 [添加新模型](./add-model.md#option-1-adding-a-pre-trained-onnx-model) 中的步骤使用这个导出的 ONNX 模型。 