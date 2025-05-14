# Training Custom Models

MyOCR allows training custom models defined using its PyTorch-based modeling components (`myocr.modeling`). The core idea is to leverage the `CustomModel` class loaded from a YAML configuration within a standard PyTorch training loop.

**Disclaimer:** This guide outlines the general approach. The project includes a `myocr/training/` directory which might contain specific training scripts, utilities, loss functions, or dataset handlers tailored for MyOCR. It is highly recommended to explore the contents of `myocr/training/` for framework-specific implementations and helpers before writing a training loop from scratch.

## 1. Prepare Your Data

*   **Dataset:** You'll need a labeled dataset suitable for your task (e.g., images with bounding boxes and transcriptions for OCR).
*   **PyTorch Dataset Class:** Create a custom `torch.utils.data.Dataset` class to load your images and labels, and perform necessary initial transformations.
*   **DataLoader:** Use `torch.utils.data.DataLoader` to create batches of data for training and validation.

## 2. Configure Your Model Architecture (YAML)

*   Define the architecture of the model you want to train in a YAML configuration file (e.g., `config/my_trainable_model.yaml`).
*   You might start training from scratch or load pre-trained weights for specific components (e.g., a pre-trained backbone specified within the `backbone` section of the YAML).


## 3. Set Up the Training Loop

*   **Load Model:** Use `ModelLoader` to load your `CustomModel` from the YAML configuration.
*   **Define Loss:** Choose or implement a suitable loss function for your task (e.g., `torch.nn.CTCLoss` for recognition, custom loss for detection based on DBNet principles). Check `myocr/modeling/` or `myocr/training/` for potentially pre-defined losses.
*   **Define Optimizer:** Select a PyTorch optimizer (e.g., `torch.optim.Adam`, `SGD`).
*   **Training Device:** Set the device (CPU or GPU).

```python
import torch
import torch.optim as optim
from myocr.modeling.model import ModelLoader, Device

# --- Configuration ---
MODEL_CONFIG_PATH = 'config/my_trainable_model.yaml'
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
OUTPUT_DIR = "./trained_models"

# --- Setup ---
device = Device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the custom model structure
loader = ModelLoader()
model = loader.load(model_format='custom', model_name_path=MODEL_CONFIG_PATH, device=device)

# Define Loss Function (Example for CTC)
# criterion = torch.nn.CTCLoss(blank=0).to(device.name) 
# Or find/implement your specific loss
criterion = ... 

# Define Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Optional: Learning rate scheduler
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

## 4. Run the Training Loop

*   Iterate through epochs and batches.
*   Set model to training mode (`model.train()`).
*   Perform forward pass, calculate loss, perform backpropagation, and update optimizer.
*   Include a validation loop using `model.eval()` and `torch.no_grad()` to monitor performance.
*   Save model checkpoints periodically (e.g., save the best performing model based on validation loss).

```python
import os

print(f"Starting training on {device.name}...")
trainer = Trainer(model,[], nn.CrossEntropyLoss(), optimizer=Adam(model.parameters(), lr=0.001), num_epochs=50, batch_size = 64)
trainer.fit(train_dataset, val_dataset)

print('Finished Training')

# Save the final model
final_model_path = os.path.join(OUTPUT_DIR, "final_model.pth")
torch.save(model.loaded_model.state_dict(), final_model_path)
print(f"Saved final model to {final_model_path}")
```

## 5. After Training

*   **Evaluation:** Load your saved weights (`.pth` file) into the `CustomModel` (potentially by setting the `pretrained` key in the YAML config to your saved path) and run evaluation.
*   **ONNX Export (Optional):** For optimized inference, you can convert your trained PyTorch model to ONNX format using the `to_onnx` method of the `CustomModel`.

```python
# Load the trained model (assuming YAML points to the saved .pth via 'pretrained' key)
# trained_model = loader.load('custom', MODEL_CONFIG_PATH, device)

# --- Or load state dict manually after loading architecture --- 
model_for_export = loader.load('custom', MODEL_CONFIG_PATH, device)
model_for_export.loaded_model.load_state_dict(torch.load(best_model_path, map_location=device.name))
model_for_export.eval()

# Create a dummy input with the correct shape and type
dummy_input = torch.randn(1, 3, 640, 640).to(device.name) # Adjust shape as needed

onnx_output_path = os.path.join(OUTPUT_DIR, "trained_model.onnx")

model_for_export.to_onnx(onnx_output_path, dummy_input)
print(f"Exported model to {onnx_output_path}")
```
*   You can then use this exported ONNX model following the steps in [Adding New Models](./add-model.md#option-1-adding-a-pre-trained-onnx-model).
