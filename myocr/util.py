import time

import matplotlib.pyplot as plt
import numpy as np
import PIL
import PIL.Image

# from IPython import display
# from IPython.display import clear_output
from PIL.Image import Image


def setup_plots():
    """初始化绘图区域"""
    # plt.close('all')  # 关闭之前的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 训练损失子图
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    (train_line,) = ax1.plot([], [], "b-", label="Train Loss")
    ax1.grid(True)

    # 验证损失子图
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    (val_line,) = ax2.plot([], [], "r-", label="Val Loss")
    ax2.grid(True)

    plt.tight_layout()
    plt.legend()

    # 在Jupyter中显示初始空图
    # display.display(fig)

    return fig, ax1, ax2, train_line, val_line


def update_plots(fig, ax1, ax2, train_line, val_line, train_losses, val_losses, epoch):
    """更新损失曲线而不清除输出"""
    # 更新训练损失曲线
    train_line.set_data(range(1, epoch + 2), train_losses)
    ax1.set_xlim(0, len(train_losses) + 1)
    ax1.set_ylim(0, max(train_losses) * 1.1)
    ax1.set_title(f"Training Loss (Epoch {epoch+1})")

    # 更新验证损失曲线
    val_line.set_data(range(1, epoch + 2), val_losses)
    ax2.set_xlim(0, len(val_losses) + 1)
    ax2.set_ylim(0, max(val_losses) * 1.1)
    ax2.set_title(f"Validation Loss (Epoch {epoch+1})")

    # 只更新图形而不清除输出
    # clear_output(wait=True)
    # display.display(fig)
    # 添加小的延迟让图形有时间更新
    time.sleep(0.1)


def crop_rectangle(image: Image, box, target_height=32):
    left, top, right, bottom = map(int, (box.left, box.top, box.right, box.bottom))
    width, height = image.size

    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))

    cropped = image.crop((left, top, right, bottom))
    orig_width, orig_height = cropped.size
    aspect_ratio = orig_width / orig_height

    new_width = int(target_height * aspect_ratio)
    resized = cropped.resize((new_width, target_height), PIL.Image.Resampling.BICUBIC)
    return resized


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # prevent overflow
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class LabelTranslator:
    def __init__(self, alphabet):
        self.alphabet = alphabet + "ç"  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def decode(self, t, length, raw=False):
        t = t[:length]
        if raw:
            return "".join([self.alphabet[i - 1] for i in t])
        else:
            char_list = []
            for i in range(length):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.alphabet[t[i] - 1])
            return "".join(char_list)
