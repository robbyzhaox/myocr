import time

import numpy as np
import PIL
import PIL.Image
from PIL.Image import Image


def define_fallback_functions():
    """Define fallback functions when IPython is not available"""

    def clear_output(wait):
        pass

    def display(x):
        print(x)

    return clear_output, display


try:
    import IPython

    if IPython.get_ipython() is not None:
        from IPython import display
        from IPython.display import clear_output
    else:
        clear_output, display = define_fallback_functions()
except ImportError:
    clear_output, display = define_fallback_functions()


def setup_plots():
    import matplotlib.pyplot as plt

    """Initialize plotting area for visualization"""
    # plt.close('all')  # Close previous figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Training loss subplot
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    (train_line,) = ax1.plot([], [], "b-", label="Train Loss")
    ax1.grid(True)

    # Validation loss subplot
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    (val_line,) = ax2.plot([], [], "r-", label="Val Loss")
    ax2.grid(True)

    plt.tight_layout()
    plt.legend()

    # Display initial empty figure in Jupyter
    display.display(fig)

    return fig, ax1, ax2, train_line, val_line


def update_plots(fig, ax1, ax2, train_line, val_line, train_losses, val_losses, epoch):
    """Update loss curves without clearing output"""
    # Update training loss curve
    train_line.set_data(range(1, epoch + 2), train_losses)
    ax1.set_xlim(0, len(train_losses) + 1)
    ax1.set_ylim(0, max(train_losses) * 1.1)
    ax1.set_title(f"Training Loss (Epoch {epoch+1})")

    # Update validation loss curve
    val_line.set_data(range(1, epoch + 2), val_losses)
    ax2.set_xlim(0, len(val_losses) + 1)
    ax2.set_ylim(0, max(val_losses) * 1.1)
    ax2.set_title(f"Validation Loss (Epoch {epoch+1})")

    # Only update figure without clearing output
    clear_output(wait=True)
    display.display(fig)
    # Add small delay to allow figure to update
    time.sleep(0.1)


def crop_rectangle(image: Image, box, target_height=32):
    """Crop a rectangle from an image and resize it to a target height while preserving aspect ratio"""
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
    """Compute softmax values for each set of scores in x"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # prevent overflow
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class LabelTranslator:
    """Translate between numeric indices and character labels"""

    def __init__(self, alphabet):
        self.alphabet = alphabet + "ç"  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def decode(self, t, length, raw=False):
        """Decode a sequence of indices to a string

        Args:
            t: Sequence of indices
            length: Length of the sequence
            raw: Whether to perform CTC decoding (merging repeated characters)

        Returns:
            Decoded string
        """
        t = t[:length]
        if raw:
            return "".join([self.alphabet[i - 1] for i in t])
        else:
            char_list = []
            for i in range(length):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.alphabet[t[i] - 1])
            return "".join(char_list)


def poly_area(points):
    area = 0.0
    for i in range(len(points)):
        j = (i + 1) % len(points)
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def poly_perimeter(points):
    perimeter = 0.0
    for i in range(len(points)):
        j = (i + 1) % len(points)
        dx = points[j][0] - points[i][0]
        dy = points[j][1] - points[i][1]
        perimeter += (dx * dx + dy * dy) ** 0.5
    return perimeter
