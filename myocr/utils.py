import time

import numpy as np


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


def softmax(x):
    """Compute softmax values for each set of scores in x"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # prevent overflow
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def extract_image_type(base64_data):
    if base64_data.startswith("data:image/"):
        prefix_end = base64_data.find(";base64,")
        if prefix_end != -1:
            return (
                base64_data[len("data:image/") : prefix_end],
                base64_data.split(";base64,")[-1],
            )
    return "png", base64_data
