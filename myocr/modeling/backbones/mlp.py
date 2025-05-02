import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_channels=28 * 28, hidden_size=128, output_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.out_channels = output_size

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.reshape(x.size(0), -1)
        return self.layers(x)
