from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size=28 * 28, hidden_size=128, output_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)
