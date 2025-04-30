import torch
import torch.nn as nn
import torch.nn.functional as F


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(1.2 * x + 3.0, inplace=self.inplace) / 6.0


class GELU(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.gelu(x)


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class Activation(nn.Module):
    def __init__(self, act_type, inplace=True):
        super().__init__()
        act_type = act_type.lower()
        if act_type == "relu":
            self.act = nn.ReLU(inplace=inplace)
        elif act_type == "relu6":
            self.act = nn.ReLU6(inplace=inplace)
        elif act_type == "sigmoid":
            raise NotImplementedError
        elif act_type == "hard_sigmoid":
            self.act = Hsigmoid(inplace)
        elif act_type == "hard_swish" or act_type == "hswish":
            self.act = Hswish(inplace=inplace)
        elif act_type == "leakyrelu":
            self.act = nn.LeakyReLU(inplace=inplace)
        elif act_type == "gelu":
            self.act = GELU(inplace=inplace)
        elif act_type == "swish":
            self.act = Swish(inplace=inplace)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        return self.act(inputs)
