import torch
import torch.nn as nn

class Lyapunov_Function(nn.Module):
    def __init__(self, type) -> None:
        super().__init__()
        self.type = type

    def forward(self, x):
         if self.type == 'l1':
            return (abs(x)).mean()
