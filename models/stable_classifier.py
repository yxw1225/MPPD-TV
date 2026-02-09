import torch
import torch.nn as nn
from models.layers import *
import torch.nn.functional as F

class StableClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=10, eps=0.):
        super().__init__()
        self.f1 = nn.Linear(input_dim, hidden_dim)
        self.f2 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.f3 = nn.Linear(hidden_dim, output_dim, bias=False)

        self.f4 = nn.Linear(hidden_dim, output_dim)
        self.eps = eps

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, input):
        T = input.shape[0]
        input = input.flatten(0, 1).contiguous()
        x = F.relu(self.bn1(self.f1(input)))
        x = self.f4(x)
        return x
