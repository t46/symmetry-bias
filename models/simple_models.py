"""
Simple models for the toy experiment.
"""
from torch import nn
import torch


class ForwardModel(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        return x


class BackwardModel(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        return x
