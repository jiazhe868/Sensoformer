import torch
import torch.nn as nn
from torch import Tensor

class ResidualBlock1D(nn.Module):
    """
    A standard 1D ResNet block.
    Structure: Conv1D -> BN -> ReLU -> Conv1D -> BN -> (+ Shortcut) -> ReLU
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=kernel_size//2, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, 
            padding=kernel_size//2, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out