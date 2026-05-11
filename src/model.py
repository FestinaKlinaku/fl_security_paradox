import torch
from torch import nn


class FemnistMLP(nn.Module):
    def __init__(self, num_classes: int = 62) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
