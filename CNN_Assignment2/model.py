import torch
import torch.nn as nn

class CNN64(nn.Module):
    """
    Input: 3x64x64
    Conv(16, 3x3, s=1, p=1) -> ReLU -> MaxPool(2x2)
    Conv(32, 3x3, s=1, p=1) -> ReLU -> MaxPool(2x2)
    Flatten -> FC 100 -> ReLU -> FC 10
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*16*16, 100), nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))