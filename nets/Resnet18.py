import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.models as models

BATCH_SIZE = 10

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        # self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(
            nn.Linear(num_ftrs, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.resnet18(x)
        return x