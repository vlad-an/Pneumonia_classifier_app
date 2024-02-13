import torch
import torch.nn as nn
import torch.nn.functional as F

class CompactCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CompactCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.sepconv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.pointwise1 = nn.Conv2d(64, 128, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 56 * 56, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.pointwise1(self.sepconv1(x))))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

