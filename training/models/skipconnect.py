import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipConnectionCNN(nn.Module):
    """
    Implements a convolutional neural network (CNN) with skip connections, enhancing the ability
    to propagate gradients through deeper network layers and improving model accuracy by
    facilitating feature reuse.

    This architecture introduces skip connections that bypass one or more layers by performing
    identity mapping and adding the output of the identity mapping to the output of layers being skipped.
    It helps to mitigate the vanishing gradient problem in deep networks.

    Args:
        num_classes (int): Number of classes for the output layer, defaulting to 2.

    The network includes several convolutional layers with batch normalization, an adaptive pooling layer
    to ensure a fixed-size output regardless of the input size, and a fully connected layer for classification.
    Skip connections are implemented to support effective training of deeper architectures.
    """
    def __init__(self, num_classes=2):
        super(SkipConnectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Adaptive pooling layer to ensure a fixed size output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer, with the input size now correctly being 64 (64 * 1 * 1)
        self.fc = nn.Linear(64, num_classes)
        
        # Identity mapping for the skip connection
        self.identity = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):

        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Prepare identity for adding
        identity = self.identity(x)

        # Second block with downsample
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Add identity
        x = x+ identity
        
        # Adaptive pooling and flatten
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        # Fully connected layer
        x = self.fc(x)
        return x



