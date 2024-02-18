import torch
import torch.nn as nn
import torch.nn.functional as F

class CompactCNN(nn.Module):
    """
    Implements a compact convolutional neural network architecture.

    Args:
        num_classes (int): Number of classes for the final output layer. Defaults to 2.

    The network consists of a sequence of convolutional layers followed by batch normalization,
    a depthwise separable convolution layer, and a fully connected layer for classification.
    Max pooling is applied after certain convolutional layers to reduce the spatial dimensions.
    """

    def __init__(self, num_classes=2):
        super(CompactCNN, self).__init__()
        # First convolutional layer with 32 filters
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Batch normalization for the first convolutional layer
        self.bn1 = nn.BatchNorm2d(32)
        # Second convolutional layer with 64 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Batch normalization for the second convolutional layer
        self.bn2 = nn.BatchNorm2d(64)
        # Depthwise separable convolution: depthwise
        self.sepconv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        # Depthwise separable convolution: pointwise
        self.pointwise1 = nn.Conv2d(64, 128, kernel_size=1)
        # Batch normalization for the depthwise separable convolution
        self.bn3 = nn.BatchNorm2d(128)
        # Pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layer for classification
        self.fc = nn.Linear(128 * 56 * 56, num_classes)

    def forward(self, x):
        # Apply first convolutional layer with ReLU activation and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        # Apply second convolutional layer with pooling, ReLU activation, and batch normalization
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Apply depthwise separable convolution with ReLU activation and batch normalization
        x = F.relu(self.bn3(self.pointwise1(self.sepconv1(x))))
        # Apply pooling to reduce spatial dimensions
        x = self.pool(x)
        # Flatten the tensor to prepare it for the fully connected layer
        x = torch.flatten(x, 1)
        # Pass the flattened tensor through the fully connected layer
        x = self.fc(x)
        return x

