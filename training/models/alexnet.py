import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    """
    Implements the AlexNet architecture for image classification.

    AlexNet is a pioneering convolutional neural network (CNN) that significantly impacted the field of deep learning,
    particularly in the area of image recognition and classification. This implementation is adapted to allow for a
    variable number of output classes, making it suitable for different image classification tasks beyond the original
    ImageNet challenge it was designed for.

    Args:
        num_classes (int): Number of classes for the output layer. Defaults to 2.
    
    The network consists of five convolutional layers, some of which are followed by max-pooling layers. After the
    convolutional layers, the network has three fully connected layers. Dropout and ReLU activations are used to
    prevent overfitting and introduce non-linearity, respectively.
    """

    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        # Feature extraction part: Convolutional and max-pooling layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # First conv layer with ReLU and max pooling
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # Second conv layer with ReLU and max pooling
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # Third conv layer with ReLU
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # Fourth conv layer with ReLU
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Fifth conv layer with ReLU and max pooling
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # Adaptive pooling to ensure a fixed size output regardless of the input image size
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # Classifier part: Fully connected layers with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(),  # Helps prevent overfitting
            nn.Linear(256 * 6 * 6, 4096),  # First fully connected layer with ReLU
            nn.ReLU(inplace=True),
            nn.Dropout(),  # Additional dropout layer for regularization
            nn.Linear(4096, 4096),  # Second fully connected layer with ReLU
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # Final fully connected layer for classification
        )

    def forward(self, x):
        x = self.features(x)  # Pass input through the feature extraction layers
        x = self.avgpool(x)  # Apply adaptive average pooling
        x = torch.flatten(x, 1)  # Flatten the output to prepare for the fully connected layers
        x = self.classifier(x)  # Pass through the classifier to get final predictions
        return x
