import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # Adjust input channels to 3
        # Second convolutional layer: 6 input channels, 16 output channels, kernel size of 5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # First fully connected layer: 16*5*5 input features, 120 output features
        self.fc1 = nn.Linear(53*53*16, 120)
        # Second fully connected layer: 120 input features, 84 output features
        self.fc2 = nn.Linear(120, 84)
        # Third (final) fully connected layer: 84 input features, num_classes output features for classification
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Apply the first convolutional layer followed by ReLU activation and max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # Apply the second convolutional layer followed by ReLU activation and max pooling
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)
        # Apply the first fully connected layer followed by ReLU activation
        x = F.relu(self.fc1(x))
        # Apply the second fully connected layer followed by ReLU activation
        x = F.relu(self.fc2(x))
        # Apply the final fully connected layer to produce classification outputs
        x = self.fc3(x)
        return x