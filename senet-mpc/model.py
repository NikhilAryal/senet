# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Placeholder mask function for future ReLU pruning (currently does nothing)
def mask(x):
    # In future, this could zero-out or remove neurons, but now it's identity.
    return x

class Net(nn.Module):
    """
    Simple Convolutional Neural Network for CIFAR-10 classification.
    This network has:
    - 2 convolutional layers
    - 2 fully connected layers
    - ReLU activations
    - Max pooling
    """

    def __init__(self):
        super(Net, self).__init__()

        # -------------------------------
        # 1. Convolutional layers
        # -------------------------------
        # Conv1: input channels=3 (RGB), output channels=16, kernel size=3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

        # Conv2: input channels=16, output channels=32, kernel size=3x3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Max pooling layer (2x2) - reduces spatial dimensions by factor of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # -------------------------------
        # 2. Fully connected layers
        # -------------------------------
        # After 2 pooling layers, input image 32x32 becomes 8x8
        # Conv2 outputs 32 channels, so flattened size = 32 * 8 * 8
        self.fc1 = nn.Linear(in_features=32*8*8, out_features=128)

        # Output layer: 10 classes for CIFAR-10
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x: input tensor of shape (batch_size, 3, 32, 32)
        Returns:
            logits: output tensor of shape (batch_size, 10)
        """

        # -------------------------------
        # 1. Convolution + ReLU + Pooling
        # -------------------------------
        x = F.relu(self.conv1(x))   # Apply ReLU to conv1 output
        x = self.pool(x)            # Max pool to reduce spatial size

        x = F.relu(self.conv2(x))   # Apply ReLU to conv2 output
        x = self.pool(x)            # Max pool again

        # -------------------------------
        # 2. Flatten for fully connected layers
        # -------------------------------
        x = x.view(x.size(0), -1)   # Flatten to (batch_size, 32*8*8)

        # -------------------------------
        # 3. Fully connected layers
        # -------------------------------
        x = F.relu(self.fc1(x))     # Apply ReLU after first FC layer
        x = self.fc2(x)             # Output layer (logits for 10 classes)

        return x
