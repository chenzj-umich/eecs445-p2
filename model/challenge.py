import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Challenge(nn.Module):
    def __init__(self):
        super().__init__()

        # Increased number of out_channels for each convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
        
        # Fully connected layers with updated input dimensions
        self.fc_1 = nn.Linear(256 * 2 * 2, 128)  # Increased output units in fully connected layer
        self.fc_2 = nn.Linear(128, 2)  # Output layer for binary classification

        # Initialize weights and biases
        self.init_weights()

        # Freeze all layers except the fully connected layers
        for param in self.parameters():
            param.requires_grad = False
        for param in self.fc_1.parameters():
            param.requires_grad = True
        for param in self.fc_2.parameters():
            param.requires_grad = True

    def init_weights(self):
        torch.manual_seed(42)

        for conv in [self.conv1, self.conv2, self.conv3]:
            nn.init.normal_(conv.weight, mean=0.0, std=sqrt(1 / (conv.kernel_size[0] * conv.kernel_size[1] * conv.in_channels)))
            nn.init.constant_(conv.bias, 0.0)

        # Initialize weights for fully connected layers
        nn.init.normal_(self.fc_1.weight, mean=0.0, std=sqrt(1 / self.fc_1.in_features))
        nn.init.constant_(self.fc_1.bias, 0.0)
        
        nn.init.normal_(self.fc_2.weight, mean=0.0, std=sqrt(1 / self.fc_2.in_features))
        nn.init.constant_(self.fc_2.bias, 0.0)

    def forward(self, x):
        N, C, H, W = x.shape

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        
        x = F.relu(self.fc_1(x))  # First fully connected layer
        x = self.fc_2(x)  # Output layer

        return x
