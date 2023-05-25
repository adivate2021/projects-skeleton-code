import torch
import torch.nn as nn


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        #self.flatten = torch.flatten()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(200 * 150 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 5)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #(n, 3, 800, 600)
        x = self.conv1(x)
        x = F.relu(x)
        #(n, 4, 800, 600)
        x = self.pool(x)
        #(n, 4, 400, 300)
        x = self.conv2(x)
        x = F.relu(x)
        #(n, 8, 400, 300)
        x = self.pool(x)
        #(n, 8, 200, 150)
        x = torch.flatten(x, start_dim = 1)
        #(n, 8*200*150)
        x = self.fc1(x)
        #(n, 256)
        x = F.relu(x)
        x = self.fc2(x)
        #(n, 128)
        x = F.relu(x)
        x = self.fc3(x)
        #(n, 5)
        return x
