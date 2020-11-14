import torch.nn as nn
import math

class CNN_1CONV(nn.Module):
    def __init__(self, in_channels, kernel_size, in_length, stride):
        super().__init__()
        # 1d convolutional layer
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            kernel_size=kernel_size,
            padding=math.floor(kernel_size/2),
            out_chanels=100
        )
        conv1_out = math.floor((in_length+2*math.floor(kernel_size/2)-kernel_size) + 1)
        self.maxpool = nn.MaxPool1d(kernel_size=5)
        maxpool_out = math.floor(1+(conv1_out-5)/5)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(maxpool_out * 100, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self,x):
        size = len(x)
        # first convolution
        x = self.conv(x)
        x = self.relu(x)

        # maxpool
        x = self.maxpool(x)

        # flatten output maps
        x = self.flatten(x)

        # first fully connected layer
        x = self.fc1(x)
        x = self.relu(x)

        # second fully connected layer - with binary output
        x = self.fc2(x)

        # here we use view to make sure the output is a 1d array
        return x.view(size)

class CNN_2CONV(nn.Module):
    def __init__(self, kernel_size, in_length):
        super().__init__()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv1d(
            in_channels = 200,
            out_chanels = 200,
            kernel_size = kernel_size,
            padding = math.floor(kernel_size/2),
        )
        conv1_out = math.floor((in_length+2*math.floor(kernel_size/2)-kernel_size) + 1)
        self.conv2 = nn.Conv1d(
            in_channels=200,
            out_chanels=100,
            kernel_size=5,
            stride=5
        )
        conv2_out = math.floor((100-5)/5 + 1)
        self.fc1 = nn.Linear(conv2_out * 100, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x):
        size = len(x)
        # first convolutional layer
        x = self.conv1(x)
        x = self.relu(x)

        # second convolutional layer
        x = self.conv2(x)
        x = self.relu(x)

        # flatten feature maps
        x = self.flatten(x)

        # first fully connected layer
        x = self.fc1(x)
        x = self.relu(x)

        # second fully connected layer
        x = self.fc2(x)

        return x.view(size)
