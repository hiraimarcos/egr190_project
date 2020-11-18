import torch.nn as nn
import math, torch, re
import preprocessor as p

class CNN_2CONV_MAX_NONSTATIC(nn.Module):
    def __init__(self, in_length=30, vocab_size=100000, embedding_dim=30, dropout=0.15):
        super(CNN_2CONV_MAX_NONSTATIC, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # 1d convolutional layer
        self.conv1 = nn.Conv1d(
            embedding_dim, 100,
            kernel_size=3,
            padding=1
        )
        self.maxpool1 = nn.MaxPool1d(kernel_size=3)

        self.conv2 = nn.Conv1d(
        100, 300,
        kernel_size=3,
        padding=1
        )
        self.maxpool2 = nn.MaxPool1d(3, stride=3, padding=1)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(1200, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self,x):
        size = len(x)
        x = self.embedding(x).transpose(1,2)
        # first convolution
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        # second convolution
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        # flatten output maps
        x = self.flatten(x)
        x = self.dropout(x)
        # first fully connected layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # second fully connected layer - with binary output
        x = self.fc2(x)

        # here we use view to make sure the output is a 1d array
        return x.view(size)

class CNN_3CONV_MAX_NONSTATIC(nn.Module):
    def __init__(self, in_length=30, vocab_size=100000, embedding_dim=30, dropout=0.15):
        super(CNN_3CONV_MAX_NONSTATIC, self).__init__()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.conv1 = nn.Conv1d(
            embedding_dim, 50,
            kernel_size = 3,
            padding = 1,
        )
        self.maxpool1 = nn.MaxPool1d(2, stride=2)

        self.conv2 = nn.Conv1d(
            50, 200,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.maxpool2 = nn.MaxPool1d(2, stride=2, padding=1)
        self.conv3 = nn.Conv1d(
            200, 400,
            kernel_size=2,
            stride=1,
            padding=1
        )
        self.maxpool3 = nn.MaxPool1d(2, stride=2)
        self.fc1 = nn.Linear(1600, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x):
        size = len(x)

        x = self.embedding(x).transpose(1,2)

        # first convolutional layer
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        # second convolutional layer
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        # third convolutional layer
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool3(x)

        # flatten feature maps
        x = self.flatten(x)
        x = self.dropout(x)
        # first fully connected layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # second fully connected layer
        x = self.fc2(x)

        return x.view(size)

class CNN_4CONV_MAX_NONSTATIC(nn.Module):
    def __init__(self, in_length=30, vocab_size=100000, embedding_dim=30, dropout=0.15):
        super(CNN_4CONV_MAX_NONSTATIC, self).__init__()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.conv1 = nn.Conv1d(
            embedding_dim, 50,
            kernel_size = 3,
            padding = 1,
        )
        self.maxpool1 = nn.MaxPool1d(2, stride=2)
        self.conv2 = nn.Conv1d(
            50, 100,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.maxpool2 = nn.MaxPool1d(2, stride=2, padding=1)
        self.conv3 = nn.Conv1d(
            100, 200,
            kernel_size=2,
            stride=1,
            padding=1
        )
        self.maxpool3 = nn.MaxPool1d(2, stride=2)
        self.conv4 = nn.Conv1d(
            200, 400,
            kernel_size=2,
            stride=1,
            padding=1
        )
        self.maxpool4 = nn.MaxPool1d(2, stride=2)
        self.fc1 = nn.Linear(800, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x):
        size = len(x)

        x = self.embedding(x).transpose(1,2)

        # first convolutional layer
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        # second convolutional layer
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        # third convolutional layer
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool3(x)

        # flatten feature maps
        x = self.flatten(x)
        x = self.dropout(x)
        # first fully connected layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # second fully connected layer
        x = self.fc2(x)

        return x.view(size)
