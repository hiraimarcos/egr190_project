import torch.nn as nn
import math, torch, re
import preprocessor as p

class CNN_2CONV_MAX_NONSTATIC(nn.Module):
    def __init__(self, in_length=30, vocab_size=100000, embedding_dim=30, dropout=0.25):
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
    def __init__(self, in_length=30, vocab_size=100000, embedding_dim=64, dropout=0.25):
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
    def __init__(self, in_length=30, vocab_size=100000, embedding_dim=64, dropout=0.25):
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

class TextClassifier(nn.ModuleList):

   def __init__(self, num_words,seq_len=30, embedding_size=64, out_size=32, stride=2):
       super(TextClassifier, self).__init__()

       # Parameters regarding text preprocessing
       self.seq_len = seq_len
       self.num_words = num_words
       self.embedding_size = embedding_size

       # Dropout definition
       self.dropout = nn.Dropout(0.25)

       # CNN parameters definition
       # Kernel sizes
       self.kernel_1 = 2
       self.kernel_2 = 3
       self.kernel_3 = 4
       self.kernel_4 = 5

       # Output size for each convolution
       self.out_size = out_size
       # Number of strides for each convolution
       self.stride = stride

       # Embedding layer definition
       self.embedding = nn.Embedding(self.num_words + 1, self.embedding_size, padding_idx=0)

       # Convolution layers definition
       self.conv_1 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)
       self.conv_2 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_2, self.stride)
       self.conv_3 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_3, self.stride)
       self.conv_4 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_4, self.stride)

       # Max pooling layers definition
       self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
       self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
       self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
       self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)

       # Fully connected layer definition
       self.fc = nn.Linear(self.in_features_fc(), 1)

   def in_features_fc(self):
       '''Calculates the number of output features after Convolution + Max pooling

       Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
       Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1

       source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
       '''
       # Calcualte size of convolved/pooled features for convolution_1/max_pooling_1 features
       out_conv_1 = ((self.embedding_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
       out_conv_1 = math.floor(out_conv_1)
       out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
       out_pool_1 = math.floor(out_pool_1)

       # Calcualte size of convolved/pooled features for convolution_2/max_pooling_2 features
       out_conv_2 = ((self.embedding_size - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
       out_conv_2 = math.floor(out_conv_2)
       out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
       out_pool_2 = math.floor(out_pool_2)

       # Calcualte size of convolved/pooled features for convolution_3/max_pooling_3 features
       out_conv_3 = ((self.embedding_size - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
       out_conv_3 = math.floor(out_conv_3)
       out_pool_3 = ((out_conv_3 - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
       out_pool_3 = math.floor(out_pool_3)

       # Calcualte size of convolved/pooled features for convolution_4/max_pooling_4 features
       out_conv_4 = ((self.embedding_size - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
       out_conv_4 = math.floor(out_conv_4)
       out_pool_4 = ((out_conv_4 - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
       out_pool_4 = math.floor(out_pool_4)

       # Returns "flattened" vector (input for fully connected layer)
       return (out_pool_1 + out_pool_2 + out_pool_3 + out_pool_4) * self.out_size

   def forward(self, x):
       # Sequence of tokes is filterd through an embedding layer
       x = self.embedding(x)

       # Convolution layer 1 is applied
       x1 = self.conv_1(x)
       x1 = torch.relu(x1)
       x1 = self.pool_1(x1)

       # Convolution layer 2 is applied
       x2 = self.conv_2(x)
       x2 = torch.relu((x2))
       x2 = self.pool_2(x2)

       # Convolution layer 3 is applied
       x3 = self.conv_3(x)
       x3 = torch.relu(x3)
       x3 = self.pool_3(x3)

       # Convolution layer 4 is applied
       x4 = self.conv_4(x)
       x4 = torch.relu(x4)
       x4 = self.pool_4(x4)

       # The output of each convolutional layer is concatenated into a unique vector
       union = torch.cat((x1, x2, x3, x4), 2)
       union = union.reshape(union.size(0), -1)

       # The "flattened" vector is passed through a fully connected layer
       out = self.fc(union)
       # Dropout is applied
       out = self.dropout(out)
       # Activation function is applied
       # out = torch.sigmoid(out)

       return out.squeeze()
