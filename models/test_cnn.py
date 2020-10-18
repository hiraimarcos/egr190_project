import torch.nn as nn

class TEST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1d convolutional layer
        self.conv = nn.Conv1d(in_channels=300, out_channels=80, kernel_size=5, padding=2)
        self.maxpool = nn.MaxPool1d(kernel_size=5)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(20*80, 100)
        self.fc2 = nn.Linear(100, 1)
        
    def forward(self,x):
        size = len(x)
        # first convolution
        x = self.conv(x)
        x = self.relu(x)
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
        