import torch.nn as nn
import math, torch
import preprocessor as p

class CNN_1CONV_MAX(nn.Module):
    def __init__(self, kernel_size=3, in_length=30, vocab_size=100000, embedding_dim=30):
        super().__init__()
        self.vocab = dict()
        self.vocab_max = embedding_dim
        self.vocab_len = 0 # initialize length to 0
        p.set_options(p.OPT.URL) # remove only URLs
        self.clean = p.clean
        self.pattern = re.compile(r'([^\s\w\@\#])+')
        self.input_length = in_length
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 1d convolutional layer
        self.conv = nn.Conv1d(
            in_channels, 300,
            kernel_size=kernel_size,
            padding=math.floor(kernel_size/2),
        )
        conv1_out = math.floor((in_length+2*math.floor(kernel_size/2)-kernel_size) + 1)
        self.maxpool = nn.MaxPool1d(kernel_size=5)
        maxpool_out = math.floor(1+(conv1_out-5)/5)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(maxpool_out * 300, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self,x):
        size = len(x)
        x = self.tokenize(x)
        x = self.embedding(x).transpose(0,1)
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

    def tokenize(self, text):
        # remove anwanted characters and split
        x = text.lower()
        x = self.clean(x)
        x = self.patter.sub("").split()
        # pad or crop to make sure all examples have same len
        if len(x) > self.input_length:
            x = x[:self.input_length]
        elif len(x) < self.input_length:
            pad = ['<pad>' for _ in range(self.example_length-len(text))]
            x = x + pad
        # initialize list with word indices
        v = []
        for word in x:
            # if word in vocab add word index
            if word in self.vocab:
                v.append(self.vocab[word])
            # if not in vocab but vocab not yet maxed, add to vocab
            elif self.vocab_len < self.vocab_max:
                self.vocab[word] = self.vocab_len
                v.append(vocab_len)
                self.vocab_len += 1
            # else, just ignore word by adding <pad> instead
            else:
                v.append(self.vocab['<pad>'])

        return torch.Tensor(v).long()
