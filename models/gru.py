from torch import nn

class STATIC_GRU(nn.Module):
    def __init__(self, embedding_size, hidden_size, batch_first=True):
        super(STATIC_GRU, self).__init__()

        self.relu = nn.ReLU()

        self.gru = nn.GRU(
        input_size=embedding_size, hidden_size=hidden_size,
        batch_first=batch_first
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(hidden_size, 400)
        self.fc2 = nn.Linear(400, 1)

    def forward(self, x):
        """
        Input will be of shape (batch, seq_length, embedding_sie)
        """
        print(x.shape)
        _, x = self.gru(x)
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class NONSTATIC_GRU(nn.Module):
    def __init__(self, embedding_size, hidden_size, batch_first=True, vocab_size=100000):
        super(NONSTATIC_GRU, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.relu = nn.ReLU()

        self.gru = nn.GRU(
        input_size=embedding_size, hidden_size=hidden_size,
        batch_first=batch_first
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(hidden_size, 400)
        self.fc2 = nn.Linear(400, 1)

    def forward(self, x):
        """
        Input will be of shape (batch, seq_length, embedding_sie)
        """
        x = self.embedding(x)
        _, x = self.gru(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
