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
        size = len(x)
        _, x = self.gru(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x.view(size)

class NONSTATIC_GRU(nn.Module):
    def __init__(self, embedding_size, hidden_size, batch_first=True, vocab_size=100000, dropout=0.25):
        super(NONSTATIC_GRU, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
        input_size=embedding_size, hidden_size=hidden_size,
        batch_first=batch_first
        )

        self.fc1 = nn.Linear(hidden_size, 400)
        self.fc2 = nn.Linear(400, 1)

    def forward(self, x):
        """
        Input will be of shape (batch, seq_length, embedding_sie)
        """
        size = len(x)
        x = self.embedding(x)
        _, x = self.gru(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.view(size)

class NONSTATIC_GRU_BIDIRECTIONAL(nn.Module):
    def __init__(self, embedding_size, hidden_size, batch_first=True, vocab_size=100000, dropout=0.25):
        super(NONSTATIC_GRU_BIDIRECTIONAL, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.relu = nn.ReLU()

        self.gru = nn.GRU(
        input_size=embedding_size, hidden_size=hidden_size,
        batch_first=True, bidirectional=True
        )

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size*2, 400)
        self.fc2 = nn.Linear(400, 1)

    def forward(self, x):
        """
        Input will be of shape (batch, seq_length, embedding_sie)
        """
        size = len(x)
        x = self.embedding(x)
        _, x = self.gru(x)
        x = self.flatten(x.transpose(0,1))
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.view(size)
