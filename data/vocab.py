import torch

class Vocab:
    def __init__(self, vocab_size=100000):
        self.idx = dict()
        self.idx['<pad>'] = 0
        self.idx_len = 1
        self.idx_max = vocab_size

    # returns the index of any token
    def get_index(self, token):
        # if token in index return the value
        if token in self.idx:
            return self.idx[token]
        # if token not in index but vocab hasn't reached max size, add token to
        # vocab
        elif self.idx_len < self.idx_max:
            idx = self.idx_len
            self.idx_len += 1
            self.idx[token] = idx
            return idx
        # if word beyond vocab size, return 0
        else:
            return 0

    # takes as input list of words and returns the associated numbers
    def to_vector(self, words):
        v = [self.get_index(w) for w in words]
        return torch.LongTensor(v)
