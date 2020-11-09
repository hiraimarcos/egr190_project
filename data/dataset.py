import torch
import os, re
import pandas as pd
import preprocessor as p
import numpy as np
from torch.utils.data import Dataset
from .embedding import word2vec

dir_path = os.path.dirname(os.path.realpath(__file__))

class TweeterData(Dataset):
    """
    This is the class for our dataset.
    When creating an instance of this dataset, select test, val, or train
    as the setname
    """
    def __init__(self, setname, example_length=30):
        assert setname in ['train', 'test', 'val']
        self.example_length = example_length
        self.setname = setname
        self.path = os.path.join(dir_path, setname)
        index = os.path.join(self.path, 'index.csv')

        # maps index of points in the dataset to tweet_ids
        self.index = pd.read_csv(index, index_col=0,
                                    names=['Tweet_id'], header=0)

        # cleans tweet before we can tokenize
        p.set_options(p.OPT.URL) # remove only URLs
        self.clean = p.clean

        # patter that identifies all but alphanumeric characters and spaces
        self.pattern = re.compile(r'([^\s\w]|_)+')

        # get dict that maps word to embeddings
        self.embeddings = word2vec()

    def __len__(self):
        return len(self.index)


    # Returns the tokenized text and label of the entry with the given id
    def __getitem__(self, id):
        # get tweet id from index
        tweet_id = self.index.iloc[id]['Tweet_id']
        label = int(self.index.iloc[id]['Party']=="D")

        # check whether this tweet was made by republican or democrat
        # and get the text
        filename = f"{tweet_id}.txt"
        if label:
            path = os.path.join(self.path, "democrat", filename)
        else:
            path = os.path.join(self.path, "republican", filename)

        with open(path, "r") as f:
            text = f.read()

        tweets = self.tokenize(text)
        sample = (tweets, label)
        return sample

    # cleans the tweet and return split version
    def tokenize(self, text):
        #  remove urls
        text = self.clean(text)

        # remove all but alphanumeric and spaces and split tweet
        text = self.pattern.sub("", text).split()

        # pad or crop so output has length 30
        if len(text) > self.example_length:
            text = text[:self.example_length]
        if len(text) < self.example_length:
            pad = ['<pad>' for _ in range(self.example_length-len(text))]
            text += pad
        return text

    # returns tensor with word embeddings from a list of words
    def embed(self, tweets)
        vectors = []
        for word in tweets:
            # if word has embedding add the embedding
            if word in self.embeddings:
                vectors.append(self.embeddings[word])

            # if word doesn't have embedding use array of zeroes (we're
            # basically ignoring the words for which we don't have an embedding)
            else:
                vectors.append(np.zeros(300))
        return torch.tensor(np.stack(vectors)).view(300,30)
