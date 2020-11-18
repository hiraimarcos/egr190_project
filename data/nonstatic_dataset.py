import torch
import os, re
import pandas as pd
import preprocessor as p
import numpy as np
from torch.utils.data import Dataset
from .vocab import Vocab

dir_path = os.path.dirname(os.path.realpath(__file__))

class TweeterDataNonstatic(Dataset):
    """
    This is the class for our dataset.
    When creating an instance of this dataset, select test, val, or train
    as the setname
    """
    def __init__(self, setname, example_length=30, vocab_size=100000):
        assert setname in ['train1', 'train2', 'train3', 'train4','train', 'test', 'val']

        self.vocab = Vocab(vocab_size=100000)

        p.set_options(p.OPT.URL) # remove only URLs
        self.clean = p.clean
        self.pattern = re.compile(r'([^\s\w\@\#])+')

        self.example_length = example_length
        self.setname = setname
        self.path = os.path.join(dir_path, setname)
        index = os.path.join(self.path, 'index.csv')

        # maps index of points in the dataset to tweet_ids
        self.index = pd.read_csv(index, index_col=0)
        self.index = self.index.drop(619056)
        self.index = self.index.reset_index(drop=False)
        self.len = len(self.index)

        # cleans tweet before we can tokenize
        p.set_options(p.OPT.URL) # remove only URLs
        self.clean = p.clean

        # patter that identifies all but alphanumeric characters and spaces
        self.pattern = re.compile(r'([^\s\w\#\@])+')

    def __len__(self):
        return len(self.index)


    # Returns the tokenized text and label of the entry with the given id
    def __getitem__(self, id):
        text, label = self.get_pure_text(id)

        # this function will also pad/crop
        tweets = self.clear_text(text)
        indices = self.vocab.to_vector(tweets)
        sample = (indices, label)
        return sample

    # cleans the tweet and return split version
    def clear_text(self, text):
        #  remove urls and to lowercase
        text = self.clean(text).lower()

        # remove all but alphanumeric and spaces and split tweet
        text = self.pattern.sub("", text)
        text = text.split()

        # pad or crop so output has length 30
        if len(text) > self.example_length:
            text = text[:self.example_length]
        if len(text) < self.example_length:
            pad = ['<pad>' for _ in range(self.example_length-len(text))]
            text = text + pad
        return text

    def get_pure_text(self, id):
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
        return text, label
