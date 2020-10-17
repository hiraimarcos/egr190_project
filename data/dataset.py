import torch
import os
import pandas as pd
import preprocessor as p
import numpy as np
from nltk.tokenize.casual import TweetTokenizer
from torch.utils.data import Dataset
from .embedding import word2vec

dir_path = os.path.dirname(os.path.realpath(__file__))

class TweeterData_v0(Dataset):
    """
    This is the class for our dataset.
    When creating an instance of this dataset, select test, val, or train
    as the setname
    """
    def __init__(self, setname):
        assert setname in ['train', 'test', 'val']
        self.setname = setname
        self.path = os.path.join(dir_path, setname)
        index = os.path.join(self.path, 'index.csv')

        # maps index of points in the dataset to tweet_ids
        self.index = pd.read_csv(index, index_col=0,
                                    names=['Tweet_id'], header=0)

        # cleans tweet before we can tokenize
        self.clean = p.clean

        # function that splits tweets into words and punctuation
        # turns everything into lower case
        self.tokenizer = TweetTokenizer(preserve_case=False)

        # get dict that maps word to embeddings
        self.embeddings = word2vec()

    def __len__(self):
        return len(self.index)


    # for now this function will return the text itself\
    # NEED TO ADJUST, so it tekenizes and adds padding, and returns a
    # list of word embeddings
    def __getitem__(self, id):
        # get tweet id from given index
        tweet_id = self.index.iloc[id]['Tweet_id']

        # check whether this tweet was made by republican or democrat
        # and get the text
        filename = f"{tweet_id}.txt"
        rep_path = os.path.join(self.path, "republican", filename)
        dem_path = os.path.join(self.path, "democrat", filename)
        if os.path.exists(rep_path):
            with open(rep_path, 'r') as f:
                text = f.read()
                label = 0 # choose republican -> 0
        elif os.path.exists(dem_path):
            with open(dem_path, 'r') as f:
                text = f.read()
                label = 1 # choose democrat -> 1

        # clean the tweet (erase mentions, hashtags, URLs, numbers)
        text = self.clean(text)
        # split into words and punctuation
        tokenized = self.tokenizer.tokenize(text)

        # crop if length is greater than 100
        if len(tokenized)>100:
            tokenized = tokenized[:100]

        vectors = []
        for word in tokenized:
            # if word has embedding add the embedding
            if word in self.embeddings:
                vectors.append(self.embeddings[word])

            # if word doesn't have embedding use array of zeroes (we're
            # basically ignoring the words for which we don't have an embedding)
            else:
                vectors.append(np.zeros(300))

        # pad if len < 100
        if len(vectors) < 100:
            zeros = [np.zeros(300) for i in range(100-len(vectors))]
            vectors = vectors + zeros

        sample = (torch.tensor(np.stack(vectors)), label)

        return sample
