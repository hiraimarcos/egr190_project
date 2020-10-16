import torch
import os
import pandas as pd
from torch.utils.data import Dataset

dir_path = os.path.dirname(os.path.realpath(__file__))

class TweeterData(Dataset):
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
        self.index = pd.read_csv(index, index_col=0,
                                    names=['Tweet_id'], header=0)

    def __len__(self):
        return len(self.index)


    # for now this function will return the text itself\
    # NEED TO ADJUST, so it tekenizes and adds padding, and returns a
    # list of word embeddings
    def __getitem__(self, id):
        tweet_id = self.index.iloc[id]['Tweet_id']
        filename = f"{tweet_id}.txt"
        rep_path = os.path.join(self.path, "republican", filename)
        dem_path = os.path.join(self.path, "democrat", filename)
        # check whether this 
        if os.path.exists(rep_path):
            with open(rep_path, 'r') as f:
                text = f.read()
        elif os.path.exists(rdem_path):
            with open(dem_path, 'r') as f:
                text = f.read()
        return text