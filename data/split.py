import os, random
import pandas as pd
import json

random.seed(0)

TRAIN_SPLIT = 0.60
TEST_SPLIT = 0.80
VAL_SPLIT = 1.0

dir_path = os.path.dirname(os.path.realpath(__file__))
val_dir = os.path.join(dir_path, 'val')
test_dir = os.path.join(dir_path, 'test')
train_dir = os.path.join(dir_path, 'train')

def main():
    test_index = []
    train_index = []
    val_index = []

    # load dataframe with account ids and party affiliation
    accounts = pd.read_csv('accounts.csv')

    # get the id of republican accounts
    rep_ids = accounts.loc[accounts['Party']=='R']['Uid'].unique()
    rep_ids = set(rep_ids)
    rep_ids

    # get id of democratic accounts
    dem_ids = accounts.loc[accounts['Party']=='D']['Uid'].unique()
    dem_ids = set(dem_ids)
    dem_ids

    # open file with tweets
    with open('tweets.jsonl', 'r') as json_lines:
        # itereate over all lines
        while True:
            tweet = json_lines.readline()
            # break if next line is blank
            if tweet == "":
                break

            # only get relevant information from json
            text = tweet['full_text']
            user_id = tweet['user']['id']
            tweet_id = tweet['id']
            filename = f"{str(tweet_id)}.txt"

            #  generate random seed we'll use to assign tweet to test,train or val
            seed = random.random()

            # initiate path variable
            path = ""

            # tweet goes to training set
            if seed < TRAIN_SPLIT:
                train_index.append(tweet_id)
                # determine whether author was rep or dem
                if user_id in rep_ids:
                    path = os.path.join(os.path.join(train_dir, "republican"),
                        filename)
                elif user_id in dem_ids:
                    path = os.path.join(os.path.join(train_dir, "democrat"),
                        filename)

            # tweet goes to test set
            if seed < TEST_SPLIT:
                test_index.append(tweet_id)
                if user_id in rep_ids:
                    path = os.path.join(os.path.join(test_dir, "republican"),
                        filename)
                elif user_id in dem_ids:
                    path = os.path.join(os.path.join(test_dir, "democrat"),
                        filename)

            # tweet goes to validation set
            else:
                val_index.append(tweet_id)
                if user_id in rep_ids:
                    path = os.path.join(os.path.join(val_dir, "republican"),
                        filename)
                elif user_id in dem_ids:
                    path = os.path.join(os.path.join(val_dir, "democrat"),
                        filename)
            
            if path:
                with open(path, 'w') as f:
                    f.write(text)
                

if __name__ == "__main__":