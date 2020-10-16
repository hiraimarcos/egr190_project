import os, random, json
import pandas as pd

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
    accounts = pd.read_csv(os.path.join(dir_path,'accounts.csv'))

    # get the id of republican accounts
    rep_names = accounts.loc[accounts['Party']=='R']['Handle'].unique()
    rep_names = set(rep_names)
    rep_names

    # get id of democratic accounts
    dem_names = accounts.loc[accounts['Party']=='D']['Handle'].unique()
    dem_names = set(dem_names)
    dem_names

    # open file with tweets
    with open(os.path.join(dir_path,'tweets.jsonl'), 'r') as json_lines:
        # itereate over all lines
        while True:
            tweet = json_lines.readline()
            # break if next line is blank
            if tweet == "":
                break

            tweet = json.loads(tweet)

            # only get relevant information from json
            text = tweet['full_text']
            username = str(tweet['user']['screen_name'])
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
                if username in rep_names:
                    path = os.path.join(os.path.join(train_dir, "republican"),
                        filename)
                elif username in dem_names:
                    path = os.path.join(os.path.join(train_dir, "democrat"),
                        filename)

            # tweet goes to test set
            elif seed < TEST_SPLIT:
                test_index.append(tweet_id)
                if username in rep_names:
                    path = os.path.join(os.path.join(test_dir, "republican"),
                        filename)
                elif username in dem_names:
                    path = os.path.join(os.path.join(test_dir, "democrat"),
                        filename)

            # tweet goes to validation set
            else:
                val_index.append(tweet_id)
                if username in rep_names:
                    path = os.path.join(os.path.join(val_dir, "republican"),
                        filename)
                    
                elif username in dem_names:
                    path = os.path.join(os.path.join(val_dir, "democrat"),
                        filename)
                    
            if path:
                with open(path, 'w') as f:
                    f.write(text)

    test_index = pd.Series(test_index)
    test_index.to_csv(os.path.join(test_dir, 'index.csv'))
    train_index = pd.Series(train_index)
    train_index.to_csv(os.path.join(train_dir, 'index.csv'))
    val_index = pd.Series(val_index)
    val_index.to_csv(os.path.join(val_dir, 'index.csv'))
                

if __name__ == "__main__":
    main()