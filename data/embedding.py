import h5py, os

dir_path = os.path.dirname(os.path.realpath(__file__))


# returns dictionary that maps words to the pre trained vector
# uses the smaller version of conceptnet
def word2vec():
    # load embeddings from file
    with h5py.File(os.path.join(dir_path, 'mini.h5'),'r') as f:
        all_words = [word.decode('utf-8') for word in f['mat']['axis1'][:]]
        all_embeddings = f['mat']['block0_values'][:]

    # select only english words
    english_words = [word[6:] for word in all_words
                     if word.startswith('/c/en/')]
    # get the index of all english words
    english_word_indices = [i for i, word in enumerate(all_words)
                            if word.startswith('/c/en/')]
    # get the embeddings of all english words
    english_embeddings = all_embeddings[english_word_indices]
    # dictionary that maps the word to the index
    index = {word: i for i, word in enumerate(english_words)}

    # dictionary that maps words to vectors
    t = {word:english_embeddings[index[word]] for word in english_words}
    return t
