from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

NUM_WORDS=25000


def filter_markdown_quoting(text):

    body_list = text.split("\n")
    body = ""
    for line in body_list:
        if len(line) == 0:
            body = body + "\n"
        elif line[0] != ">":
            body = body + line + "\n"

    return body

def preprocessing_infer(text, tokenizer, labels=None, max_len=None):
    seq = tokenizer.texts_to_sequences(text)
    X = pad_sequences(seq, padding="post", maxlen=max_len)

    y = None
    if not labels is None:
        y = to_categorical(np.asarray(labels))

    return X, y

def preprocessing(text_train, text_val, label_train, label_val,
                    text_unlabeled, tokenizer=None, get_max_len_seq_unl=None):
    assert isinstance(text_train, list)
    assert isinstance(text_val, list)


    if tokenizer is None:
        tokenizer = Tokenizer(num_words=NUM_WORDS,filters='"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n\'',
                      lower=True)

        if not text_unlabeled is None:
            tokenizer.fit_on_texts(text_train + text_val + text_unlabeled)
        else:
            tokenizer.fit_on_texts(text_train + text_val)

    seq_train = tokenizer.texts_to_sequences(text_train)
    seq_val=tokenizer.texts_to_sequences(text_val)

    X_train = pad_sequences(seq_train, padding="post")
    X_val = pad_sequences(seq_val, padding="post")
    max_len = max(X_train.shape[1], X_val.shape[1])

    X_train_unl = None
    if not text_unlabeled is None:
        seq_train_unl = tokenizer.texts_to_sequences(text_unlabeled)
        X_train_unl = pad_sequences(seq_train_unl, padding="post")

        max_len = max(X_train.shape[1], max(X_val.shape[1], X_train_unl.shape[1]))
        X_train_unl = pad_sequences(seq_train_unl, maxlen=max_len, padding="post")
    elif not get_max_len_seq_unl is None:
        max_len_unl = get_max_len_seq_unl(tokenizer)
        max_len = max(X_train.shape[1], max(X_val.shape[1], max_len_unl))

    X_train = pad_sequences(seq_train, maxlen=max_len, padding="post")
    X_val = pad_sequences(seq_val, maxlen=max_len, padding="post")

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))


    y_train = to_categorical(np.asarray(label_train))
    y_val = to_categorical(np.asarray(label_val))

    return X_train, X_val, y_train, y_val, X_train_unl, tokenizer, min(len(word_index)+1,NUM_WORDS)
