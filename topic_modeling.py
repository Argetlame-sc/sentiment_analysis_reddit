
from gensim.models import LdaMulticore
from gensim.corpora.dictionary import Dictionary
import gensim
import argparse
import numpy as np

from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from gensim.models.phrases import Phrases, Phraser
import pickle
import sys
import os
import time
import copy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import concurrent
from reddit_dataset import load_all_comments

def preprocess_tokenize(docs, stoplist):
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()

    texts = [tokenizer.tokenize(doc.lower()) for doc in docs]
    texts = [[token for token in text if not token.isnumeric() and token not in stoplist] for text in texts]
    #texts = [[token for token in text if token not in stoplist] for text in texts]
    # Remove words that are only one character.
    texts = [[token for token in text if len(token) > 1] for text in texts]


    texts = [[lemmatizer.lemmatize(token) for token in text] for text in texts]

    return texts

def preprocess_text(docs):
    num_task = os.cpu_count()
    len_slices = len(docs) // num_task
    remainder_slices = len(docs) % num_task

    texts = []
    stoplist = set(stopwords.words('english'))
    
    wn.ensure_loaded()
    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_task) as executor:

        futures_tokenize = []
        for n in range(0, num_task):

            upper_bound = (n+1) * len_slices
            if n == num_task - 1:
                upper_bound = (n+1) * len_slices + remainder_slices

            print(n, upper_bound)
            futures_tokenize.append(executor.submit(preprocess_tokenize, docs[n * len_slices:upper_bound],
                            stoplist))

        for future in concurrent.futures.as_completed(futures_tokenize):
            texts += future.result()

    t_stop = time.perf_counter()
    print("removed stopwords and lemmatized in {} s".format(t_stop - t_start))
    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigram = Phraser(Phrases(texts, min_count=20))
    for idx in range(len(texts)):
        for token in bigram[texts[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                texts[idx].append(token)

    print("Done bigrams")
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=30, no_above=0.5)
    dictionary.filter_tokens(bad_ids=[dictionary.token2id["like"]])
    special_tokens = {'_pad_': 0}
    dictionary.patch_with_special_tokens(special_tokens)

    return texts, dictionary

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', required=True, type=str, help="Reddit data to be extracted")
    parser.add_argument('--load', action="store_true", default=False, help="Load pre existing lda")
    parser.add_argument('--load_preprocess', default=False, action="store_true", help="Load preprocessed documents from temp file. Used only after a first usage of topic modeling")


    return parser

def text2corpus(texts, dictionary):

    comments_corpus = [dictionary.doc2bow(txt) for txt in texts]
    corpus_tmp = []
    for d in comments_corpus:
        if len(d) > 0:
            corpus_tmp.append(d)
        else:
            corpus_tmp.append([(0, 1)])

    return corpus_tmp

if __name__ == "__main__":

    parser = common_arg_parser()
    args = parser.parse_args()


    comments_text = load_all_comments(only="body", db_name=args.data)

    print("loaded data")
    if not args.load_preprocess:
        #comments_text = [c["body"] for c in comments_list]

        print("extracted text")

        comments_text_filtered, dictionary = preprocess_text(comments_text)
        print("filtered text")

        comments_corpus = text2corpus(comments_text_filtered, dictionary)
        del comments_text_filtered
        with open("temp_corpus.pickle", "wb") as f:
            pickle.dump((comments_corpus, dictionary), f)
    else:
        with open("temp_corpus.pickle", "rb") as f:
            comments_corpus, dictionary = np.array(pickle.load(f))

    print("created corpus")
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(comments_corpus))

    num_topics = 150
    if args.load:
        model = LdaMulticore.load("topic_models/model_comments")
    else:
        model = LdaMulticore(comments_corpus, id2word=dictionary, num_topics=num_topics)
        print("model done")
        model.save("topic_models/model_comments")

    print(model.print_topics(20))

    top_topics = model.top_topics(comments_corpus) #, num_words=20)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)

    #from pprint import pprint
    #pprint(top_topics)

    for _ in range(10):
        idx = np.random.randint(0, len(comments_text))

        print("comment: {} - topics: {}".format(comments_text[idx],
                [(model.show_topic(tid, topn=10), v) for tid, v
                in model[comments_corpus[idx]] if v > 0.15]))
