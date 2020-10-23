import argparse
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import json
import numpy as np
from datetime import datetime
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from gensim.models import LdaMulticore
import pickle

from topic_modeling import preprocess_text, text2corpus
from reddit_dataset import load_all_comments

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_results', help='Save inference results', default=None, type=str)
    parser.add_argument('--saved_results', help='Use previous inference results', required=True, type=str)
    parser.add_argument('--pos_bias', help='Bias adjustment for positive results', default=1, type=float)
    parser.add_argument('--comments_cached', help='Cached comments file', required=True, type=str)
    parser.add_argument('--topic_model', type=str, help="topic model file to load")
    parser.add_argument('--load_preprocess', default=False, action="store_true", help="Load preprocessed documents from temp file. Used only after a first usage of topic modeling")

    return parser

CLASSES = {0: "negative", 2: "positive", 1: "neutral"}
COLOR_CLASSES = {0:"red", 1:"green", 2:'blue'}
POS_BIAS = 1.0

def filter_comment_flair(comments_list, data, filter):
    d = np.array([(c[data], idx) for idx, c in enumerate(comments_list) if c["submission_flair"] == filter])

    indice_data = np.array([t[1] for t in d]).astype(np.int)
    d = np.array([t[0] for t in d])

    return d, indice_data

#This function will plot 2d histogram of score by neg/pos index
def plot_sentiment_score(results, scores):

    res_reduced = np.argmax(results, axis=-1)
    num_classes = len(CLASSES)

    hist_data = []
    label_data = []
    color = []
    index_class = []
    for idx, (k,v) in enumerate(CLASSES.items()):

        #ax = fig.add_subplot(1, num_classes, idx + 1, title="Hist for class {}".format(v))

        class_indices = np.nonzero(res_reduced == k)[0]
        hist_data.append(scores[class_indices])
        label_data.append(v)
        color.append(COLOR_CLASSES[k])
        index_class.append(k)
        #ax.hist(scores[class_indices], bins=50, range=[-50, 100], log=True)

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    n, bins, p = ax.hist(hist_data, label=label_data, density=False, log=True, bins=30, range=[-50, 350],  color=color)
    ax.set_xlabel("Reddit score")
    ax.set_ylabel("Number of comment (Log scale)")
    ax.legend()
    ax.set_title("Histogram of comments per reddit score")

    fig2, ax2 = plt.subplots(1, 1, figsize=(20, 20))
    total_count = np.sum(n, axis=0)
    data_pos = POS_BIAS * n[index_class.index(2)] / total_count * 100
    data_neg = n[index_class.index(0)] / total_count * 100

    mid_bins = np.zeros((len(data_pos),))
    for idx in range(len(bins) - 1):
        mid_bins[idx] = 0.5 * (bins[idx] + bins[idx + 1])
    width = 5

    ax2.bar(mid_bins + width / 2, data_pos, width, color="blue")
    ax2.bar(mid_bins - width / 2, data_neg, width, color="red")

    ax2.legend(["Positive", "Negative"])
    ax2.set_ylabel("Percentage of total comments")
    ax2.set_xlabel("Reddit score")
    ax2.set_title("Percentage of total comments per reddit score")

def poly_interpolation(x, y, degree=2):
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]

    model = make_pipeline(PolynomialFeatures(degree), ElasticNet(alpha=0.1, l1_ratio=0.2))
    model.fit(x, y)
    y_plot = model.predict(x)

    return y_plot

def plot_sentiment_time(results, time, bins=30, title_suffix=""):

    res_reduced = np.argmax(results, axis=-1)

    time_min = np.min(time)
    time_max = np.max(time)
    time = (time - time_min) / (time_max - time_min)

    num_classes = len(CLASSES)

    hist_data = []
    label_data = []
    color = []
    index_class = []
    for idx, (k,v) in enumerate(CLASSES.items()):

        class_indices = np.nonzero(res_reduced == k)[0]
        hist_data.append(time[class_indices])
        label_data.append(v)
        color.append(COLOR_CLASSES[k])
        index_class.append(k)

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    n, bins, p = ax.hist(hist_data, label=label_data, density=False, bins=bins, color=color)

    num_ticks = 15
    step = 1 / num_ticks
    xticks = np.arange(0, 1 + step, step)
    xtick_labels = [datetime.utcfromtimestamp((time_max - time_min) * t + time_min).strftime('%Y-%m-%d')
                    for t in xticks]

    ax.set_xticklabels(xtick_labels)
    ax.set_xticks(xticks)
    ax.set_ylabel("Number of comments")
    ax.set_xlabel("Date")
    ax.legend()
    ax.set_title("Histogram of comments along times" + title_suffix)

    time = np.array([c["created_utc"] for c in comments_list])
    fig2, ax2 = plt.subplots(1, 1, figsize=(20, 20))

    total_count = np.sum(n, axis=0)
    total_count[total_count == 0] = 1

    data_pos = POS_BIAS * n[index_class.index(2)] / total_count * 100
    data_neg = n[index_class.index(0)] / total_count * 100

    mid_bins = np.zeros((len(data_pos),))
    for idx in range(len(bins) - 1):
        mid_bins[idx] = 0.5 * (bins[idx] + bins[idx + 1])


    ax2.plot(mid_bins, data_pos, "bo-")
    ax2.plot(mid_bins, data_neg, "ro-")

    enet_pos = poly_interpolation(mid_bins, data_pos, degree=7)
    enet_neg = poly_interpolation(mid_bins, data_neg, degree=7)
  
    ax2.plot(mid_bins, enet_pos, color="cyan")
    ax2.plot(mid_bins, enet_neg, color="orange")

    ax2.legend(["Positive - polynomial regression", "Negative - polynomial regression", "Positive", "Negative"])
    ax2.set_ylabel("Percentage of total comments")
    ax2.set_xlabel("Date")
    ax2.set_xticklabels(xtick_labels)
    ax2.set_xticks(xticks)
    ax2.set_ylim(bottom=0, top=25)
    ax2.set_title("Percentage of total comments along times" + title_suffix)

def plot_sentiment_topics(results, topic_model, comments_topics, comments_corpus):

    results = np.argmax(results, axis=-1)

    top_topics = [tid for tid, v in topic_model.print_topics(num_topics=50)]

    hist_data = []
    label_data = []
    color = []
    index_class = []
    for idx, (k,v) in enumerate(CLASSES.items()):

        class_indices = np.nonzero(results == k)[0]
        class_topics = comments_topics[class_indices]

        topics = []
        for t in class_topics:
            topics += [top_topics.index(tid) for tid in list(t) if tid in top_topics]

        hist_data.append(topics)
        label_data.append(v)
        color.append(COLOR_CLASSES[k])
        index_class.append(k)

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    n, bins, p = ax.hist(hist_data, label=label_data, density=False, bins=len(top_topics), color=color)
    
    ax.set_ylabel("Number of comments")
    ax.set_xlabel("Topics id")
    ax.set_xticks(np.arange(len(top_topics)))
    ax.legend()
    ax.set_title("Histogram of comments per top topics")

    fig3, ax3 = plt.subplots(1, 1, figsize=(20, 20))
    total_count = np.sum(n, axis=0)
    total_count[total_count == 0] = 1

    data_pos = POS_BIAS * n[index_class.index(2)] / total_count * 100
    data_neg = n[index_class.index(0)] / total_count * 100

    mid_bins = np.arange(len(data_pos))
    width = 0.35

    ax3.bar(mid_bins + width / 2, data_pos, width, color="blue")
    ax3.bar(mid_bins - width / 2, data_neg, width, color="red")

    ax3.legend(["Positive", "Negative"])
    ax3.set_ylabel("Percentage of total comments")
    ax3.set_xlabel("Topics id")
    ax3.set_xticks(np.arange(len(top_topics)))
    ax3.set_title("Percentage of total comments per top topics")

    top_topics_str = [topic_model.print_topic(tid).split("+") for tid in top_topics]

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 13))
    ax2.axis("off")
    #ax2.axis('tight')
    table = ax2.table(cellText=top_topics_str, loc="center", rowLabels=[str(i) for i in range(len(top_topics))])

    table.scale(1.15, 1.1)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    #ax2.set_frame_on(False)
    ax2.set_title("Representation of top topics")
    fig2.savefig('topic representation.png', bbox_inches="tight",
                dpi=200)

if __name__ == "__main__":

    parser = common_arg_parser()

    args = parser.parse_args()

    POS_BIAS = args.pos_bias
    print(POS_BIAS)

    results = np.load(args.saved_results)["results"]

    comments_list = load_all_comments(db_name=args.comments_cached, only=["score", "created_utc", "submission_flair"])

    score = np.array([c["score"] for c in comments_list])
    time = np.array([c["created_utc"] for c in comments_list])

    plot_sentiment_score(results, score)

    time_filtered, indices = filter_comment_flair(comments_list, "created_utc", "NEWS")
    res_reduced = results[indices]
    plot_sentiment_time(res_reduced, time_filtered, title_suffix=", for roadmap update")

    plot_sentiment_time(results, time, bins=100)

    time_filtered, indices = filter_comment_flair(comments_list, "created_utc", "OFFICIAL")
    res_reduced = results[indices]
    plot_sentiment_time(res_reduced, time_filtered, title_suffix=", for official flair", bins=100)

    time_filtered, indices = filter_comment_flair(comments_list, "created_utc", "FLUFF")
    res_reduced = results[indices]
    plot_sentiment_time(res_reduced, time_filtered, title_suffix=", for fluff flair", bins=60)

    time_filtered, indices = filter_comment_flair(comments_list, "created_utc", "IMAGE")
    res_reduced = results[indices]
    plot_sentiment_time(res_reduced, time_filtered, title_suffix=", for image flair", bins=60)


    if args.topic_model:
        topic_model = LdaMulticore.load(args.topic_model)

        if not args.load_preprocess:
            comments_text = load_all_comments(db_name=args.comments_cached, only="body")

            comments_text_filtered, dictionary, comments_tokenized = preprocess_text(comments_text)
            print("filtered text")

            comments_corpus = text2corpus(comments_text_filtered, dictionary)
            del comments_text_filtered
            with open("temp_corpus.pickle", "wb") as f:
                pickle.dump((comments_corpus, dictionary), f)
        else:
            with open("temp_corpus.pickle", "rb") as f:
                comments_corpus, dictionary = np.array(pickle.load(f))

        comments_topics = np.array([[tid for tid, v
                in topic_model[comments_corpus[idx]] if v > 0.2] for idx in range(len(comments_list))])

        plot_sentiment_topics(results, topic_model, comments_topics, comments_corpus)

    plt.show()
