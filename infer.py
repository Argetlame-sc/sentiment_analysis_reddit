import os
import sys
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
import argparse
import pandas as pd
import importlib
import pickle
import json
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
from keras import backend as K
from model import *
from preprocessing import *
import numpy as np

import reddit_dataset
from train import data_generator

try:
    import matplotlib.pyplot as plt
except:
    plt = None
from datasets import DATASET


def display_confusion_matrix(cmat, score, precision, recall, num_cat):

    if plt is None:
        return

    CLASSES = np.arange(num_cat)

    plt.figure(figsize=(22,22))
    ax = plt.gca()
    ax.matshow(cmat, cmap='Reds')
    ax.set_xticks(range(num_cat))
    ax.set_xticklabels(CLASSES, fontdict={'fontsize': 12})
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticks(range(num_cat))
    ax.set_yticklabels(CLASSES, fontdict={'fontsize': 12})
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    titlestring = ""
    if score is not None:
        titlestring += 'f1 = {:.3f} '.format(score)
    if precision is not None:
        titlestring += '\nprecision = {:.3f} '.format(precision)
    if recall is not None:
        titlestring += '\nrecall = {:.3f} '.format(recall)
    if len(titlestring) > 0:
        ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})
    plt.show()

def create_onpick_event(txt, y, pos_path, neg_path, pos_ind, neg_ind,
                            neut_path=None, neut_ind=None):
    def onpick_x(event):

        ind = event.ind

        print(ind)
        for i in ind:
            if event.artist == pos_path:
                print(txt[pos_ind][i])
                print(y[pos_ind][i])
            elif event.artist == neg_path:
                print(txt[neg_ind][i])
                print(y[neg_ind][i])
            elif not neut_path is None and event.artist == neut_path:
                print(txt[neut_ind][i])
                print(y[neut_ind][i])

    return onpick_x

def tsne_plot(txt, latent_res, y, ndim=2, num_cat=2):
    assert ndim in [2, 3]
    assert num_cat in [2, 3]

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20, 20))
    if ndim == 2:
        ax = fig.add_axes([0, 0, 1, 1])
    else:
        ax = Axes3D(fig)

    perp = 200
    tsne = manifold.TSNE(n_components=ndim, init="random",
    random_state=42, perplexity=perp, verbose=2, n_iter=1300)
    X_transformed = tsne.fit_transform(latent_res)

    print(X_transformed.shape)
    ax.set_title("tSNE projection")
    if not y is None:
        y = np.argmax(y, axis=-1)

        if num_cat == 2:
            blue = y == 1
            green = None
        else:
            blue = y == 2
            green = np.nonzero(y == 1)[0]
        red = y == 0

        neut_path = None
        if ndim == 2:
            pos_path = ax.scatter(X_transformed[blue, 0], X_transformed[blue, 1], c='blue', picker=True)
            neg_path = ax.scatter(X_transformed[red, 0], X_transformed[red, 1], c='r', picker=True)

            if num_cat == 3:
                neut_path = ax.scatter(X_transformed[green, 0], X_transformed[green, 1], c='green', picker=True)
        else:
            pos_path = ax.scatter(X_transformed[blue, 0], X_transformed[blue, 1], zs=X_transformed[blue, 2], c='blue', picker=True)
            neg_path = ax.scatter(X_transformed[red, 0], X_transformed[red, 1], zs=X_transformed[red, 2], c='r', picker=True)
            if num_cat == 3:
                neut_path = ax.scatter(X_transformed[green, 0], X_transformed[green, 1], zs=X_transformed[green, 2], c='green', picker=True)

        fig.canvas.mpl_connect('pick_event', create_onpick_event(np.array(txt), y, pos_path,
                                        neg_path, np.nonzero(blue)[0], np.nonzero(red)[0],
                                        neut_path=neut_path, neut_ind=green))
    else:
        if ndim == 2:
            ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c='black', picker=True)
        else:
            ax.scatter(X_transformed[:, 0], X_transformed[:, 1], zs=X_transformed[:, 2], c='black', picker=True)

    ax.axis("tight")


    plt.show()


def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--load_model', help='Model to load to start training', required=True, type=str)
    parser.add_argument('--save_results', help='Save inference results', default=None, type=str)
    parser.add_argument('--use_saved_results', help='Use previous inference results', default=None, type=str)
    parser.add_argument('--num_hidden', help='Number of hidden dimension for lstm', default=32, type=int)
    parser.add_argument('--num_cat', help='Number of categories', default=2, type=int)
    parser.add_argument('--show_infer', type=int, help="Show a number of inference results")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for inference")
    parser.add_argument('--latent_vis', type=int, choices=[2, 3], help="Model output latent space with ndim")
    parser.add_argument('--confusion_vis', default=False, action="store_true", help="Show confusion matrix")
    parser.add_argument('--kfold_model', type=int, help="Use Kfold model")
    parser.add_argument('--infer_generator', default=False, action="store_true", help="If infer sample is compatible use a generator")
    parser.add_argument('--infer_samples', help="File containing samples to analyse", required=False, type=str)
    parser.add_argument('--dataset', required=True, choices=DATASET,
                            help="chosen dataset", type=str)
    return parser


if __name__ == "__main__":

    parser = common_arg_parser()

    args = parser.parse_args()

    tokenizer, vocabulary_size = pickle.load(open(args.dataset + ".pickle", "rb"))
    embedding_matrix = None
    text = []
    labels = None
    comments_list = None

    if args.infer_generator: # dir of comments to big to be loade at once

        X = reddit_dataset.load_comments_id(db_name=args.infer_samples)
        print("loaded {} ids".format(len(X)))
        len_seq = reddit_dataset.get_max_len_seq(tokenizer, args.dataset, db_name=args.infer_samples)
        print(len_seq)

        def fun_read_x(batch_id):
            comments = reddit_dataset.load_comments(batch_id, only="body", db_name=args.infer_samples)
            batch_x, _ = preprocessing_infer(comments, tokenizer, max_len=len_seq)

            return batch_x
        y_dummy = np.zeros((X.shape[0], args.num_cat))
        y = None
        gen = data_generator(X, y=y_dummy, fun_read_x=fun_read_x, batch_size=args.batch_size, validation=True)

    else:
        if args.infer_samples.endswith(".txt"):
            with open(args.infer_samples) as f:
                text = f.readlines()

        elif args.infer_samples.endswith(".csv"):
            df = pd.read_csv(args.infer_samples)
            if args.num_cat == 2:
                df = reddit_dataset.set_pos_review(df, 1)
            labels = list(df.loc[:, "review"])
            text = list(df.loc[: , "text"])

        elif args.infer_samples.endswith(".json"):
            comments_list = json.loads(open(args.infer_samples).read())
            for c in comments_list:
                text.append(c["body"])
        else:
            sys.exit(1)

        X, y = preprocessing_infer(text, tokenizer, labels=labels)
        len_seq = X.shape[1]

        print(X.shape)


    if not args.kfold_model:
        run = LSTM_Model(len_seq, vocabulary_size, 300, args.num_hidden, args.num_cat,
            embedding_matrix=embedding_matrix, trainable_embedding=False,
            out_latent=args.latent_vis, use_at=False, use_vat=False, lr=1e-3, use_weights=False)
    else:
        run = KFold_LSTM_Model(args.kfold_model, len_seq, vocabulary_size, 300, args.num_hidden, args.num_cat,
            embedding_matrix=embedding_matrix, trainable_embedding=False,
            out_latent=args.latent_vis, use_at=False, use_vat=False, lr=1e-3, use_weights=False)

    run.load_weights(args.load_model, by_name=True)

    y_dummy = np.zeros((X.shape[0], args.num_cat))

    if args.use_saved_results:
        results = np.load(args.use_saved_results)["results"]
        if args.latent_vis:
            latent_res = np.load(args.use_saved_results)["latent"]
            results = (results, latent_res)
    else:
        if not args.infer_generator:
            results = run.predict([X, y_dummy], batch_size=args.batch_size, verbose=1)
        else:
            results = run.predict(gen, steps=(X.shape[0] // args.batch_size + 1), verbose=1)[:X.shape[0]]


    if args.latent_vis:
        results, latent_res = results
        if args.save_results:
            np.savez_compressed(args.save_results, results=results, latent=latent_res)

        range_idx = np.arange(results.shape[0])
        np.random.shuffle(range_idx)
        range_idx = range_idx[:5000]

        if args.infer_generator:
            text = reddit_dataset.load_comments(range_idx, only="body", db_name=args.infer_samples)
        else:
            text = np.array(text)[range_idx]

        if not y is None:
            y = y[range_idx]

        latent_res = latent_res[range_idx]
        tsne_plot(text, latent_res, y, args.latent_vis, num_cat=args.num_cat)
        sys.exit(0)

    for i in range(args.num_cat):
        res = np.argmax(results, axis=-1)
        num_i = len(np.nonzero(res == i)[0])
        print("Number of element of class {} : {}".format(i, num_i))

    if args.save_results:
        np.savez_compressed(args.save_results, results=results)

    if not y is None:
        loss, accuracy, loss_cat = run.evaluate([X, y], np.ones_like(y), batch_size=args.batch_size, verbose=1)

        print("loss {} - accuracy {} - loss_cat {}".format(loss, accuracy, loss_cat), file=sys.stderr)

        res_reduced = np.argmax(results, axis=-1)
        labels = np.argmax(y, axis=-1)
        cmat = confusion_matrix(labels, res_reduced, labels=range(args.num_cat))
        score = f1_score(labels, res_reduced, labels=range(args.num_cat), average='micro')
        precision = precision_score(labels, res_reduced, labels=range(args.num_cat), average='micro')
        precision_macro = precision_score(labels, res_reduced, labels=range(args.num_cat), average='macro')

        recall = recall_score(labels, res_reduced, labels=range(args.num_cat), average='micro')
        cmat = (cmat.T / cmat.sum(axis=1)).T # normalized

        print('f1 score: {:.3f}, precision: {:.3f} - macro {:.3f} - recall: {:.3f}'.format(score, precision,
                                                precision_macro, recall))

        for idx, cls in enumerate(np.arange(args.num_cat)):
            score = f1_score(labels, res_reduced, labels=[idx], average='macro')
            precision = precision_score(labels, res_reduced, labels=[idx], average='macro')
            recall = recall_score(labels, res_reduced, labels=[idx], average='macro')
            print('For class {} - {} : f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(idx, cls, score, precision, recall))
        if args.confusion_vis:
            display_confusion_matrix(cmat, score, precision, recall, args.num_cat)


    if args.show_infer:
        range_idx = np.arange(results.shape[0])
        np.random.shuffle(range_idx)
        range_idx = range_idx[:min(results.shape[0], args.show_infer)]
        res_idx = range_idx.copy()
        if args.infer_generator:
            text = reddit_dataset.load_comments(range_idx, only="body", db_name=args.infer_samples)
            comments_list = reddit_dataset.load_comments(range_idx, db_name=args.infer_samples,
                                                        only=["ups", "score", "downs", "controversiality"])
            range_idx = range(len(text))

        for idx_r, idx in zip(res_idx, range_idx):
            #if (np.argmax(results[idx]) != 2):
            #    continue
            print()
            print(text[idx])
            print("review is {} ".format(results[idx_r, :]))
            print("review thresholded is {} ".format(np.argmax(results[idx_r])))
            if not y is None:
                print("correct value is {}".format(np.argmax(y[idx_r])))
            if not comments_list is None:
                print("Stats : ups {} - downs {} - score {} - controversiality {}".format(
                        comments_list[idx]["ups"], comments_list[idx]["downs"],
                        comments_list[idx]["score"], comments_list[idx]["controversiality"]))
