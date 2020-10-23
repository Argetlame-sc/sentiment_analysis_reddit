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
from functools import partial
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


from datasets import DATASET


def extend_array(X1, X2):
    # Extend array X2 to X1 shape on axis 0
    num_tile = X1.shape[0] // X2.shape[0]
    num_remainder = X1.shape[0] % X2.shape[0]

    X2_extend = X2
    if num_tile > 1:
        if X2.ndim == 1:
            reps = num_tile
        else:
            reps = tuple([num_tile] +  [1] * len(X2.shape[1:]))
        X2_extend = np.tile(X2, reps)
    if num_remainder > 0:
        X2_extend = np.concatenate([X2_extend, X2[:num_remainder]], axis=0)

    return X2_extend

def adjust_size_unlabeled(X, X_unl=None, y=None, sample_weights=None):
    if not X_unl is None:
        if X.shape[0] > X_unl.shape[0]:
            X_unl = extend_array(X, X_unl)
        elif X.shape[0] < X_unl.shape[0]:
            if not y is None:
                assert X.shape[0] == y.shape[0]
                y = extend_array(X_unl, y)
            if not sample_weights is None:
                assert sample_weights.shape[0] == X.shape[0]
                sample_weights = extend_array(X_unl, sample_weights)

            X = extend_array(X_unl, X)

    return X, X_unl, y, sample_weights

def data_generator(X, X_unl=None, y=None, batch_size=128,
                    validation=False, sample_weights=None, fun_read_unlabeled=None,
                    fun_read_x=None):

    X_batch = []
    X_unl_batch = []
    y_batch = []
    w_batch = []

    if y is None and not sample_weights is None:
        raise ValueError("y should not be none if sample weight is provided")

    smallest_size = X.shape[0]
    biggest_size = X.shape[0]
    x_smallest = True
    if not X_unl is None:
        if X.shape[0] > X_unl.shape[0]:
            smallest_size =  X_unl.shape[0]
            x_smallest = False
        else:
            biggest_size = X_unl.shape[0]

    #X, X_unl, y, sample_weights = adjust_size_unlabeled(X, X_unl=X_unl, y=y, sample_weights=sample_weights)

    ind = np.arange(biggest_size)
    ind_smallest = np.arange(smallest_size)
    while True:

        if not validation:
            np.random.shuffle(ind)
            np.random.shuffle(ind_smallest)
        for idx_id, id in enumerate(ind):

            idx_id_smallest = idx_id % smallest_size

            if x_smallest:
                id_x = ind_smallest[idx_id_smallest]
                id_unl = id
            else:
                id_x = id
                id_unl = ind_smallest[idx_id_smallest]

            X_batch.append(X[id_x])
            if not y is None:
                y_batch.append(y[id_x])
            if not X_unl is None:
                X_unl_batch.append(X_unl[id_unl])
            if not sample_weights is None:
                w_batch.append(sample_weights[id_x])

            if len(X_batch) >= batch_size or (validation and idx_id == len(ind) - 1):
                if not fun_read_x is None:
                    batch = fun_read_x(X_batch)
                else:
                    batch = np.stack(X_batch, axis=0)

                if y is None and X_unl is None:
                    yield batch
                elif y is None:
                    yield [batch, np.stack(X_unl_batch, axis=0)]
                elif X_unl is None:
                    if sample_weights is None:
                        yield [batch, np.stack(y_batch, axis=0)], np.ones_like(np.stack(y_batch, axis=0))
                    else:
                        [batch, np.stack(y_batch, axis=0)], np.stack(w_batch, axis=0)
                else:
                    if not fun_read_unlabeled is None:
                        batch_unl = fun_read_unlabeled(X_unl_batch)
                    else:
                        batch_unl = np.stack(X_unl_batch, axis=0)
                    if sample_weights is None:
                        yield [batch, np.stack(y_batch, axis=0), batch_unl], np.ones_like(np.stack(y_batch, axis=0))
                    else:
                        yield [batch, np.stack(y_batch, axis=0), batch_unl], np.stack(w_batch, axis=0)


                X_batch = []
                y_batch = []
                X_unl_batch = []
                w_batch = []


def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_path', help='Path to save trained model to', default="lstm.{epoch:02d}-{val_loss:.2f}.hdf5", type=str)
    parser.add_argument('--load_model', help='Model to load to start training', default=None, type=str)
    parser.add_argument('--lr', help='Learning rate', default=1e-3, type=float)
    parser.add_argument('--batch_size', help='Batch size', default=256, type=int)
    parser.add_argument('--num_hidden', help='Number of hidden dimension for lstm', default=32, type=int)
    parser.add_argument('--num_epochs', help='Number of epochs', default=50, type=int)
    parser.add_argument('--num_cat', help='Number of categories', default=2, type=int)
    parser.add_argument('--start_epoch', help='Starting epoch', default=0, type=int)
    parser.add_argument('--patience', help='Number of epoch without improvement before reducing lr', default=3, type=int)
    parser.add_argument('--use_generator', help='Use generator during training', action="store_true", default=False)
    parser.add_argument('--use_at', default=False, action='store_true')
    parser.add_argument('--use_vat', default=False, action='store_true')
    parser.add_argument('--save_best_only', default=False, action='store_true', help="Save weights with lower loss only")
    parser.add_argument('--kfold_training', type=int, help="Use Kfold training")
    parser.add_argument('--transfer', type=str, choices=DATASET, help="transfer learning from dataset")
    parser.add_argument('--trainable_embedding', default=False, action='store_true')
    parser.add_argument('--save_embedding', default=None, help="Save embedding matrix", type=str)
    parser.add_argument('--use_embedding', default=None, help="Load embedding matrix", type=str)
    parser.add_argument('--dataset', required=True, choices=DATASET,
                            help="chosen dataset", type=str)
    return parser

def get_sample_weight(y_train):
    y_reduced = np.argmax(y_train, axis=-1)
    classes=np.unique(y_reduced)
    class_weight = compute_class_weight(class_weight="balanced",
                                        classes=classes, y=y_reduced)
    dict_weight = {}
    for idx, w in enumerate(class_weight):
        dict_weight[classes[idx]] = w

    sample_weight = compute_sample_weight(dict_weight, y_reduced)

    return sample_weight



def kfold_train(num_split, args, len_seq, vocabulary_size, use_vat,
                X_train, y_train, X_train_unl, X_test, y_test, sample_weight):

    sss = StratifiedKFold(n_splits=num_split, shuffle=True, random_state=42)

    run = KFold_LSTM_Model(num_split, len_seq, vocabulary_size, 300, args.num_hidden, args.num_cat,
        embedding_matrix=embedding_matrix, trainable_embedding=args.trainable_embedding,
            out_latent=False, use_at=args.use_at, use_vat=use_vat, lr=args.lr, use_weights=True)

    iter = 0
    base_name = args.save_path[:-5]
    for train_ind, val_ind in sss.split(X_train, np.argmax(y_train, axis=-1)):

        x_fold_train = X_train[train_ind]
        y_fold_train = y_train[train_ind]
        sw_fold_train = sample_weight[train_ind]

        x_fold_val = X_train[val_ind]
        y_fold_val = y_train[val_ind]

        args.save_path = base_name + "_fold_{}.hdf5".format(iter)

        train_model(args, run[iter], use_vat, x_fold_train, y_fold_train, X_train_unl,
                    x_fold_val, y_fold_val, sw_fold_train)

        iter += 1
        print("Done iteration {} in {} Kfold split".format(iter, num_split))



    print("Evaluate kfold on test set")

    loss, accuracy, loss_cat = run.evaluate([X_test, y_test, X_test], np.ones_like(y_test), batch_size=args.batch_size, verbose=1)
    print("loss {} - accuracy {} - loss_cat {}".format(loss, accuracy, loss_cat))

def train_model(args, run, use_vat, X_train, y_train, X_train_unl, X_val, y_val, sample_weight):


    if not args.load_model is None:
        run.load_weights(args.load_model, by_name=True)


    callbacks = [keras.callbacks.EarlyStopping(patience=8, monitor="val_metric_loss_cat"),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=args.patience,
                                                                monitor="val_metric_loss_cat", verbose=1),
                keras.callbacks.ModelCheckpoint(args.save_path,
                    save_weights_only=True, save_best_only=args.save_best_only, monitor="val_metric_loss_cat")]

    steps_per_epoch = X_train.shape[0] // args.batch_size + 1
    val_steps = X_val.shape[0] // args.batch_size
    if val_steps % args.batch_size > 0:
        val_steps += 1

    if not args.use_generator:
        if use_vat:
            X_train, X_train_unl, y_train, sample_weight = adjust_size_unlabeled(X_train, X_unl=X_train_unl,
                                                        y=y_train, sample_weights=sample_weight)

            train_data = [X_train, y_train, X_train_unl]
            val_data = ([X_val, y_val, X_val], np.ones_like(y_val))
        else:
            train_data = [X_train, y_train]
            val_data = ([X_val, y_val], np.ones_like(y_val))

        history = run.fit(train_data, y=sample_weight, batch_size=args.batch_size,
                        epochs=args.num_epochs,
                       validation_data=val_data,
                        initial_epoch=args.start_epoch,
                        callbacks=callbacks)
    else:
        train_gen = data_generator(X_train, y=y_train, X_unl=(X_train_unl if use_vat else None),
                                sample_weights=sample_weight, batch_size=args.batch_size,
                                fun_read_unlabeled=fun_read_unlabeled)
        val_gen = data_generator(X_val, y=y_val, X_unl=(X_val if use_vat else None),
                                    batch_size=args.batch_size, validation=True)

        history = run.fit_generator(train_gen, steps_per_epoch=steps_per_epoch,
                        epochs=args.num_epochs,
                       validation_data=val_gen, validation_steps=val_steps,
                        initial_epoch=args.start_epoch,
                        callbacks=callbacks)


if __name__ == "__main__":

    parser = common_arg_parser()

    args = parser.parse_args()

    dataset = importlib.import_module(args.dataset)
    text_train, text_val, label_train, label_val, text_unlabeled, fun_read_unlabeled = dataset.read_dataset()

    tokenizer = None
    if args.transfer:
        tokenizer, vocabulary_size = pickle.load(open(args.transfer + ".pickle", "rb"))
    print("dataset read")

    get_max_len_seq_unl = None
    if not fun_read_unlabeled is None:
        X_unl = text_unlabeled
        text_unlabeled = None
        get_max_len_seq_unl = partial(dataset.get_max_len_seq, datasetname=args.dataset)

    X_train, X_val, y_train, y_val, X_train_unl, tokenizer, vocabulary_size = preprocessing(text_train, text_val,
                                                                        label_train, label_val,
                                                                        text_unlabeled, tokenizer=tokenizer,
                                                                        get_max_len_seq_unl=get_max_len_seq_unl)
    len_seq = X_train.shape[1]

    if not fun_read_unlabeled is None:
        X_train_unl = X_unl
        fun_read_unlabeled = partial(fun_read_unlabeled, tokenizer=tokenizer, len_seq=len_seq)

    print("preprocessing done")
    sample_weight = get_sample_weight(y_train)
    pickle.dump([tokenizer, vocabulary_size], open(args.dataset + ".pickle", "wb"))

    if not args.use_embedding is None:
        embedding_matrix = np.load(args.use_embedding)
    elif not args.load_model is None:
        embedding_matrix = None
    else:
        embedding_matrix = embedding_matrix_word2vec(tokenizer, vocabulary_size, normalize=True)

        if not args.save_embedding is None:
            np.save(args.save_embedding, embedding_matrix)

    print("embedding done")


    use_vat = args.use_vat and not X_train_unl is None
    if not args.kfold_training:
        run = LSTM_Model(len_seq, vocabulary_size, 300, args.num_hidden, args.num_cat,
        embedding_matrix=embedding_matrix, trainable_embedding=args.trainable_embedding,
            out_latent=False, use_at=args.use_at, use_vat=use_vat, lr=args.lr, use_weights=True)

        train_model(args, run, use_vat, X_train, y_train, X_train_unl, X_val, y_val, sample_weight)
    else:
        kfold_train(args.kfold_training, args, len_seq, vocabulary_size, use_vat,
                X_train, y_train, X_train_unl, X_val, y_val, sample_weight)
