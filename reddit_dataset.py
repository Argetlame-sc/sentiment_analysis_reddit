import pandas as pd
import numpy as np
import argparse
from preprocessing import filter_markdown_quoting, preprocessing_infer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import json
import os
import sqlite3

def get_max_len_seq(tokenizer, datasetname="dataset", db_name="comments_cached.db"):
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS len_sequence (name TEXT, length INT, PRIMARY KEY(name))")
    conn.commit()

    req = "SELECT length FROM len_sequence WHERE name = ?"

    result = conn.execute(req, (datasetname,)).fetchone()
    if not result is None:
        conn.close()
        return result[0]
    else:

        def get_max_len(max_len, txt):
            x, _ = preprocessing_infer(txt, tokenizer)

            if x.shape[1] > max_len:
                max_len = x.shape[1]

            return max_len

        max_len = 0
        count = 0
        txt = []
        for row in conn.execute("SELECT body FROM comments"):
            txt.append(row["body"])

            if len(txt) >= 300000:
                max_len = get_max_len(max_len, txt)
                count += len(txt)
                txt = []
                print("max seq len compute : count done is {}".format(count), end="\r")

        if len(txt) > 0:
            max_len = get_max_len(max_len, txt)
            count += len(txt)
            print("max seq len compute : count done is {}".format(count), end="\r")
            print()

        conn.execute("INSERT INTO len_sequence(name, length) VALUES (?, ?)", (datasetname, max_len))
        conn.commit()
        conn.close()

    return max_len

def insert_comment(id, com, conn, cols_table):

    cols = ["ID_COMMENT"] + [col for col in com.keys() if col in cols_table and not type(com[col]) in [list, dict]]
    req = "INSERT INTO comments(" + ", ".join(list(cols)) + ") VALUES(" + ", ".join(["?"] * len(cols)) + ")"
    t = tuple([id] + [com[col] for col in cols if col != "ID_COMMENT"])
    conn.execute(req, t)

def save_comments(comments_list, append=False, db_name="comments_cached.db"):

    conn = sqlite3.connect(db_name)

    def get_type(a):
        if type(a) == str:
            return "TEXT"
        return "NUMERIC"

    com = comments_list[0]
    cols_table = [col for col in com.keys() if not type(com[col]) in [list, dict]]

    conn.execute("CREATE TABLE IF NOT EXISTS comments ( ID_COMMENT INT,"
                + ", ".join([col + " " + get_type(c[col]) for col in cols_table])
                + ", PRIMARY KEY(name), UNIQUE(ID_COMMENT))")

    offset_idx = 0
    if not append:
        conn.execute("DELETE FROM comments")
    else:
        result = conn.execute("SELECT ID_COMMENT FROM comments ORDER BY ID_COMMENT DESC LIMIT 1").fetchone()
        if not result is None:
            offset_idx = result[0]

    for idx, c in enumerate(comments_list):
        insert_comment(idx + offset_idx, c, conn, cols_table)

    conn.commit()
    conn.close()

def load_all_comments(only=None, db_name="comments_cached.db"):

    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row

    comments_list = []
    for row in conn.execute("SELECT * FROM comments ORDER BY ID_COMMENT ASC"):
        append_only_cols(comments_list, row, only)


    conn.close()
    return comments_list

def append_only_cols(comments_list, row, only):
    if only is None or len(only) == 0:
        com = dict(zip(row.keys(), tuple(row)))
    else:
        if type(only) == str:
            com = row[only]
        elif type(only) == list:
            com = dict(zip(only, [row[c] for c in only]))
    comments_list.append(com)

def load_comments(idx_list, only=None, db_name="comments_cached.db"):

    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row

    comments_list_raw = []
    ind_sorted = np.argsort(idx_list)
    comments_list = [None] * len(idx_list)

    for row in conn.execute("SELECT * FROM comments WHERE ID_COMMENT IN ("
                    + ", ".join([str(i) for i in idx_list]) + ") ORDER BY ID_COMMENT ASC"):
        append_only_cols(comments_list_raw, row, only)
    conn.close()

    for idx, c in enumerate(comments_list_raw):
        comments_list[ind_sorted[idx]] = c

    return comments_list

def load_comments_id(db_name="comments_cached.db"):
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row

    comments_id = []
    for row in conn.execute("SELECT ID_COMMENT FROM comments ORDER BY ID_COMMENT ASC"):
        comments_id.append(row["ID_COMMENT"])

    conn.close()
    return np.array(comments_id)

def set_pos_review(df, pos_value):
    df_pos = df[df.loc[:, 'review'] == 2]
    df_pos.loc[:, 'review'] = pos_value
    df = pd.concat([df_pos, df[df.loc[:, 'review'] == 0]])

    return df

def read_dataset():

    df_train = pd.read_csv("sc_dataset_train.csv")
    df_val = pd.read_csv("sc_dataset_val.csv")

    #df_train = set_pos_review(df_train, 1)
    #df_val = set_pos_review(df_val, 1)

    X_train = list(df_train.loc[:, "text"])
    y_train = list(df_train.loc[:, "review"])

    X_val = list(df_val.loc[:, "text"])
    y_val = list(df_val.loc[:, "review"])

    texts = load_comments_id(db_name="comments_cached.db")
    id_avoid = set(list(df_val.loc[:, "id"]))
    X_unlabeled = []

    for idx in texts:
        if not idx in id_avoid:
            X_unlabeled.append(idx)

    def fun_read_unlabeled(batch_ids, tokenizer=None, len_seq=None):
        comments = load_comments(batch_ids, only="body", db_name="comments_cached.db")
        batch_x_unl, _ = preprocessing_infer(comments, tokenizer, max_len=len_seq)

        return batch_x_unl


    return X_train, X_val, y_train, y_val, np.array(X_unlabeled), fun_read_unlabeled

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--reindex_dataset', default=False, action='store_true')
    parser.add_argument('--split_train_val', default=None, type=float, help="Split dataset into train and val.\
                                            Proportion of dataset to be used as val, must be between 0 and 1")
    parser.add_argument('--dataset', default="sc_dataset.csv", help="dataset csv file for reddit comments")
    parser.add_argument('--comments_file', default="comments_cached.db", help="file of extracted comments from reddit")
    return parser

if __name__ == "__main__":

    parser = common_arg_parser()
    args = parser.parse_args()

    texts = load_all_comments(db_name=args.comments_file, only="body")
    print("loaded comments")
    if args.dataset[-4:] != ".csv":
        raise ValueError("{} is not a valid csv name".format(args.dataset))

    df_dataset = pd.read_csv(args.dataset)

    if args.reindex_dataset:
        txt_to_idx = {}

        for idx, txt in enumerate(texts):
            txt_to_idx[txt] = idx
        del texts

        num_failed = 0
        col_indices = []
        for idx, row in df_dataset.iterrows():
            txt = row[1]

            try:
                indice = txt_to_idx[txt]
                col_indices.append(indice)
            except:
                #print("Failed to retrieve comment {}".format(txt))
                col_indices.append(-1)
                num_failed += 1

        print("Failed to retrieve {} comments".format(num_failed))
        df_dataset.loc[:, "id"] = col_indices

        df_dataset.to_csv(args.dataset, index=False)

    if args.split_train_val:

        if args.split_train_val < 0 or args.split_train_val > 1:
            raise ValueError("split_train_val : {} should be between 0 and 1".format(args.split_train_val))

        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.split_train_val)

        x = list(df_dataset.loc[:, "text"])
        y = list(df_dataset.loc[:, "review"])

        train_idx, val_idx = next(sss.split(x, y))

        df_train = df_dataset.iloc[train_idx, :]
        df_val = df_dataset.iloc[val_idx, :]

        df_train.to_csv(args.dataset[:-4] + "_train.csv", index=False)
        df_val.to_csv(args.dataset[:-4] + "_val.csv", index=False)
