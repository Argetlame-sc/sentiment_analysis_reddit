import pandas as pd
import numpy as np


#0 is negative for a label, 1 is positive
def read_csv(X, label, df):
    for idx, row in df.iterrows():
        if row[0] == 0:
            label.append(0)
        elif row[0] == 4:
            label.append(1)
        else:
            continue

        X.append(row[5])

def read_dataset():

    X_train = []
    X_val = []

    label_train = []
    label_val = []

    df_train = pd.read_csv("sentiment140/training.1600000.processed.noemoticon.csv", encoding = "ISO-8859-1", engine='python')
    print("read train csv")
    df_val = pd.read_csv("sentiment140/testdata.manual.2009.06.14.csv")

    read_csv(X_train, label_train, df_train)
    print("created X and label train")
    read_csv(X_val, label_val, df_val)

    return X_train, X_val, label_train, label_val, None, None
