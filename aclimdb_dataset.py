import os
import numpy as np

def add_label(label_list, cat):
    if cat == "neg":
        label_list.append(0)
    elif cat == "pos":
        label_list.append(1)
    else:
        raise ValueError("Category {} is not valid".format(cat))

def read_dataset():
    #0 is negative reviex and 1 is positive review

    text_train = []
    text_val = []

    label_train = []
    label_val = []

    text_unlabeled = []

    for entry in os.scandir("aclImdb"):
        if entry.is_dir():
            name = entry.name

            for entry_cat in os.scandir(entry.path):
                if entry_cat.is_dir() and (entry_cat.name == "pos" or entry_cat.name == "neg"):
                    cat = entry_cat.name
                    for entry_text in os.scandir(entry_cat.path):
                        if entry_text.is_file() and entry_text.name.endswith(".txt"):

                            content = open(entry_text.path).read()
                            if name == "train":
                                text_train.append(content)
                                add_label(label_train, cat)
                            elif name == "test":
                                text_val.append(content)
                                add_label(label_val, cat)
                            else:
                                raise ValueError("Name {} is not a valid folder".format(name))
                elif entry_cat.is_dir() and entry_cat.name == "unsup":
                    for entry_text in os.scandir(entry_cat.path):
                        if entry_text.is_file() and entry_text.name.endswith(".txt"):

                            content = open(entry_text.path).read()
                            text_unlabeled.append(content)



    text_train = text_train +  text_val[:11500] + text_val[12500:-1000]
    text_val = text_val[11500:12500] + text_val[-1000:]

    label_train = label_train +  label_val[:11500] + label_val[12500:-1000]
    label_val = label_val[11500:12500] + label_val[-1000:]

    text_unlabeled += text_train

    return text_train, text_val, label_train, label_val, text_unlabeled, None


if __name__ == "__main__":

    text_train, text_val, label_train, label_val = read_dataset()

    text_train_concat = "\n".join(text_train)
    text_val_concat = "\n".join(text_val)

    with open("imdb_train.txt", "w") as f:
        f.write(text_train_concat)
    with open("imdb_val.txt", "w") as f:
        f.write(text_val_concat)

    np.savez_compressed("imdb_label_train.npz", np.stack(label_train, axis=0))
    np.savez_compressed("imdb_label_val.npz", np.stack(label_val, axis=0))
