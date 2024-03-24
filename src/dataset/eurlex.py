from datasets import list_datasets, load_dataset
from pprint import pprint
from sklearn.preprocessing import MultiLabelBinarizer
from src.dataset.metrics_dataset import splits_distribution, print_stats
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def eurlex(seed, is_small):
    dataset = load_dataset('lex_glue', 'eurlex') 
    train_df = pd.DataFrame({"text": dataset["train"]["text"], "label": dataset["train"]["labels"]})
    val_df = pd.DataFrame({"text": dataset["validation"]["text"], "label": dataset["validation"]["labels"]})
    test_df = pd.DataFrame({"text": dataset["test"]["text"], "label": dataset["test"]["labels"]})

    mlb = MultiLabelBinarizer()
    mlb.fit(train_df["label"].tolist())
    
    cats = {}
    for idx, cat in enumerate(list(mlb.classes_)):
        cats[idx] = str(cat)
    
    if is_small:
        _, train_df = train_test_split(train_df, test_size = 0.05, random_state=seed)
        _, val_df = train_test_split(val_df, test_size = 0.2, random_state=seed)
        _, test_df = train_test_split(test_df, test_size = 0.2, random_state=seed)

    label_bin_train = mlb.transform(train_df["label"].tolist())
    label_bin_val = mlb.transform(val_df["label"].tolist())
    label_bin_test = mlb.transform(test_df["label"].tolist())
  
    id_train = [0] * train_df.shape[0]
    id_val = [0] * val_df.shape[0]
    id_test = [0] * test_df.shape[0]
    
    print(train_df)
    print(val_df)
    print(test_df)

    return cats, train_df["text"].tolist(), label_bin_train, id_train, val_df["text"].tolist(), label_bin_val, id_val,  test_df["text"].tolist(), label_bin_test, id_test



def eurlex_full(seed):
    out = ("eurlex_full", ) + eurlex( seed, False)
    return out


def eurlex_small(seed):
    out = ("eurlex_small", ) + eurlex( seed, True)
    return out