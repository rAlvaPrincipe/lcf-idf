from datasets import list_datasets, load_dataset
from pprint import pprint
from sklearn.preprocessing import MultiLabelBinarizer
from src.dataset.metrics_dataset import splits_distribution, print_stats
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def ecthr(seed, is_small=False):
    dataset = load_dataset('lex_glue', 'ecthr_b')  
    texts = [" ".join(text) for text in dataset["train"]["text"]]
    train_df = pd.DataFrame({"text": texts, "label": dataset["train"]["labels"]})
    
    texts = [" ".join(text) for text in dataset["validation"]["text"]]
    val_df = pd.DataFrame({"text": texts, "label": dataset["validation"]["labels"]})
    
    texts = [" ".join(text) for text in dataset["test"]["text"]]
    test_df = pd.DataFrame({"text": texts, "label": dataset["test"]["labels"]})
    
    mlb = MultiLabelBinarizer()
    mlb.fit(train_df["label"].tolist())
    
    if is_small:
        _, train_df = train_test_split(train_df, test_size = 0.20, random_state=seed)


    cats = {}
    for idx, cat in enumerate(list(mlb.classes_)):
        cats[idx] = str(cat)
        
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


def ecthr_full(seed):
    out = ("ecthr", ) + ecthr( seed, False)
    return out


def ecthr_small(seed):
    out = ("ecthr_small", ) + ecthr( seed, True)
    return out

