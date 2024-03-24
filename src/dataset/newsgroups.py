from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import pandas as pd
from src.dataset.metrics_dataset import splits_distribution
from src.classifier.dataloader import cat2id2cat

def newsgroups(seed, is_small):
    test_set = fetch_20newsgroups(subset='test')
    x_test= [text for text in test_set.data]
    y_test = test_set.target
    id_test = [filename[len("/home/soperga/scikit_learn_data/20news_home/"):] for filename in test_set.filenames]

    train_set = fetch_20newsgroups(subset='train')
    x_train = [text for text in train_set.data]
    y_train = train_set.target
    id_train = [filename[len("/home/soperga/scikit_learn_data/20news_home/20news-bydate-train/"):] for filename in train_set.filenames]
    
    cats = test_set.target_names
    y_train = [cats[el] for el in y_train]
    y_test = [cats[el] for el in y_test]
    
    train_df = pd.DataFrame({"id": id_train, "text": x_train, "label": y_train})
    test_df = pd.DataFrame({"id": id_test, "text": x_test, "label": y_test})
    
    train_df, val_df = train_test_split(train_df, stratify=train_df["label"], test_size=0.1, random_state=seed)   
    _, train_df = train_test_split(train_df, stratify=train_df["label"], test_size=0.2, random_state=seed)          
    _, test_df = train_test_split(test_df, stratify=test_df["label"], test_size=0.15, random_state=seed)
    
    
    # convert original labels into numbers
    cat2id, id2cat = cat2id2cat(cats)
    train_df["label"] = [cat2id[el] for el in train_df["label"].tolist()]
    val_df["label"]= [cat2id[el] for el in val_df["label"].tolist()]
    test_df["label"] = [cat2id[el] for el in test_df["label"].tolist()]
    
    print(train_df)
    print(val_df)
    print(test_df)
    splits_distribution(train_df["label"].tolist(), val_df["label"].tolist(), test_df["label"].tolist(), "multiclass")
    print(id2cat)
    return id2cat, train_df["text"].tolist(), train_df["label"].tolist(), train_df["id"].tolist(), val_df["text"].tolist(), val_df["label"].tolist(), val_df["id"].tolist(),  test_df["text"].tolist(), test_df["label"].tolist(), test_df["id"].tolist()



def newsgroups_small(seed):
    out = ("newsgroups_small", ) + newsgroups( seed, True)
    return out

