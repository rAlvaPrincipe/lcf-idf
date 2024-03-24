import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from src.dataset.metrics_dataset import splits_distribution
from src.classifier.dataloader import cat2id2cat


def scotus(seed, is_small):
    dataset = load_dataset('lex_glue', 'scotus') 
    train_df = pd.DataFrame({"text": dataset["train"]["text"], "label": dataset["train"]["label"]})
    val_df = pd.DataFrame({"text": dataset["validation"]["text"], "label": dataset["validation"]["label"]})
    test_df = pd.DataFrame({"text": dataset["test"]["text"], "label": dataset["test"]["label"]})
    
    if is_small:
        _, train_df = train_test_split(train_df, stratify=train_df["label"], test_size = 0.15, random_state=seed)
        _, val_df = train_test_split(val_df, stratify=val_df["label"],  test_size = 0.2, random_state=seed)
        _, test_df = train_test_split(test_df, stratify=test_df["label"], test_size = 0.2, random_state=seed)

    cats = list(set(test_df["label"].tolist()))
    
    # convert original labels into numbers
    cat2id, id2cat = cat2id2cat(cats)
    train_df["label"] = [cat2id[el] for el in train_df["label"].tolist()]
    val_df["label"]= [cat2id[el] for el in val_df["label"].tolist()]
    test_df["label"] = [cat2id[el] for el in test_df["label"].tolist()]
    
    train_df["id"] = [0] * train_df.shape[0]
    val_df["id"] = [0] * val_df.shape[0]
    test_df["id"] = [0] * test_df.shape[0]
    
    print(train_df)
    print(val_df)
    print(test_df)
    splits_distribution(train_df["label"].tolist(), val_df["label"].tolist(), test_df["label"].tolist(), "multiclass")
    print(id2cat)
    
    return id2cat, train_df["text"].tolist(), train_df["label"].tolist(), train_df["id"].tolist(), val_df["text"].tolist(), val_df["label"].tolist(), val_df["id"].tolist(),  test_df["text"].tolist(), test_df["label"].tolist(), test_df["id"].tolist()



def scotus_small(seed):
    out = ("scotus_small", ) + scotus( seed, True)
    return out