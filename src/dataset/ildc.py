import pandas as pd
from sklearn.model_selection import train_test_split
from src.classifier.dataloader import cat2id2cat
from src.dataset.metrics_dataset import splits_distribution


def ildc(seed, is_small):
    df = pd.read_csv("./data/ILDC/ILDC_multi.csv")
    train_df = df[df["split"] == 'train'].sample(frac=1, random_state=seed)
    val_df = df[df["split"] == 'dev'].sample(frac=1, random_state=seed)
    test_df = df[df["split"] == 'test'].sample(frac=1, random_state=seed)
    
    if is_small:
        _, train_df = train_test_split(train_df, stratify=train_df["label"], test_size = 0.05, random_state=seed) # 0.05 è quello che supporta

    
    cats = list(set(test_df["label"].tolist()))
    print("AAAAA-.-------------")
    print(cats)
    
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


def ildc_small(seed):
    out = ("ildc_small", ) + ildc( seed, True)
    return out