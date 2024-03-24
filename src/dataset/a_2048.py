import pandas as pd
import os
import json
import os
import re
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from src.dataset.metrics_dataset import splits_distribution
from src.classifier.dataloader import cat2id2cat


def a_2048(seed, is_small):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lowercase=True)   
    
    ids, texts, labels = [], [], []
    for line in open("data/amazon_books/reviews_Books_5.json"):
        diz = json.loads(line)
        ids.append(diz["reviewerID"])
        texts.append(diz["reviewText"])
        labels.append(diz["overall"])

    n_subwords = []
    for sub_corpus  in [texts[i:i+1500] for i in range(0, len(texts), 1500)]:
        for tokens_list in tokenizer(sub_corpus)["input_ids"]:
            n_subwords.append(len(tokens_list))
    
    df = pd.DataFrame({"id": ids,"text": texts, "label":labels, "n_subwords": n_subwords})
    df['is_long'] = df['n_subwords'].apply(lambda x: True if x > 2048 else False)
    df = df[df['is_long'] == True]


    _, df = train_test_split(df, stratify=df["label"], test_size=0.04, random_state=seed)   

    train_df, test_df = train_test_split(df, stratify=df["label"], test_size=0.4, random_state=seed)    
    test_df, val_df = train_test_split(test_df, stratify=test_df["label"], test_size=0.5, random_state=seed)          #     
    
    cats = list(set(test_df["label"].tolist()))

    # convert original labels into numbers
    cat2id, id2cat = cat2id2cat(cats)
    train_df["label"] = [cat2id[el] for el in train_df["label"].tolist()]
    val_df["label"]= [cat2id[el] for el in val_df["label"].tolist()]
    test_df["label"] = [cat2id[el] for el in test_df["label"].tolist()]
    
    print(train_df)
    print(val_df)
    print(test_df)
    print(id2cat)
    splits_distribution(train_df["label"].tolist(), val_df["label"].tolist(), test_df["label"].tolist(), "multiclass")
    
    return id2cat, train_df["text"].tolist(), train_df["label"].tolist(), train_df["id"].tolist(), val_df["text"].tolist(), val_df["label"].tolist(), val_df["id"].tolist(),  test_df["text"].tolist(), test_df["label"].tolist(), test_df["id"].tolist()



def a_2048_small(seed):
    out = ("a_512_small", ) + a_2048( seed, True)
    return out
