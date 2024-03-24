# script from https://github.com/allenai/longformer/tree/master/scripts

from src.dataset.metrics_dataset import print_stats, splits_distribution
import pandas as pd
import os
import jsonlines
import tqdm
from datasets import load_dataset


def hyperpartisan(seed):
    dataset = load_dataset('jonathanli/hyperpartisan-longformer-split')  
    train_df = pd.DataFrame({"id": dataset["train"]["id"], "text": dataset["train"]["text"], "label": dataset["train"]["label"]}).sample(frac=1, random_state=seed)
    val_df = pd.DataFrame({"id": dataset["validation"]["id"], "text": dataset["validation"]["text"], "label": dataset["validation"]["label"]}).sample(frac=1, random_state=seed)
    test_df = pd.DataFrame({"id": dataset["test"]["id"], "text": dataset["test"]["text"], "label": dataset["test"]["label"]}).sample(frac=1, random_state=seed)
    
    train_df['label'] = train_df['label'].map({'true': 1, 'false': 0})   
    val_df['label'] = val_df['label'].map({'true': 1, 'false': 0})    
    test_df['label'] = test_df['label'].map({'true': 1, 'false': 0})    
 
    id2cat = {0: "false", 1: "true"}   

    return "hyperpartisan",  id2cat, train_df["text"].tolist(), train_df["label"].tolist(), train_df["id"].tolist(), val_df["text"].tolist(), val_df["label"].tolist(), val_df["id"].tolist(), test_df["text"].tolist(), test_df["label"].tolist(), test_df["id"].tolist(),
    

