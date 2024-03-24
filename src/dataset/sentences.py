from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from src.classifier.dataloader import cat2id2cat
from src.dataset.metrics_dataset import print_stats, splits_distribution
import hashlib

# https://huggingface.co/datasets/financial_phrasebank


# MULTICLASS
# classi: 3
# train: 1584
# val: 340
# test 340
# train avg #words:  22.36
# word in vector space: 35.418
def sentences(seed):
    dataset = load_dataset('financial_phrasebank', 'sentences_allagree')  
    
    train_df = pd.DataFrame({"sentence": dataset["train"]["sentence"], "label": dataset["train"]["label"]})
    x_train = train_df["sentence"].tolist()
    y_train = train_df["label"].tolist()

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,stratify=y_train, test_size=0.3, random_state=seed)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,stratify=y_test, test_size=0.5, random_state=seed)

    id_train = [0] * len(x_train)
    id_val = [0] * len(x_val)
    id_test = [0] * len(x_test)
    
    cats = list(set(y_test))
    y_train = [cats[el] for el in y_train]
    y_val= [cats[el] for el in y_val]
    y_test = [cats[el] for el in y_test]
    
    print("HASH", hashlib.md5(' '.join(map(str,x_train + x_val + x_test)).encode('utf-8')).hexdigest())
    
    
    splits_distribution(y_train, y_val, y_test, "multiclass")
    
   # convert original labels into numbers
    cat2id, id2cat = cat2id2cat(cats)
    y_train = [cat2id[el] for el in y_train]
    y_val = [cat2id[el] for el in y_val]
    y_test = [cat2id[el] for el in y_test]

    print_stats(x_train, x_val, x_test, id2cat)

    return "sentences", id2cat, x_train, y_train, id_train, x_val, y_val, id_val, x_test, y_test, id_test