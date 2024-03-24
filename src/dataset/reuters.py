from sklearn.preprocessing import MultiLabelBinarizer
import nltk
nltk.download('reuters')
import pandas as pd
from nltk.corpus import reuters
from sklearn.model_selection import train_test_split
from src.dataset.metrics_dataset import splits_distribution, print_stats

def reuters_news(seed):
    fileids = reuters.fileids()
    splits = [el.split("/")[0] for el in fileids]

    categories, text = [], []
    for file in fileids:
        categories.append(reuters.categories(file))
        text.append(reuters.raw(file))
    df = pd.DataFrame({'ids':fileids, 'split': splits, 'categories':categories, 'text':text})
    train_df = df[df["split"]== "training"]
    test_df = df[df["split"]== "test"]
    train_df, val_df = train_test_split(train_df, test_size = 0.2, random_state=seed)

    mlb = MultiLabelBinarizer()
    mlb.fit(test_df['categories'].tolist())
    
    cats = {}
    for idx, cat in enumerate(list(mlb.classes_)):
        cats[idx] = cat
        
        
    splits_distribution(train_df['categories'].tolist(), val_df['categories'].tolist(), test_df['categories'].tolist(), "multilabel" )


    label_bin_train = mlb.transform(train_df['categories'].tolist())
    label_bin_val = mlb.transform(val_df['categories'].tolist())
    label_bin_test = mlb.transform(test_df['categories'].tolist())
    
    print_stats(train_df['text'].tolist(), val_df['text'].tolist(), test_df['text'].tolist(), cats)

    
    return "reuters", cats, train_df["text"].tolist(), label_bin_train, train_df["ids"].tolist(), val_df["text"].tolist(), label_bin_val, val_df["ids"].tolist(),  test_df["text"].tolist(), label_bin_test, test_df["ids"].tolist()
