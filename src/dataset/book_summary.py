from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import os
import os
import urllib.request
import tarfile
import pandas as pd
import os   
import json
from sklearn.model_selection import train_test_split


def get_book_summary():
    url = "http://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz"
    file_path = "data/booksummaries.tar.gz"
    urllib.request.urlretrieve(url, file_path)

    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall("data")
    os.remove(file_path)
    
    

def book_summaries(seed, pairs, is_small=False):
    book_path='data/booksummaries/booksummaries.txt'
    if not os.path.exists(book_path):
        get_book_summary()

    text_set = {'train': [], 'dev': [], 'test': []}
    label_set = {'train': [], 'dev': [], 'test': []}
    train, dev, test = load_booksummaries_data(book_path)

    if not pairs:
        text_set['train'] = train['summary'].tolist()
        text_set['dev'] = dev['summary'].tolist()
        text_set['test'] = test['summary'].tolist()

        train_genres = train['genres'].tolist()
        label_set['train'] = [list(genre.values()) for genre in train_genres]
        dev_genres = dev['genres'].tolist()
        label_set['dev'] = [list(genre.values()) for genre in dev_genres]
        test_genres = test['genres'].tolist()
        label_set['test'] = [list(genre.values()) for genre in test_genres]
    else:
        train_temp = train['summary'].tolist()
        dev_temp = dev['summary'].tolist()
        test_temp = test['summary'].tolist()

        train_genres = train['genres'].tolist()
        train_genres_temp = [list(genre.values()) for genre in train_genres]
        dev_genres = dev['genres'].tolist()
        dev_genres_temp = [list(genre.values()) for genre in dev_genres]
        test_genres = test['genres'].tolist()
        test_genres_temp = [list(genre.values()) for genre in test_genres]

        for i in range(0, len(train_temp) - 1, 2):
            text_set['train'].append(train_temp[i] + train_temp[i+1])
            label_set['train'].append(list(set(train_genres_temp[i] + train_genres_temp[i+1])))

        for i in range(0, len(dev_temp) - 1, 2):
            text_set['dev'].append(dev_temp[i] + dev_temp[i+1])
            label_set['dev'].append(list(set(dev_genres_temp[i] + dev_genres_temp[i+1])))

        for i in range(0, len(test_temp) - 1, 2):
            text_set['test'].append(test_temp[i] + test_temp[i+1])
            label_set['test'].append(list(set(test_genres_temp[i] + test_genres_temp[i+1])))

    train_df, val_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    train_df["text"] = text_set["train"]
    train_df["label"] = label_set["train"]
    val_df["text"] = text_set["dev"]
    val_df["label"] = label_set["dev"]
    test_df["text"] = text_set["test"]
    test_df["label"] = label_set["test"]
    
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
    
    return cats, train_df["text"].tolist(), label_bin_train, id_train, val_df["text"].tolist(), label_bin_val, id_val,  test_df["text"].tolist(), label_bin_test, id_test

    
    
def load_booksummaries_data(book_path):
    """
    Load the Book Summary data and split it into train/dev/test sets
    :param book_path: path to the booksummaries.txt file
    :return: train, dev, test as pandas data frames
    """
    book_df = pd.read_csv(book_path, sep='\t', names=["Wikipedia article ID",
                                                      "Freebase ID",
                                                      "Book title",
                                                      "Author",
                                                      "Publication date",
                                                      "genres",
                                                      "summary"],
                          converters={'genres': parse_json_column})
    book_df = book_df.dropna(subset=['genres', 'summary']) # remove rows missing any genres or summaries
    book_df['word_count'] = book_df['summary'].str.split().str.len()
    book_df = book_df[book_df['word_count'] >= 10]
    train = book_df.sample(frac=0.8, random_state=22)
    rest = book_df.drop(train.index)
    dev = rest.sample(frac=0.5, random_state=22)
    test = rest.drop(dev.index)
    return train, dev, test


def parse_json_column(genre_data):
    """
    Read genre information as a json string and convert it to a dict
    :param genre_data: genre data to be converted
    :return: dict of genre names
    """
    try:
        return json.loads(genre_data)
    except Exception as e:
        return None # when genre information is missing

