from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import statistics
import requests

TOKENIZER_STATS_ENDPOINT = 'http://localhost:6002/analyze/docs'
TOKENIZER = 'bert-base-uncased'


def splits_distribution(y_train, y_val, y_test, task_type):
    if task_type == "multilabel" or task_type == "multilabel-topone":
        counts_train = Counter(c for clist in y_train for c in clist)
        counts_val = Counter(c for clist in y_val for c in clist)
        counts_test = Counter(c for clist in y_test for c in clist)
    elif task_type == "multiclass" or task_type == "binary":
        counts_train = Counter(y_train)
        counts_val = Counter(y_val)
        counts_test = Counter(y_test)
        
    counts_train = OrderedDict(counts_train.most_common())
    counts_val = OrderedDict(counts_val.most_common())
    counts_test = OrderedDict(counts_test.most_common())

    train_vals = list()
    val_vals = list()
    test_vals = counts_test.values()

    for cat in counts_test.keys():
        if cat in counts_train.keys():
            train_vals.append(counts_train[cat])
        else: 
            train_vals.append(0)
        if cat in counts_val:
            val_vals.append(counts_val[cat])
        else:
            val_vals.append(0)

    plt.bar(list(counts_test.keys()), train_vals)
    plt.savefig('train.png')
    plt.close()

    plt.bar(list(counts_test.keys()), val_vals)
    plt.savefig('val.png')
    plt.close()

    plt.bar(list(counts_test.keys()), test_vals)
    plt.savefig('test.png')
    plt.close()




def print_stats(x_train, x_val, x_test, id2cat):
    print("classi: ", len(id2cat))
    print("train: ", len(x_train))
    print("val: ",  len(x_val))
    print("test: ", len(x_test))
    print("train avg #words: ", statistics.mean([len(doc.split()) for doc in x_train]))
    print("train #words: ", sum([len(doc.split()) for doc in x_train]))
    print("val avg #words: ", statistics.mean([len(doc.split()) for doc in x_val]))
    print("val #words: ", sum([len(doc.split()) for doc in x_val]))
    print("test avg #words: ", statistics.mean([len(doc.split()) for doc in x_test]))
    print("test #words: ", sum([len(doc.split()) for doc in x_test]))
    # avg_num_subwords, num_corpus_subwords, train_num_doc_subwords = subwords_stats(x_train)
    # print("train avg #sub-words: ", avg_num_subwords)
    # print("train #sub-words: ", num_corpus_subwords)
    # avg_num_subwords, num_corpus_subwords, val_num_doc_subwords= subwords_stats(x_val)
    # print("val avg #sub-words: ", avg_num_subwords)
    # print("val #sub-words: ", num_corpus_subwords)
    # avg_num_subwords, num_corpus_subwords, test_num_doc_subwords = subwords_stats(x_test)
    # print("test avg #sub-words: ", avg_num_subwords)
    # print("test #sub-words: ", num_corpus_subwords)

    #return train_num_doc_subwords, val_num_doc_subwords, test_num_doc_subwords
    
    
def subwords_stats(corpus):
    r = requests.post(TOKENIZER_STATS_ENDPOINT, json={"docs": corpus, "tokenizer": TOKENIZER})
    resp = r.json()
    return resp["stats"]["avg_num_subwords"], resp["stats"]["num_corpus_subwords"], resp["stats"]["num_doc_subwords"]
    
    