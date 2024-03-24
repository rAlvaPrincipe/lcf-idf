import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from src.dataset.metrics_dataset import splits_distribution
from src.classifier.dataloader import cat2id2cat
from itertools import compress
import numpy as np
from collections import Counter


def get_cats(num_cats):
    if num_cats == 20:
        return ["ALTRA DOCUMENTAZIONE",  "ESTRATTO CONTO", "CORRISPONDENZA", "CONTO CORRENTE ORDINARIO", "VISURA CATASTALE", "LETTERA DI MESSA IN MORA", "REPORT-DOSSIER", "DECRETO INGIUNTIVO",
           "VISURA IPOTECARIA", "CTU", "PERIZIA INTERNA BANCA", "AVVISO D'ASTA", "ATTO DI PRECETTO", "ATTO DI PIGNORAMENTO", "PRECISAZIONE DI CREDITO", "MUTUO CHIROGRAFARIO", "APPUNTI", 
           "FIDEJUSSIONE OMNIBUS", "ATTO DI INTEVENTO",  "APERTURA DI CREDITO CHIROGRAFARIA"]
    elif num_cats == 19:
        return ["ESTRATTO CONTO", "CORRISPONDENZA", "CONTO CORRENTE ORDINARIO", "VISURA CATASTALE", "LETTERA DI MESSA IN MORA", "REPORT-DOSSIER", "DECRETO INGIUNTIVO",
           "VISURA IPOTECARIA", "CTU", "PERIZIA INTERNA BANCA", "AVVISO D'ASTA", "ATTO DI PRECETTO", "ATTO DI PIGNORAMENTO", "PRECISAZIONE DI CREDITO", "MUTUO CHIROGRAFARIO", "APPUNTI", 
           "FIDEJUSSIONE OMNIBUS", "ATTO DI INTEVENTO",  "APERTURA DI CREDITO CHIROGRAFARIA"]
    elif num_cats == 9:
        return ["VISURA CATASTALE", "LETTERA DI MESSA IN MORA", "DECRETO INGIUNTIVO", "VISURA IPOTECARIA", "AVVISO D'ASTA", "CTU", "PERIZIA INTERNA BANCA", "ATTO DI PRECETTO", "ATTO DI PIGNORAMENTO"]
    elif num_cats == 2:
        return ["CORRISPONDENZA", "LETTERA DI MESSA IN MORA"]
   
   
def protos2(seed, uniform):
    return protos(get_cats(2), seed, uniform, "protos2")


def protos9(seed, uniform):
    return protos(get_cats(9), seed, uniform, "protos9")


def protos19(seed, uniform):
    return protos(get_cats(19), seed, uniform, "protos19")


def protos20(seed, uniform):
    return protos(get_cats(20), seed, uniform, "protos20")


def protos_all(seed, uniform):
    return protos(None, seed, uniform, "protos_all")


def protos(cats, seed, uniform, version):
    df, id2cat  = read(cats, uniform)
    x_train, y_train, id_train, x_val, y_val, id_val, x_test,  y_test, id_test = split(df, seed)
    splits_distribution(y_train, y_val, y_test, "multiclass")
    return version, id2cat, x_train, y_train, id_train, x_val, y_val, id_val, x_test, y_test, id_test
    
    
def read(cats, uniform):
    df = pd.read_csv("./data/protos-1.csv")
    if not cats:
        counts =  df['Micro'].value_counts()
        cats = list()
        for cat in counts.keys():
            if counts[cat] > 10:
                cats.append(cat)
    df = df[df["Micro"].isin(cats)][["id", "Micro", "text"]]
    df['id'] = df['id'].astype(str)
    # remove document with only whitespaces and \n 
    df = df[df["text"].str.isspace()==False]
        
    if uniform:
        # find class with less samples
        classes_numerosities = list()
        for label in df.Micro.unique().tolist():
            classes_numerosities.append(df[df["Micro"] == label].shape[0])
        min_numerosity = min(classes_numerosities)

        # create uniform distributed dataset
        uniform_df = pd.DataFrame()
        for label in df.Micro.unique().tolist():
            df_label = shuffle(df[df["Micro"] == label])
            df_label = df_label.head(min_numerosity)
            uniform_df = uniform_df.append(df_label)
        df = shuffle(uniform_df)

    # convert original labels into numbers
    cat2id, id2cat = cat2id2cat(cats)
    df['Micro'] = df['Micro'].apply(lambda x: cat2id[x])
    return df, id2cat


def split(df, seed):
    x = df
    y = df['Micro'].to_frame()
    train, test, y_train, y_test = train_test_split(x, y,stratify=y, test_size=0.4, random_state=seed)
    test, val, y_test, y_val = train_test_split(test, y_test, stratify=y_test, test_size=0.5, random_state=seed)

    x_train, y_train, id_train = train["text"].tolist(), train["Micro"].tolist(), train["id"].tolist() 
    x_val, y_val, id_val = val["text"].tolist(), val["Micro"].tolist(), val["id"].tolist() 
    x_test, y_test, id_test = test["text"].tolist(), test["Micro"].tolist(), test["id"].tolist() 

    return x_train, y_train, id_train, x_val, y_val, id_val, x_test, y_test, id_test



# WIP
# Serve :
# un protos solo long text: sul dataset originale: consideri i doc con numero di subwords >n, consideri solo le classi con un minimo di istanze, (da valutare) considera se taglaire un po' di subsampling delle classi numerose
# un protos2, protos9, ecc senza filtri sulla lunghegzza
# un protos che: il protos long text magari è semplice perchè cmq bastano iprmi 512  di BERTper descriminare tra le classi long text ma magari non è abb se aggiungiamo delle classi "normali", ad esempio immagina
# che in un dataset long abbiamo corrispondenza, mutuo, avviso d'asta: BERt riesce cmq a dividere perchè bastano i p rimi 512..... ma se aggiungiamo lettera di messa in mora, BERT farà casino, ma magari CTFIDF riesce a cogliere
# qualcosa in più e quindi questa volta a fare la differenza
# def protos(cats, seed, uniform, version):
#     df = pd.read_csv("./data/protos-1.csv")
#     if not cats:
#         counts =  df['Micro'].value_counts()
#         cats = list()
#         for cat in counts.keys():
#             if counts[cat] > 10:
#                 cats.append(cat)

#     df = df[df["Micro"].isin(cats)][["id", "Micro", "text"]]
#     df['id'] = df['id'].astype(str)
#     # remove document with only whitespaces and \n 
#     df = df[df["text"].str.isspace()==False]
        
#     if uniform:
#         # find class with less samples
#         classes_numerosities = list()
#         for label in df.Micro.unique().tolist():
#             classes_numerosities.append(df[df["Micro"] == label].shape[0])
#         min_numerosity = min(classes_numerosities)

#         # create uniform distributed dataset
#         uniform_df = pd.DataFrame()
#         for label in df.Micro.unique().tolist():
#             df_label = shuffle(df[df["Micro"] == label])
#             df_label = df_label.head(min_numerosity)
#             uniform_df = uniform_df.append(df_label)
#         df = shuffle(uniform_df)

#     # convert original labels into numbers
#     cat2id, id2cat = cat2id2cat(cats)
#     df['Micro'] = df['Micro'].apply(lambda x: cat2id[x])


#     ############## TEMP
#     x_train = df['text'].to_list()
#     y_train = df['Micro'].to_list()
#     id_train = df['id'].to_list()

#     only_long = True
#     if only_long:
#         min_subwords = 500
#         _, _,  train_num_doc_subwords  = subwords_stats(x_train)
#         long_bools = list(np.array(train_num_doc_subwords) > min_subwords)
#         x_train = list(compress(x_train, long_bools))
#         y_train = list(compress(y_train, long_bools))
#         id_train = list(compress(id_train, long_bools))
#         x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(x_train, y_train, id_train, stratify=y_train, test_size=0.30, random_state=21)
#         x_test, x_val, y_test, y_val, id_test, id_val= train_test_split(x_test, y_test, id_test, stratify=y_test, test_size=0.50, random_state=21)

#     cat2id, id2cat = cat2id2cat(cats)
#     print_stats(x_train, x_val, x_test, id2cat)
#     splits_distribution(y_train, y_val, y_test, "multiclass")

    # convert original labels into numbers
    
    #y_train = [cat2id[el] for el in y_train]
    # y_val = [cat2id[el] for el in y_val]
    # y_test = [cat2id[el] for el in y_test]
    


    ############### split

    # x = df
    # y = df['Micro'].to_frame()
    # train, test, y_train, y_test = train_test_split(x, y,stratify=y, test_size=0.4, random_state=seed)
    # test, val, y_test, y_val = train_test_split(test, y_test, stratify=y_test, test_size=0.5, random_state=seed)

    # x_train, y_train, id_train = train["text"].tolist(), train["Micro"].tolist(), train["id"].tolist() 
    # x_val, y_val, id_val = val["text"].tolist(), val["Micro"].tolist(), val["id"].tolist() 
    # x_test, y_test, id_test = test["text"].tolist(), test["Micro"].tolist(), test["id"].tolist() 

    # ####################

    # splits_distribution(y_train, y_val, y_test, "multiclass")
    # return version, id2cat, x_train, y_train, id_train, x_val, y_val, id_val, x_test, y_test, id_test
    