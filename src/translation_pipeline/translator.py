    
from src.translation_pipeline.encoder import Encoder
from src.translation_pipeline.clusterizer import Clusterizer
from src.translation_pipeline.reductor import Reductor
import torch

class Translator:

    def __init__(self, tokenizer, embedder, base_dir, reduction, clustering_alg, n_clusters=None):
        self.tokenizer = tokenizer 
        self.embedder = embedder
        self.reduction = reduction
        self.encoder = Encoder(tokenizer, embedder, "cuda")
        self.base_dir = base_dir
            
        if self.reduction and len(self.reduction) > 0:
            self.reductor1 = Reductor(self.reduction[0]["alg"], self.reduction[0]["dim"], self.base_dir + "/embedder/translator/")
        if self.reduction and len(self.reduction) == 2:
            self.reductor2 = Reductor(self.reduction[1]["alg"], self.reduction[1]["dim"], self.base_dir + "/embedder/translator/")
              
        if clustering_alg == "kmeans":
            self.clusterizer = Clusterizer(clustering_alg, n_clusters)
        elif clustering_alg == "hdbscan":
            self.clusterizer = Clusterizer(clustering_alg) 
            

    def fit(self, train_corpus, val_corpus):
        '''
        STEP 1: traduce ogni doc della forma "Ciao, questo è un documento in italiano " in [12, 32, 4, 1, 65, 2, 127] dove ogni parola è tradotta in id di concetti.
                L'obiettivo è astrarre sia la sintassi che le piccole sfumature di significato delle parole (e.g., i numeri stiano nello stesso concetto  piuttosto che le coninugazione 
                del verbo essere o le date o i sinonimi), in altre parole passare da uno spazio continuo di rappresentazione dato dagli embeddings di BERT a uno spazio discreto dato dai 
                concetti (clusters). Il numero massimo di concetti che possiamo creare è il numero di parole nel corpus (questo però è inutile). Se i concetti sono troppo pochi avremo 
                dei gruppi troppo eterogenei e rumorosi fino al caso limite di un unico concetto per tutto.
        input:  corpus su cui fare fit, numero di cluster (ovvero il numero di termini del dizionario del traduttore)
        output: un kmeans fittato sul corpus che servirà per assegnare ad ogni vettore di word un cluster (quindi un concetto/termine nel volcabolario)
        '''
        
        print("\nFITTING...")
        print("--> corpus2wordembeddings (train_corpus)")
        
        embeddings = self.encoder.corpus2wordembeddings(train_corpus)      # get BERT word embeddings from train_corpus
        embeddings = embeddings.detach().to("cpu")

        if self.reduction and len(self.reduction) > 0:
            if self.reductor1.alg == "Autoencoder":
                print("--> corpus2wordembeddings (val_corpus")
                val_embeddings = self.encoder.corpus2wordembeddings(val_corpus)   # get BERT word embeddings from val_corpus
                val_embeddings = val_embeddings.detach().to("cpu")
                embeddings = self.reductor1.fit(embeddings, val_embeddings)
            else:
                embeddings = self.reductor1.fit(embeddings, None)
        if self.reduction and len(self.reduction) == 2:
            embeddings = self.reductor2.fit(embeddings, None)
           
        self.clusterizer.run(embeddings)                             # find synsets cluster




    def translate(self, corpus):
        translated_corpus = list()
        corpus_embeddings = list()
        corpus_lens = list()
        print("\nTRASLATING...")

        print("--> corpus2wordembeddings")
        for doc in corpus:
            embeddings = self.encoder.corpus2wordembeddings([doc])
            corpus_embeddings.append(embeddings)
            corpus_lens.append(embeddings.shape[0])
        corpus_embeddings = torch.cat(corpus_embeddings)

            
        if self.reduction and len(self.reduction) > 0:
            print("--> reductor1")
            corpus_embeddings = self.reductor1.trasform(corpus_embeddings)
        if self.reduction and len(self.reduction) == 2:
            print("--> reductor2")
            corpus_embeddings = self.reductor2.trasform(corpus_embeddings)
            
            
        print("--> translating to concepts")
        start = 0
        for doc_len in corpus_lens:
            end = start + doc_len
            fetta = corpus_embeddings[start:end]
            start = start + doc_len

            concepts = self.clusterizer.predict(fetta)
            concepts_str = ' '.join([str(concept) for concept in concepts])
            translated_corpus.append(concepts_str)
        return translated_corpus



    def save(self):
        self.clusterizer.save(self.base_dir + "/embedder/translator/clusterizer.pk")
        if self.reduction and len(self.reduction) > 0:
            self.reductor1.save()
        if self.reduction and len(self.reduction) == 2:
            self.reductor2.save()


    def load(self, base_dir):
        self.clusterizer.load(base_dir + "/embedder/translator/clusterizer.pk")
        if self.reduction and len(self.reduction) > 0:
            self.reductor1.load()
        if self.reduction and len(self.reduction) == 2:
            self.reductor2.load()