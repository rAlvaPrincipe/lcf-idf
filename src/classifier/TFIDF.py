from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import WhitespaceTokenizer
import pickle
from sklearn.decomposition import TruncatedSVD
import pickle
from pathlib import Path
import torch
import os 

class TFIDF_Embedder:

    def fit(self, training, token_type, ngram_range, max_features):
        if token_type == "word":
            self.vectorizer = TfidfVectorizer(tokenizer=self.whitespace_tok, ngram_range=ngram_range, max_features = max_features)
        elif token_type == "char":
            self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=ngram_range, max_features = max_features)
        self.vectorizer.fit(training) 


    def transform(self, docs):
        '''
        convert a corpus of docs in embeddings
        input: list of strings
        output: tensor nXm with n=#docs m=length tfidf_vocab
        '''
        embeddings = self.vectorizer.transform(docs)
        embeddings = embeddings.todense()
        embeddings = torch.from_numpy(embeddings).float()
        return embeddings
        

    def embed(self, docs):
        return self.transform(docs)


    def whitespace_tok(self, doc):
        return WhitespaceTokenizer().tokenize(doc)


    def save(self, path = "./models/vectorizer.pk"):
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        pickle.dump(self.vectorizer, open(path, 'wb'))

    
    def load(self, path):
        self.vectorizer = pickle.load(open(path, "rb"))


    def get_dictionary(self):
        return self.vectorizer.get_feature_names_out()