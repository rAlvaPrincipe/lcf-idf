from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import torch
import numpy as np
import pickle
import hdbscan 
import numpy
import os
from pathlib import Path

class Clusterizer():

    def __init__(self, algorithm, n_clusters=None):
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        if self.algorithm == "kmeans":
            self.model = KMeans(n_clusters=n_clusters, random_state=0)
        elif self.algorithm == "hdbscan":
            self.model = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True, gen_min_span_tree=False, leaf_size=40, prediction_data=True,  metric='euclidean', min_cluster_size=5, min_samples=None, p=None)

    def run(self, embeddings):
        if not isinstance(embeddings, numpy.ndarray):
            embeddings = embeddings.numpy()
        print("--> Fitting " + str(type(self.model).__name__), ", n_clusters: " + str(self.n_clusters))
        self.model = self.model.fit(embeddings)


    def predict(self, embedding):
        if not isinstance(embedding, numpy.ndarray):
            embedding = embedding.numpy()

        if self.algorithm == "kmeans":
            return self.model.predict(embedding)
        elif self.algorithm == "hdbscan":
            test_labels, strengths = hdbscan.approximate_predict(self.model, embedding)
            return strengths


    def save(self, path="models/clusterizer.pk"):
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        pickle.dump(self.model, open(path, "wb"))


    def load(self, path):
        self.model = pickle.load(open(path, "rb"))


    def analyze_kmeans(self, attempts, embeddings):
        '''
        input: attempts is the number of clusters to try and use for comparison, embeddings is a numpy array nx768 with n=#tokens
        output: saves the Elbow methods analysis in a png
        '''
        wcss = []
        for i in attempts:
            kmeans = KMeans(n_clusters=i, random_state=0).fit(embeddings)
            wcss.append(kmeans.inertia_)
          
        plt.plot(attempts, wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.savefig("B.png")
        plt.close()