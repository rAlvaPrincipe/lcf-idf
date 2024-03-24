import umap
import pickle
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from src.translation_pipeline.autoencoder import Autoencoder
import torch


class Reductor:
    def __init__(self, algorithm, n_components, base_dir):
        self.base_dir = base_dir
        if algorithm == "pca":
            self.reductor = PCA(n_components)
            #self.reductor = PCA(n_components=0.50, svd_solver='full')
        elif algorithm == "umap":
           self.reductor = umap.UMAP(random_state=42, n_components=n_components, low_memory=True)

        elif algorithm == "autoencoder":
           self.reductor = Autoencoder(base_dir).to("cuda") 

        self.alg = str(type(self.reductor).__name__)
        self.n_commponents = n_components


    def fit(self, train_embeddings, val_embeddings):
        print("--> Fitting " + self.alg + ":", str(train_embeddings.shape[1]) + " --> " + str(self.n_commponents))
        if self.alg == "PCA":
            self.reductor = self.reductor.fit(train_embeddings)
            return self.reductor.transform(train_embeddings)
        elif self.alg == "UMAP":
            self.reductor = self.reductor.fit(train_embeddings)
            return self.reductor.embedding_
        elif self.alg == "Autoencoder":
            return self.reductor.fit(train_embeddings, val_embeddings)

    def trasform(self, embeddings):
        return self.reductor.transform(embeddings)


    def save(self):
        if self.alg == "PCA":
            pickle.dump(self.reductor, open(self.base_dir  + "/reductor_pca.pk", "wb"))
        elif self.alg == "UMAP":
            pickle.dump(self.reductor, open(self.base_dir  + "/reductor_umap.pk", "wb"))
        elif self.alg == "Autoencoder":
            self.reductor.save(self.base_dir  +  "/reductor_autoencoder.pt")

    def load(self):
        if self.alg == "PCA":
            self.reductor = pickle.load(open(self.base_dir  + "/reductor_pca.pk", "rb"))
        elif self.alg == "UMAP":
            self.reductor = pickle.load(open(self.base_dir  + "/reductor_umap.pk", "rb"))
        elif self.alg == "Autoencoder":
            self.reductor.load(self.base_dir  + "/reductor_autoencoder.pt")


    def show_variance(self):
        if type(self.reductor).__name__ == "PCA":
            # plt.plot(np.cumsum(self.reductor.explained_variance_ratio_))
            # plt.xlabel('number of components')
            # plt.ylabel('cumulative explained variance');
            # plt.savefig("img.png")
            
            
            exp_var_pca = self.reductor.explained_variance_ratio_
            
            cum_sum_eigenvalues = np.cumsum(exp_var_pca)
            #
            # Create the visualization plot
            #
            plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
            plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
            plt.ylabel('Explained variance ratio')
            plt.xlabel('Principal component index')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig("img.png")