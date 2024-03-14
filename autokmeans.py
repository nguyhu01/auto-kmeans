from __future__ import annotations
from typing import *

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from joblib import Parallel, delayed

class AutoKMeans:
    def __init__(self, data = None, file_path = None, max_clusters= None, alpha_k = 0.02):
        """
        data: np.ndarray (clean data)
        file_path: clean data file path
        max_clusters: maximum number of clusters to consider
        alpha_k: manually tuned factor that gives penalty to the number of clusters
        """
        if data is not None:
            self.data = data
        elif file_path is not None:
            self.data = pd.read_csv(file_path)
        else:
            raise ValueError("Either 'data' or 'file_path' must be provided.")

        self.k_range = range(2, max_clusters + 1)
        self.alpha_k = alpha_k
        self.cat_col = self.data.select_dtypes(include=['object']).columns
        self.num_col = self.data.select_dtypes(exclude=['object']).columns
        self.autokmeans = False
        self.cluster_col_added = False

    def get_clusters(self):
        if not self.autokmeans:
            raise ValueError("AutoKMeans has not been run yet.")
        return self.clusters

    def get_best_k(self, clustering_data):
        if self.autokmeans:
            return self.best_k
        else:
            return self._choose_best_k_for_kmeans_parallel(clustering_data, self.k_range)[0]

    def get_cluster_labels(self):
        if not self.autokmeans:
            raise ValueError("AutoKMeans has not been run yet.")
        return self.cluster_labels

    def _scaled_inertia(self, scaled_data: np.ndarray, k: int) -> float:
        '''
        Parameters
        ----------
        scaled_data: matrix
            scaled data. rows are samples and columns are features for clustering
        k: int
            current k for applying KMeans
        Returns
        -------
        scaled_inertia: float
            scaled inertia value for current k
        '''
        inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
        # Fit k-means
        kmeans = KMeans(n_clusters=k, random_state=0).fit(scaled_data)
        scaled_inertia = kmeans.inertia_ / inertia_o + self.alpha_k * k

        return scaled_inertia

    def _choose_best_k_for_kmeans_parallel(self, scaled_data: np.ndarray, k_range) -> Tuple[int, pd.DataFrame]:
        '''
        Parameters
        ----------
        scaled_data: matrix
            scaled data. rows are samples and columns are features for clustering
        k_range: list of integers
            k range for applying KMeans
        Returns
        -------
        best_k: int
            chosen value of k out of the given k range.
            chosen k is k with the minimum scaled inertia value.
        results: pandas DataFrame
            adjusted inertia value for each k in k_range
        '''
        ans = Parallel(n_jobs=-1, verbose=10)(delayed(self._scaled_inertia)(scaled_data, k) for k in k_range)
        ans = list(zip(k_range, ans))
        results = pd.DataFrame(ans, columns=['k', 'Scaled Inertia']).set_index('k')
        results['Scaled Inertia'] = pd.to_numeric(results['Scaled Inertia'], errors='coerce')
        results = results.dropna()

        if results.empty:
            raise ValueError("No valid results found for the given k range.")

        best_k = results.idxmin()[0]
        print("The best number of clusters is:", best_k)

        return best_k, results

    def fit(self, clustering_features : list) -> None:
        self.clustering_features = clustering_features
        self.clustering_data = self.data[clustering_features]
        
        self.best_k, results = self._choose_best_k_for_kmeans_parallel(self.clustering_data, self.k_range)

        kmeans = KMeans(n_clusters=self.best_k, n_init='auto').fit(self.clustering_data)
        self.cluster_labels = kmeans.predict(self.clustering_data)
        self.cluster_centers = kmeans.cluster_centers_
        self.clustering_data['Clusters'] = pd.Series(data = self.cluster_labels)

        self.autokmeans = True

        self._map_clusters()

    def evaluate_clustering(self):
        if not self.autokmeans:
            raise ValueError("AutoKMeans has not been run yet.")
        silhouette_avg = silhouette_score(self.clustering_data, self.cluster_labels)
        print(f"Silhouette Score for Clustering: {silhouette_avg}")

    def visualize_clusters_PCA(self):
        if not self.autokmeans:
            raise ValueError("AutoKMeans has not been run yet.")
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(self.clustering_data)
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=self.clustering_data['Clusters'], cmap='viridis')

        plt.title('Cluster Visualization')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(label='Cluster')
        plt.show()

    def _map_clusters(self):
        unique_labels = np.unique(self.cluster_labels)
        label_map = {label: f"Cluster {label + 1}" for label in unique_labels}
        self.clustering_data['Clusters'] = self.clustering_data['Clusters'].map(label_map)
        print(self.clustering_data['Clusters'].isnull().sum())

    def remove_cluster_column(self):
        self.clustering_data.drop(['Clusters'], axis=1, inplace=True)

