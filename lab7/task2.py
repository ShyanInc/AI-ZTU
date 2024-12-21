import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_iris

def load_iris_data():
    iris = load_iris()
    return iris['data'], iris['target']

def plot_clusters(X, labels, centers=None, title="Clustering Visualization"):
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', label='Data Points')
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5, label='Cluster Centers')
    plt.title(title)
    plt.legend()
    plt.show()

def perform_kmeans(X, n_clusters, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    return kmeans.predict(X), kmeans.cluster_centers_

def custom_find_clusters(X, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        labels = pairwise_distances_argmin(X, centers)
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels

def main():
    # Load data
    X, y = load_iris_data()
    y_kmeans, centers = perform_kmeans(X, n_clusters=5)

    plot_clusters(X, y_kmeans, centers, title="KMeans Clustering with 5 Clusters")
    custom_centers, custom_labels = custom_find_clusters(X, n_clusters=3, rseed=2)
    plot_clusters(X, custom_labels, custom_centers, title="Custom Clustering (rseed=2)")
    custom_centers, custom_labels = custom_find_clusters(X, n_clusters=3, rseed=0)
    plot_clusters(X, custom_labels, custom_centers, title="Custom Clustering (rseed=0)")

    y_kmeans_3, _ = perform_kmeans(X, n_clusters=3, random_state=0)
    plot_clusters(X, y_kmeans_3, title="KMeans Clustering with 3 Clusters")

if __name__ == "__main__":
    main()
