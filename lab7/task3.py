import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

def load_data(file_path):
    try:
        data = np.loadtxt(file_path, delimiter=',')
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise

def perform_meanshift_clustering(data, quantile=0.1):
    bandwidth = estimate_bandwidth(data, quantile=quantile, n_samples=len(data))
    meanshift_model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    meanshift_model.fit(data)
    cluster_centers = meanshift_model.cluster_centers_
    labels = meanshift_model.labels_
    num_clusters = len(np.unique(labels))
    return cluster_centers, labels, num_clusters

def plot_clusters(data, cluster_centers, labels, num_clusters):
    plt.figure()
    markers = cycle('o*xvs')

    for i, marker in zip(range(num_clusters), markers):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], marker=marker, label=f'Cluster {i}')

    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='o', color='red', s=150, label='Cluster Centers')
    plt.title('MeanShift Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def main():
    file_path = 'data_clustering.txt'
    data = load_data(file_path)

    cluster_centers, labels, num_clusters = perform_meanshift_clustering(data)

    print(f'\nCenters of clusters:\n{cluster_centers}')
    print(f"\nNumber of clusters in input data = {num_clusters}")

    plot_clusters(data, cluster_centers, labels, num_clusters)

if __name__ == "__main__":
    main()
