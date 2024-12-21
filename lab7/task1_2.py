import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def load_data(file_path):
    try:
        data = np.loadtxt(file_path, delimiter=',')
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise


def plot_initial_data(data):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], marker='o', facecolors='none', edgecolors='black', s=80)
    plt.title('Input Data')
    plt.xlim(data[:, 0].min() - 1, data[:, 0].max() + 1)
    plt.ylim(data[:, 1].min() - 1, data[:, 1].max() + 1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def perform_kmeans_clustering(data, num_clusters):
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(data)

    step_size = 0.01
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)

    plt.figure()
    plt.clf()
    plt.imshow(output, interpolation='nearest', extent=(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()),
               cmap=plt.cm.Paired, aspect='auto', origin='lower')
    plt.scatter(data[:, 0], data[:, 1], marker='o', facecolors='none', edgecolors='black', s=80)
    cluster_centers = kmeans.cluster_centers_
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='o', s=210, linewidths=4, color='black', zorder=12,
                facecolors='black')
    plt.title('Cluster Boundaries')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def main():
    file_path = 'data_clustering.txt'  # Path to the data file
    num_clusters = 5  # Number of clusters

    data = load_data(file_path)
    plot_initial_data(data)

    perform_kmeans_clustering(data, num_clusters)


if __name__ == "__main__":
    main()
