import matplotlib.pyplot as plt
import numpy as np


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


def main():
    # Parameters
    file_path = 'data_clustering.txt'  # Path to the data file
    num_clusters = 5  # Number of clusters

    data = load_data(file_path)
    plot_initial_data(data)


if __name__ == "__main__":
    main()
