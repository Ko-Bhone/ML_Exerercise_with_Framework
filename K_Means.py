import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import KMeans

class KMeansFramework:
    def __init__(self, file_path, K=3):
        self.file_path = file_path
        self.K = K
        self.X = None
        self.initial_centroids = None

    # Load Dataset
    def load_data(self):
        mat = loadmat(self.file_path)
        self.X = mat["X"]
        print("Dataset Shape :", self.X.shape)

    # Manual Initial Centroids
    def set_initial_centroids(self, centroids):
        self.initial_centroids = np.array(centroids)

    # Plot Every Iteration
    def plot_iterations(self, num_iters=10):
        rows = int(np.ceil(num_iters / 2))
        fig, ax = plt.subplots(rows, 2, figsize=(10, rows * 5))
        ax = ax.ravel()
        for i in range(num_iters):
            model = KMeans(n_clusters=self.K, init=self.initial_centroids, n_init=1, max_iter=i + 1, random_state=42)
            model.fit(self.X)
            labels = model.labels_
            centroids = model.cluster_centers_
            ax[i].scatter(self.X[:, 0], self.X[:, 1], c=labels, cmap="viridis", s=25)
            ax[i].scatter(centroids[:, 0], centroids[:, 1], marker="X", c="red", s=250, label="Centroids")
            ax[i].set_title(f"Iteration {i+1}")
            ax[i].legend()
        plt.tight_layout()
        plt.show()

    # Final Training
    def fit(self, max_iter=10):
        model = KMeans(n_clusters=self.K, init=self.initial_centroids, n_init=1, max_iter=max_iter, random_state=42)
        model.fit(self.X)
        print("\nFinal Centroids")
        print(model.cluster_centers_)

# Main Program

if __name__ == "__main__":
    kmeans = KMeansFramework(file_path="C:/Users/User/Desktop/Machine learning exercise/data1/data/ex7/ex7data2.mat", K=3)
    kmeans.load_data()
    # Manual Initial Centroids
    kmeans.set_initial_centroids([
        [3, 3],
        [6, 5],
        [8, 5]])

    # Show Every Iteration
    kmeans.plot_iterations(num_iters=10)

    # Final Result
    kmeans.fit(max_iter=10)