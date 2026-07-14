import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class KMeansFramework:
    def __init__(self, file_path, K=3):
        self.file_path = file_path
        self.K = K
        self.x = None
        self.model = None
        self.initial_centroids = None

    # Load Dataset
    def load_data(self):
        mat = loadmat(self.file_path)
        self.x = mat["X"]
        print("Dataset Loaded Successfully")
        print("Dataset Shape :", self.x.shape)

    #Dataset Information
    def dataset_info(self):
        print("=====Dataset Info=====")
        print("Shape",self.x.shape)
        print("Number of Samples:",self.x.shape[0])
        print("Number of Features:",self.x.shape[1])
        print("Minimum Value:",np.min(self.x,axis=0))
        print("Maximum Value:",np.max(self.x,axis=0))
        print("Mean Value:",np.mean(self.x,axis=0))

    #Plot Raw Data
    def plot_raw_data(self):
        plt.figure(figsize=(6,6))
        plt.scatter(self.x[:,0],self.x[:,1],c="blue",s=30)
        plt.title("Raw Dataset")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True)
        plt.show()

    #Elbow Method
    def elbow_method(self,max_k=10):
        inertia = []
        for k in range(1,max_k + 1):
            model = KMeans(n_clusters=k,random_state=42,n_init=10)
            model.fit(self.x)
            inertia.append(model.inertia_)
        plt.figure(figsize=(7,5))
        plt.plot(range(1,max_k+1),inertia,marker="o")
        plt.title("Elbow Method")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        plt.grid(True)
        plt.show()

    # Manual Initial Centroids
    def set_initial_centroids(self, centroids):
        self.initial_centroids = np.array(centroids)
        print(self.initial_centroids)

    #Train KMeans Model
    def fit(self,max_iter=10):
        if self.initial_centroids is None:
            raise ValueError("Please Set the Initial Centroids")
        self.model = KMeans(n_clusters=self.K, init=self.initial_centroids,n_init=1, random_state=42)
        self.model.fit(self.x)
        print("===== Training Complete =====")
        print("Model converged successfully")
        print("Final Centroids")
        print(self.model.cluster_centers_)
        print("Number of Iterations:",self.model.n_iter_)

    # Plot Every Iteration
    def plot_iterations(self, num_iters=10):
        rows = int(np.ceil(num_iters / 2))
        fig, ax = plt.subplots(rows, 2, figsize=(10, rows * 5))
        ax = ax.ravel()
        for i in range(num_iters):
            model = KMeans(n_clusters=self.K, init=self.initial_centroids, n_init=1, max_iter=i + 1, random_state=42)
            model.fit(self.x)
            labels = model.labels_
            centroids = model.cluster_centers_
            ax[i].scatter(self.x[:, 0], self.x[:, 1], c=labels, cmap="viridis", s=25)
            ax[i].scatter(centroids[:, 0], centroids[:, 1], marker="X", c="red", s=250, label="Centroids")
            ax[i].set_title(f"Iteration {i+1}")
            ax[i].legend()
        plt.tight_layout()
        plt.show()

    #Plot Final Clusters
    def plot_clusters(self):
        if self.model is None:
            print("Model has not been trained")
            return
        label = self.model.labels_
        centroids = self.model.cluster_centers_
        plt.figure(figsize=(7,7))
        plt.scatter(self.x[:,0],self.x[:,1],c=label, camp="viridis", s=35)
        plt.scatter(centroids[:,0], centroids[:,1], marker="x",c="red",s=300,label="centroids")
        plt.title("Final Clusters")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(True)
        plt.show()

    #Evaluate Model
    def evaluate(self):
        if self.model is None:
            print("Please Train the Model First")
            return
        print("===== Model Evaluation =====")
        print(f"Inertia: {self.model.inertia_:.4f}")
        score = silhouette_score(self.x, self.model.labels_)
        print(f"Silhouette score: {score:.4f}")
        print(f"Iterations Used: {self.model.n_iter_}")

    #Predict New Data
    def predict(self,new_data):
        if self.model is None:
            print("Please Train the Model First")
            return
        cluster = self.model.predict([new_data])
        print("Input Point:",new_data)
        print("Assigned Cluster:",cluster[0])
        return cluster[0]

    def save_model(self,file_name="kmeans_model.pkl"):
        if self.model is None:
            print("No Trained Model Found")
            return
        joblib.dump(self.model, file_name)
        print(f"Model Saved Successfully {file_name}")

    #load Model
    def load_model(self,file_name = "kmeans_model.pkl"):
        self.model = joblib.load(file_name)
        print(f"Model '{file_name}' Loaded Successfully")



# Main Program
if __name__ == "__main__":
    kmeans = KMeansFramework(file_path="C:/Users/User/Desktop/Machine learning exercise/data1/data/ex7/ex7data2.mat", K=3)
    kmeans.load_data()
    kmeans.dataset_info()
    kmeans.plot_raw_data()
    kmeans.elbow_method(max_k=10)
    # Manual Initial Centroids
    kmeans.set_initial_centroids([
        [3, 3],
        [6, 5],
        [8, 5]])d
    kmeans.fit()
    # Show Every Iteration
    kmeans.plot_iterations(num_iters=10)
    kmeans.evaluate()
    kmeans.predict([5,4])
    kmeans.save_model()
    kmeans.load_model()