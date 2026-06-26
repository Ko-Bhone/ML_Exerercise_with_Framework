import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

class RegularizedLogisticRegressionFramework:
    def __init__(self, file_path, degree=6, C=1.0):
        self.file_path = file_path
        self.degree = degree
        self.C = C
        self.df = None
        self.X = None
        self.y = None
        self.pipeline = None
        self.X_poly = None

    # Load Dataset
    def load_data(self):
        self.df = pd.read_csv(self.file_path, header=None)
        self.X = self.df.iloc[:, :-1].values
        self.y = self.df.iloc[:, -1].values

    # Build ML Pipeline
    def build_model(self):
        self.pipeline = Pipeline([("poly", PolynomialFeatures(degree=self.degree, include_bias=True)), ("logistic", LogisticRegression( C=self.C, solver="lbfgs", max_iter=5000))])

    # Train Model
    def train(self):
        self.pipeline.fit(self.X, self.y)
        self.X_poly = self.pipeline.named_steps["poly"].transform(self.X)

    # PCA Visualization
    def pca_projection(self):
        pca = PCA(n_components=2)
        return pca.fit_transform(self.X_poly[:,1:])

    # Decision Boundary
    def plot_decision_boundary(self, ax):
        pos = self.y == 1
        neg = self.y == 0
        ax.scatter(self.X[pos,0], self.X[pos,1], c="green", s=70, edgecolors="black", label="Accepted")
        ax.scatter(self.X[neg,0], self.X[neg,1], c="red", marker="X", s=70, edgecolors="black", label="Rejected")
        u = np.linspace(-1,1.5,300)
        v = np.linspace(-1,1.5,300)
        xx, yy = np.meshgrid(u,v)
        grid = np.c_[xx.ravel(), yy.ravel()]
        z = self.pipeline.predict(grid)
        z = z.reshape(xx.shape)
        ax.contour(xx,yy,z, levels=[0.5], linewidths=3)
        ax.set_title("Decision Boundary")
        ax.grid(alpha=0.3)
        ax.legend()

    # Dashboard
    def plot_dashboard(self):
        pos = self.y == 1
        neg = self.y == 0
        fig, ax = plt.subplots(1,3,figsize=(18,5))
        # Raw Data
        ax[0].scatter(self.X[pos,0], self.X[pos,1], c="green", s=70, edgecolors="black")
        ax[0].scatter(self.X[neg,0], self.X[neg,1], c="red", marker="X", s=70, edgecolors="black")
        ax[0].set_title("Raw Data")
        ax[0].grid(alpha=0.3)

        # PCA
        X_pca = self.pca_projection()
        ax[1].scatter(X_pca[pos,0], X_pca[pos,1], c="green", s=70, edgecolors="black")
        ax[1].scatter(X_pca[neg,0], X_pca[neg,1], c="red", marker="X", s=70, edgecolors="black")
        ax[1].set_title("Polynomial Features (PCA)")
        ax[1].grid(alpha=0.3)
        # Decision Boundary
        self.plot_decision_boundary(ax[2])
        plt.tight_layout()
        plt.show()

    # Accuracy
    def evaluate(self):
        accuracy = self.pipeline.score(self.X,self.y)
        print("="*40)
        print("Accuracy :",accuracy)
        print("="*40)

   # Run Pipeline
    def run(self):
        self.load_data()
        print("Raw Shape :",self.X.shape)
        self.build_model()
        self.train()
        print("Mapped Shape :",self.X_poly.shape)
        self.evaluate()
        self.plot_dashboard()


if __name__ == "__main__":
    model = RegularizedLogisticRegressionFramework(
        file_path="C:/Users/User/Desktop/Machine learning exercise/data1/data/ex2data2.txt", degree=6, C=1.0)
    model.run()