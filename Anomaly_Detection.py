import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope

class GaussianAnomalyFramework:
    def __init__(self):
        self.file_path = "C:/Users/User/Desktop/Machine learning exercise/data1/data/ex8/ex8data1.mat"
        self.X = None
        self.X_val = None
        self.y_val = None
        self.pipeline = None
        self.load_data()
        self.build_pipeline()

    # Load Dataset
    def load_data(self):
        data = loadmat(self.file_path)
        self.X = data["X"]
        self.X_val = data["Xval"]
        self.y_val = data["yval"].ravel()
        print("=" * 50)
        print("Dataset Loaded")
        print("=" * 50)
        print(self.X.shape)
        print(self.X_val.shape)
        print(self.y_val.shape)

    # ML Pipeline
    def build_pipeline(self):
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("detector",
             EllipticEnvelope(
                 contamination=0.01,
                 random_state=42
             ))

        ])

    # Train
    def fit(self):
        self.pipeline.fit(self.X)
        print("\nModel Training Finished")

    # Predict
    def predict(self):
        prediction = self.pipeline.predict(self.X)
        anomaly = prediction == -1
        print(f"Detected Outliers : {np.sum(anomaly)}")
        return anomaly

    # Decision Score
    def anomaly_score(self):
        score = self.pipeline.decision_function(self.X)
        return score

    # Visualization
    def plot_result(self):
        anomaly = self.predict()
        plt.figure(figsize=(10,7))
        plt.scatter(self.X[:,0], self.X[:,1], marker="x", label="Normal")
        plt.scatter(self.X[anomaly,0], self.X[anomaly,1], s=220, facecolors="none", edgecolors="red", linewidths=2, label="Outlier")
        plt.xlabel("Latency")
        plt.ylabel("Throughput")
        plt.title("Gaussian Anomaly Detection (Framework)")
        plt.grid(True)
        plt.legend()
        plt.show()

    # Evaluate
    def evaluate(self):
        pred = self.pipeline.predict(self.X_val)
        pred = (pred == -1).astype(int)
        tp = np.sum((pred == 1) & (self.y_val == 1))
        fp = np.sum((pred == 1) & (self.y_val == 0))
        fn = np.sum((pred == 0) & (self.y_val == 1))
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) \
            if precision + recall else 0
        print("\nEvaluation")
        print("=" * 30)
        print("Precision :", precision)
        print("Recall    :", recall)
        print("F1 Score  :", f1)

    # Run
    def run(self):
        self.fit()
        self.evaluate()
        self.plot_result()

if __name__ == "__main__":
    model = GaussianAnomalyFramework()
    model.run()