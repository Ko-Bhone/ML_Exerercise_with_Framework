import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class LogisticRegressionFramework:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = LogisticRegression()
        self.scaler = StandardScaler()

    # Load Data
    def load_data(self):
        data = pd.read_csv(self.file_path, header=None)
        self.X = data.iloc[:, :-1].values
        self.y = data.iloc[:, -1].values

    # Plot Raw Data
    def plot_raw_data(self):
        pos = self.y == 1
        neg = self.y == 0
        plt.figure(figsize=(8, 6))
        plt.scatter(self.X[pos, 0], self.X[pos, 1], marker="+", s=100, label="Admitted")
        plt.scatter(self.X[neg, 0], self.X[neg, 1], marker="o", s=60, alpha=0.7, label="Not Admitted")
        plt.xlabel("Exam 1")
        plt.ylabel("Exam 2")
        plt.title("Raw Data")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Train Model
    def fit(self):
        self.load_data()
        self.plot_raw_data()
        # Feature Scaling
        X_scaled = self.scaler.fit_transform(self.X)
        # Train-Test Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, self.y, test_size=0.2, random_state=42)
        # Training
        self.model.fit(self.X_train, self.y_train)

    # Predict Probability
    def predict_probability(self, scores):
        scores = np.array(scores).reshape(1, -1)
        scores = self.scaler.transform(scores)
        probability = self.model.predict_proba(scores)[0][1]
        return probability

    # Accuracy
    def accuracy(self):
        prediction = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, prediction) * 100

    # Decision Boundary
    def plot_decision_boundary(self):
        X_scaled = self.scaler.transform(self.X)
        plt.figure(figsize=(8, 6))
        pos = self.y == 1
        neg = self.y == 0
        plt.scatter(X_scaled[pos, 0], X_scaled[pos, 1], marker="+", s=100,label="Admitted")
        plt.scatter(X_scaled[neg, 0], X_scaled[neg, 1], marker="o", s=60, alpha=0.7, label="Not Admitted")
        x_values = np.array([X_scaled[:, 0].min(),X_scaled[:, 0].max()])
        w = self.model.coef_[0]
        b = self.model.intercept_[0]
        y_values = -(b + w[0] * x_values) / w[1]
        plt.plot(x_values, y_values, color="red", linewidth=2, label="Decision Boundary")
        plt.xlabel("Exam 1 (Scaled)")
        plt.ylabel("Exam 2 (Scaled)")
        plt.title("Decision Boundary")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Run Project
    def run(self):
        self.fit()
        print("Intercept (Bias):")
        print(self.model.intercept_)
        print("\nCoefficients:")
        print(self.model.coef_)
        probability = self.predict_probability([45, 85])
        print(f"\nAdmission Probability: {probability * 100:.2f}%")
        print(f"\nAccuracy: {self.accuracy():.2f}%")
        self.plot_decision_boundary()

# Main
def main():
    model = LogisticRegressionFramework("C:/Users/User/Desktop/Machine learning exercise/data1/data/ex2data1.txt")
    model.run()

if __name__ == "__main__":
    main()