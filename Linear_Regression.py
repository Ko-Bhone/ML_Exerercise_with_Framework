import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
import joblib

class LinearRegressionFramework:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X = None
        self.y = None
        self.model = LinearRegression()

    # Load Dataset
    def load_data(self):
        self.df = pd.read_csv( self.file_path, header=None, names=["Population", "Profit"])
        print("=" * 60)
        print("DATASET LOADED")
        print("=" * 60)
        print("\nFirst 5 Rows")
        print(self.df.head())
        print("\nDataset Shape")
        print(self.df.shape)
        print("\nData Types")
        print(self.df.dtypes)
        print("\nMissing Values")
        print(self.df.isnull().sum())
        print("\nStatistical Summary")
        print(self.df.describe())

    # Clean Data
    def clean_data(self):
        self.df.drop_duplicates(inplace=True)
        self.df.dropna(inplace=True)
        self.df["Population"] = pd.to_numeric(self.df["Population"], errors="coerce")
        self.df["Profit"] = pd.to_numeric(self.df["Profit"], errors="coerce")
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        print("\nData Cleaning Completed.")
        print(self.df.shape)

    # Prepare Data
    def prepare_data(self):
        self.X = self.df[["Population"]]
        self.y = self.df["Profit"]
        print("\nX Shape :", self.X.shape)
        print("y Shape :", self.y.shape)

    # Pipeline
    def dataset_pipeline(self):
        self.load_data()
        self.clean_data()
        self.prepare_data()

    # Train Model
    def train(self):
        print("\nTraining Model...")
        self.model.fit(self.X, self.y)
        print("\nIntercept :", self.model.intercept_)
        print("Coefficient :", self.model.coef_[0])

    # Evaluate
    def evaluate(self):
        prediction = self.model.predict(self.X)
        mae = mean_absolute_error(self.y, prediction)
        mse = mean_squared_error(self.y, prediction)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y, prediction)
        print("\nMODEL EVALUATION")
        print("=" * 40)
        print(f"MAE  : {mae:.4f}")
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R²   : {r2:.4f}")

    # Predict
    def predict(self, population):
        X_new = pd.DataFrame(
            {"Population": [population]})
        prediction = self.model.predict(X_new)
        print(f"Population : {population * 10000:,.0f}")
        print(f"Predicted Profit : ${prediction[0] * 10000:,.2f}")
        return prediction

    # Save Model
    def save_model(self):
        joblib.dump(self.model, "linear_regression_model.pkl")
        print("\nModel Saved.")

    # Load Model
    def load_model(self):
        self.model = joblib.load("linear_regression_model.pkl")
        print("\nModel Loaded.")

    # Plot Regression Line
    def plot_regression_line(self):
        prediction = self.model.predict(self.X)
        plt.figure(figsize=(8,5))
        plt.scatter(self.X, self.y, color="red")
        plt.plot(self.X, prediction, color="blue")
        plt.title("Linear Regression")
        plt.xlabel("Population")
        plt.ylabel("Profit")
        plt.grid(True)
        plt.show()

    # Plot Actual vs Predicted
    def plot_actual_vs_predicted(self):
        prediction = self.model.predict(self.X)
        plt.figure(figsize=(6,6))
        plt.scatter(self.y, prediction)
        m = min(self.y.min(), prediction.min())
        M = max(self.y.max(), prediction.max())
        plt.plot([m,M],[m,M],'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.grid(True)
        plt.show()

    # Residual Plot
    def plot_residuals(self):
        prediction = self.model.predict(self.X)
        residual = self.y - prediction
        plt.figure(figsize=(8,5))
        plt.scatter(prediction, residual)
        plt.axhline(0,color="red",linestyle="--")
        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    FILE_PATH = "C:/Users/User/Desktop/Machine learning exercise/data1/data/ex1data1.txt"
    model = LinearRegressionFramework(FILE_PATH)
    model.dataset_pipeline()
    model.train()
    model.evaluate()
    model.predict(3.5)
    model.save_model()
    model.plot_regression_line()
    model.plot_actual_vs_predicted()
    model.plot_residuals()