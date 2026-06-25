import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class LinearRegressionModel:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.X = None
        self.y = None
        self.model = LinearRegression()

    def load_data(self):
        """Load dataset"""
        data = pd.read_csv(self.file_path, header=None)
        self.X = data.iloc[:, 0].values.reshape(-1, 1)
        self.y = data.iloc[:, 1].values
        print(f"Dataset Loaded: {len(self.y)} samples")

    def train(self):
        """Train Linear Regression model"""
        self.model.fit(self.X, self.y)
        print("\nTraining Complete")
        print(f"Intercept (θ0): {self.model.intercept_:.4f}")
        print(f"Slope (θ1): {self.model.coef_[0]:.4f}")

    def evaluate(self):
        """Evaluate model performance"""
        predictions = self.model.predict(self.X)
        mse = mean_squared_error(self.y, predictions)
        r2 = r2_score(self.y, predictions)
        print("\nModel Evaluation")
        print(f"MSE : {mse:.4f}")
        print(f"R²  : {r2:.4f}")

    def predict(self, population):
        """
        population = city population in 10,000s
        """
        prediction = self.model.predict([[population]])
        return prediction[0] * 10000

    def plot_result(self):
        """Plot training data and regression line"""
        plt.figure(figsize=(8, 5))
        plt.scatter(self.X, self.y, color="red", label="Training Data")
        plt.plot(self.X, self.model.predict(self.X), linewidth=2, label="Regression Line")
        plt.xlabel("Population of City (10,000s)")
        plt.ylabel("Profit (10,000 $)")
        plt.title("Linear Regression")
        plt.legend()
        plt.grid(True)
        plt.show()


def main():
    file_path = r"C:\Users\User\Desktop\Machine learning exercise\data1\data\ex1data1.txt"
    model = LinearRegressionModel(file_path)
    model.load_data()
    model.train()
    model.evaluate()
    model.plot_result()
    prediction = model.predict(3.5)

    print(
        f"\nPredicted profit for population 35,000 = ${prediction:,.2f}")


if __name__ == "__main__":
    main()