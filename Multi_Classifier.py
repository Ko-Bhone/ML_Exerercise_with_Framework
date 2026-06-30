import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.io import loadmat

class DigitNNTorch:
    def __init__(self, path, lr=0.8, epochs=800, Lambda=1):
        self.path = path
        self.lr = lr
        self.epochs = epochs
        self.Lambda = Lambda
        self.model = None
        self.loss_history = []

    # Load Data
    def load_data(self):
        data = loadmat(self.path)
        X = data["X"]
        y = data["y"].ravel() - 1  # convert 1–10 → 0–9
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        print("Loaded:", self.X.shape)

    # Model (same 400→25→10)
    def build_model(self):
        self.model = nn.Sequential(
            nn.Linear(400, 25),
            nn.Sigmoid(),
            nn.Linear(25, 10))

    # Train
    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.model.parameters(), lr=self.lr, weight_decay=self.Lambda)  # L2 regularization)

        for i in range(self.epochs):
            # forward
            outputs = self.model(self.X)
            loss = criterion(outputs, self.y)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())
            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss.item():.4f}")
        print("Training finished")

    # Predict
    def predict(self):
        with torch.no_grad():
            outputs = self.model(self.X)
            pred = torch.argmax(outputs, dim=1)
            acc = (pred == self.y).float().mean().item() * 100
            print(f"Accuracy: {acc:.2f}%")
            return pred.numpy()

    # Plot loss
    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(self.loss_history)
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.show()

if __name__ == "__main__":
    model = DigitNNTorch(
        path="C:/Users/User/Desktop/Machine learning exercise/data1/data/ex3data1.mat", lr=0.8, epochs=800, Lambda=1)
    model.load_data()
    model.build_model()
    model.train()
    model.predict()
    model.plot_loss()