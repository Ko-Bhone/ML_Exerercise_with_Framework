import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat


class DigitClassifierTorch:
    def __init__(self, data_path, lr=0.1, epochs=300):
        self.data_path = data_path
        self.lr = lr
        self.epochs = epochs

        self.X = None
        self.y = None

        self.model = None
        self.loss_history = []

    # -------------------
    # Load Data
    # -------------------
    def load_data(self):
        data = loadmat(self.data_path)
        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.y = torch.tensor(data["y"].ravel() - 1, dtype=torch.long)
        # PyTorch uses 0–9 labels

        print("Loaded:", self.X.shape)

    # -------------------
    # Model (Neural Net)
    # -------------------
    def build_model(self):
        self.model = nn.Sequential(
            nn.Linear(400, 25),
            nn.Sigmoid(),
            nn.Linear(25, 10)
        )

    # -------------------
    # Train
    # -------------------
    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            # forward
            outputs = self.model(self.X)
            loss = criterion(outputs, self.y)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.loss_history.append(loss.item())

            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        print("Training Finished")

    # -------------------
    # Predict
    # -------------------
    def predict(self):
        with torch.no_grad():
            outputs = self.model(self.X)
            pred = torch.argmax(outputs, dim=1)

            acc = (pred == self.y).float().mean().item() * 100
            print(f"Accuracy: {acc:.2f}%")

            return pred.numpy()

    # -------------------
    # Random test
    # -------------------
    def show_random(self, pred):
        import matplotlib.pyplot as plt

        idx = np.random.randint(0, self.X.shape[0])

        img = self.X[idx].numpy().reshape(20, 20).T
        plt.imshow(img, cmap="gray")
        plt.title(f"Actual: {self.y[idx].item()} | Pred: {pred[idx]}")
        plt.axis("off")
        plt.show()


# -------------------
# Run
# -------------------
if __name__ == "__main__":
    model = DigitClassifierTorch(
        data_path="C:/Users/User/Desktop/Machine learning exercise/data1/data/ex3data1.mat",
        lr=0.1,
        epochs=300
    )

    model.load_data()
    model.build_model()
    model.train()

    pred = model.predict()
    model.show_random(pred)