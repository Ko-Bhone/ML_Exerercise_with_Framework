import torch
import torch.optim as optim
from scipy.io import loadmat

class CollaborativeFilteringTorch:
    def __init__(self, movie_file, param_file, Lambda=1, lr=0.001, epochs=1000):
        self.movie_file = movie_file
        self.param_file = param_file
        self.Lambda = Lambda
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Y = None
        self.R = None
        self.X = None
        self.Theta = None
        self.num_movies = 0
        self.num_users = 0
        self.num_features = 0

    # Load Data
    def load_data(self):
        movie = loadmat(self.movie_file)
        self.Y = torch.tensor(movie["Y"], dtype=torch.float32, device=self.device)
        self.R = torch.tensor(movie["R"], dtype=torch.float32, device=self.device)
        param = loadmat(self.param_file)
        self.X = torch.tensor(param["X"], dtype=torch.float32, device=self.device, requires_grad=True)
        self.Theta = torch.tensor(param["Theta"], dtype=torch.float32, device=self.device, requires_grad=True)
        self.num_movies = int(param["num_movies"].item())
        self.num_users = int(param["num_users"].item())
        self.num_features = int(param["num_features"].item())
        print("Data Loaded")
        print("Y:",self.Y.shape)
        print("X:",self.X.shape)
        print("Theta:",self.Theta.shape)

    # Cost Function
    def cost_function(self):
        prediction = self.X @ self.Theta.T
        error = (prediction - self.Y) * self.R
        loss = 0.5 * torch.sum(error**2)
        reg = (self.Lambda/2 * (torch.sum(self.X**2) + torch.sum(self.Theta**2)))
        return loss + reg

    # Training
    def train(self):
        optimizer = optim.Adam([self.X,self.Theta], lr=self.lr)
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            loss = self.cost_function()
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch} Loss {loss.item():.2f}")
        print("Training Finished")

    # Predict Rating
    def predict(self):
        with torch.no_grad():
            return (self.X @ self.Theta.T)

    # Recommend
    def recommend(self, user_id, top_n=10):
        prediction = self.predict()
        user_rating = prediction[:,user_id]
        values, indices = torch.sort(user_rating, descending=True)
        print(f"\n===== User {user_id} Recommendation =====")
        for i in range(top_n):
            print(
                f"Movie {indices[i].item()} "
                f"→ {values[i].item():.2f}"
            )

if __name__=="__main__":
    movie_path = (
    "C:/Users/User/Desktop/Machine learning exercise/data1/data/ex8/ex8_movies.mat")
    param_path = (
    "C:/Users/User/Desktop/Machine learning exercise/data1/data/ex8/ex8_movieParams.mat")

    model = CollaborativeFilteringTorch(movie_path, param_path, Lambda=1, lr=0.001, epochs=1000)
    model.load_data()
    model.train()
    model.recommend(user_id=10, top_n=5)