import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 📌 Step 1: Load Dataset (E-Commerce User Ratings)
url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv"  # Example dataset
df = pd.read_csv(url)

# 📌 Step 2: Preprocessing
df = df[['user_id', 'book_id', 'rating']]  # Select relevant columns
df.columns = ['user', 'item', 'rating']  # Rename for consistency

# Encode IDs for matrix factorization
df['user'] = df['user'].astype("category").cat.codes
df['item'] = df['item'].astype("category").cat.codes

# Split into train & test
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# 📌 Step 3: Matrix Factorization with SGD
class SGDRS:
    def __init__(self, num_users, num_items, latent_factors=10, lr=0.01, reg=0.1, epochs=20):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_factors = latent_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs

        # Initialize user & item matrices
        self.U = np.random.normal(scale=0.1, size=(num_users, latent_factors))
        self.V = np.random.normal(scale=0.1, size=(num_items, latent_factors))

    def train(self, train_data):
        for epoch in range(self.epochs):
            np.random.shuffle(train_data.values)
            for u, i, r in train_data.values:
                pred = np.dot(self.U[u], self.V[i].T)
                error = r - pred

                # Update rules
                self.U[u] += self.lr * (error * self.V[i] - self.reg * self.U[u])
                self.V[i] += self.lr * (error * self.U[u] - self.reg * self.V[i])

            print(f"Epoch {epoch+1}/{self.epochs} completed")

    def predict(self, user, item):
        return np.dot(self.U[user], self.V[item].T)

# 📌 Step 4: Train the Model
num_users = df['user'].nunique()
num_items = df['item'].nunique()

model = SGDRS(num_users, num_items, latent_factors=10, lr=0.01, reg=0.1, epochs=20)
model.train(train_data)

# 📌 Step 5: Evaluate using RMSE
predictions = []
actuals = []

for u, i, r in test_data.values:
    pred = model.predict(u, i)
    predictions.append(pred)
    actuals.append(r)

rmse = np.sqrt(mean_squared_error(actuals, predictions))
print(f"RMSE: {rmse:.4f}")
