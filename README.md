# Recommendation System using Stochastic Gradient Descent (SGD)

## 📌 Project Overview
This project implements a **Collaborative Filtering Recommendation System** using **Matrix Factorization with Stochastic Gradient Descent (SGD)**. The model is trained on an online **book rating dataset**, and its performance is evaluated using **Root Mean Squared Error (RMSE).**

## 📂 Dataset
- Dataset: [Goodbooks-10K](https://github.com/zygmuntz/goodbooks-10k)
- Contains user ratings for books.
- The dataset consists of three key columns:
  - `user_id`: Unique identifier for each user.
  - `book_id`: Unique identifier for each book.
  - `rating`: User rating for the book (1-5 scale).

## 🚀 Features
- Uses **Matrix Factorization** for recommendation.
- Optimized with **Stochastic Gradient Descent (SGD)**.
- Includes **Regularization** to prevent overfitting.
- Evaluates performance using **RMSE**.

## 📌 Installation
Ensure you have Python and the necessary libraries installed:

```bash
pip install numpy pandas scikit-learn
```

## 🏗️ Implementation Steps
### 1️⃣ Load and Preprocess Data
- Read dataset from an online source.
- Convert user and item IDs into numerical indices.
- Split data into **80% training** and **20% testing**.

### 2️⃣ Implement Matrix Factorization with SGD
- Initialize user (`U`) and item (`V`) matrices randomly.
- Train model using SGD updates:
  ```python
  U[u] += lr * (error * V[i] - reg * U[u])
  V[i] += lr * (error * U[u] - reg * V[i])
  ```
- Use **regularization** to reduce overfitting.

### 3️⃣ Train the Model
- Run training for **20 epochs**, printing progress after each.
- Shuffle training data at each epoch for better convergence.

### 4️⃣ Evaluate with RMSE
- Compute RMSE on the test set:
  ```python
  rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
  ```
- Lower RMSE indicates better model performance.

## 🔥 Results
- **Final RMSE:** `0.8409`

## 🛠️ Usage
To train and evaluate the model, run:
```python
python recommendation_sgd.py
```
