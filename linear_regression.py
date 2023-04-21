# linear regression

# 1. Initialize the coefficients b0, b1, ..., bn to zero or small random values
# 2. Repeat until convergence or a maximum number of iterations is reached:
#     3. For each data point (x_i, y_i), compute the predicted value y_pred = b0 + b1*x_i + ... + bn*x_n
#     4. Compute the cost J = (1/2m) * sum((y_pred - y_i)^2) over all data points, where m is the number of data points
#     5. Compute the partial derivative of the cost with respect to each coefficient: dJ/db0, dJ/db1, ..., dJ/dbn
#     6. Update each coefficient using the gradient descent update rule: bi = bi - learning_rate * dJ/dbi, where the learning rate is a hyperparameter
# 7. Return the final coefficients

import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

def main():
    # Generate toy data
    np.random.seed(0)
    X = np.random.rand(100, 5)
    y = 2 * X.dot(np.ones(5)) + np.random.normal(scale=0.1, size=100)
    
    # Fit linear regression model
    model = LinearRegression(learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    
    # Print results
    print("Coefficients:", model.weights)
    print("Intercept:", model.bias)
    y_pred = model.predict(X)
    print("R-squared:", 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

if __name__ == '__main__':
    main()