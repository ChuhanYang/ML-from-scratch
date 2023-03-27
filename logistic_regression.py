# Logistic Regression is a machine learning algorithm used for classification problems. It models the probability of a binary outcome (0 or 1) as a function of the input variables. The basic idea is to model the relationship between the input variables and the probability of the output variable using a logistic function.

# 1. Initialize weights w and bias b to random values
# 2. Define a sigmoid function as:
#    sigmoid(z) = 1 / (1 + exp(-z))
# 3. Repeat until convergence:
#    a. Compute the cost function J(w,b) and its gradients:
#       J(w,b) = -1/m * sum(y*log(sigmoid(X*w + b)) + (1-y)*log(1-sigmoid(X*w + b)))
#       dw = 1/m * X^T * (sigmoid(X*w + b) - y)
#       db = 1/m * sum(sigmoid(X*w + b) - y)
#    b. Update the weights and bias using the gradients and a learning rate alpha:
#       w = w - alpha * dw
#       b = b - alpha * db
# 4. Predict the output y_hat for a new input x using:
#    y_hat = sigmoid(x*w + b)

import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=100000, fit_intercept=True, verbose=False):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.learning_rate * gradient
            
            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold

def main():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 1])
    model = LogisticRegression(learning_rate=0.001, num_iterations=30000)
    model.fit(X, y)
    print(model.predict(np.array([[1, 2], [3, 4], [5, 6]]), 0.5))

if __name__ == '__main__':
    main()
