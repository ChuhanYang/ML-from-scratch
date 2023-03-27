# KNN (K-Nearest Neighbors) is a supervised machine learning algorithm used for classification and regression tasks. 
# It works on the principle of finding the k nearest neighbors to a sample point and assigning the class/label of the majority of the k neighbors to the sample point.

# Procedure for KNN:

# Choose the number of k neighbors to consider.
# Calculate the distance between the test data and each training data point.
# Select the k-nearest neighbors based on the smallest distance.
# Assign the class/label of the majority of the k neighbors to the test data.

import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for x in X:
            distances = []
            for i in range(len(self.X_train)):
                distance = np.sqrt(np.sum(np.square(x - self.X_train[i])))
                distances.append((distance, self.y_train[i]))
            distances.sort()
            neighbors = distances[:self.k]
            classes = {}
            for neighbor in neighbors:
                if neighbor[1] in classes:
                    classes[neighbor[1]] += 1
                else:
                    classes[neighbor[1]] = 1
            y_pred.append(max(classes, key=classes.get))
        return y_pred

def main():
    X_train = np.array([[1, 2], [1, 4], [4, 2], [4, 4]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[2, 2], [2, 3], [3, 2], [3, 3]])
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(y_pred) # expected output: [0, 0, 1, 1]

if __name__ == "__main__":
    main()

