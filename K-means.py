# K-means is an unsupervised learning algorithm used for clustering data points into groups based on their similarity. 
# The algorithm is based on the principle of minimizing the sum of squared distances between data points and the centroid of their corresponding cluster. The procedure for K-means clustering is as follows:

# Select the number of clusters (K) you want to form.
# Initialize K centroids randomly.
# Assign each data point to the nearest centroid, forming K clusters.
# Recalculate the centroid of each cluster based on the data points assigned to it.
# Repeat steps 3 and 4 until the centroids no longer move significantly.

import numpy as np

class KMeans:
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        
    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for i in range(self.max_iter):
            # Assign each data point to the nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Recalculate the centroid of each cluster
            for j in range(self.k):
                self.centroids[j] = X[labels == j].mean(axis=0)
                
        return labels
    
def main():
    # Generate sample data
    np.random.seed(0)
    X = np.random.randn(100, 2)
    
    # Run K-means clustering
    kmeans = KMeans(k=3)
    labels = kmeans.fit(X)
    
    # Print the labels of each data point
    print(labels)
    
if __name__ == '__main__':
    main()