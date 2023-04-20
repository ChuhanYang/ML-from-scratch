# Principal Components Analysis (PCA) is a technique used for reducing the dimensionality of a dataset while preserving the most important information. It works by transforming the original variables into a new set of uncorrelated variables called principal components.

# Standardize the data by subtracting the mean and dividing by the standard deviation of each variable.
# Compute the covariance matrix of the standardized data.
# Compute the eigenvectors and eigenvalues of the covariance matrix.
# Sort the eigenvalues in descending order and select the top k eigenvectors that correspond to the k largest eigenvalues.
# Project the original data onto the k eigenvectors to obtain the reduced-dimensionality data.

# The resulting reduced-dimensionality data contains the same amount of information as the original data but with a smaller number of variables. PCA is often used as a preprocessing step for machine learning algorithms to improve their performance and reduce overfitting.

import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        
    def fit(self, X):
        # Subtract mean from each feature
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        
        # Calculate covariance matrix
        cov = np.cov(X.T)
        
        # Calculate eigenvectors and eigenvalues of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Sort eigenvectors by eigenvalues in descending order
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[idxs]
        eigenvalues = eigenvalues[idxs]
        
        # Store first n eigenvectors as components
        self.components = eigenvectors[0:self.n_components]
        
    def transform(self, X):
        # Subtract mean from each feature
        X = X - self.mean
        
        # Project data onto components
        return np.dot(X, self.components.T)

def main():
    # Generate toy dataset
    X = np.random.randn(100, 5)
    
    # Create PCA object and fit to data
    pca = PCA(n_components=2)
    pca.fit(X)
    
    # Transform data onto new principal components
    X_transformed = pca.transform(X)
    
    # Print original and transformed data
    print("Original data:\n", X)
    print("\nTransformed data:\n", X_transformed)

if __name__ == '__main__':
    main()