# Decision tree creates a model based on training data that is used to make predictions about a new data point. It works by recursively partitioning the feature space into smaller regions, and at each step, it chooses the feature that maximizes the information gain. The resulting model is a binary tree, where each internal node represents a test on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label.


# Choose a target variable that is to be predicted.
# Split the dataset into smaller subsets based on the values of the predictor variables.
# Calculate the entropy of each subset.
# Calculate the information gain of each feature.
# Choose the feature with the highest information gain as the root node.
# Repeat steps 2-5 recursively for each branch of the tree, stopping when all instances belong to the same class or when no more features are left.
import numpy as np

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        
    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)
        
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return {'type': 'leaf', 'value': leaf_value}
        
        best_feature, best_threshold = self._best_criteria(X, y)
        left_indices, right_indices = self._split(X[:, best_feature], best_threshold)
        
        left_branch = self._grow_tree(X[left_indices, :], y[left_indices], depth+1)
        right_branch = self._grow_tree(X[right_indices, :], y[right_indices], depth+1)
        
        return {'type': 'split', 'feature': best_feature, 'threshold': best_threshold, 
                'left': left_branch, 'right': right_branch}
    
    def _most_common_label(self, y):
        return np.bincount(y).argmax()
    
    def _best_criteria(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_index in range(X.shape[1]):
            feature_values = X[:, feature_index]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                gain = self._information_gain(y, feature_values, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _information_gain(self, y, X, threshold):
        parent_entropy = self._entropy(y)
        
        left_indices, right_indices = self._split(X, threshold)
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0
        
        left_labels, right_labels = y[left_indices], y[right_indices]
        
        left_entropy = self._entropy(left_labels)
        right_entropy = self._entropy(right_labels)
        
        child_entropy = (len(left_labels)/len(y)) * left_entropy + (len(right_labels)/len(y)) * right_entropy
        
        return parent_entropy - child_entropy
    
    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs))
    
    def _split(self, feature_values, threshold):
        left_indices = np.where(feature_values <= threshold)[0]
        right_indices = np.where(feature_values > threshold)[0]
        return left_indices, right_indices
    
    def _traverse_tree(self, x, node):
        if node['type'] == 'leaf':
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])

def main():
    # Generate sample data
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 0, 1, 1, 1])
    
    # Initialize and fit the decision tree
    tree = DecisionTree()
    tree.fit(X, y)
    
    # Test the decision tree
    X_test = np.array([[1, 2], [3, 4], [5, 6]])
    y_pred = tree.predict(X_test)
    print(y_pred)

if __name__ == '__main__':
    main()