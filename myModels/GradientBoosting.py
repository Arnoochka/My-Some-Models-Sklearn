from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from numpy.typing import NDArray

class Metrics:
    def __init__(self):
        self.metrics = {
            'euclidean': Metrics.euclidean
        }
    
    def __call__(self, name: str):
        if name not in self.metrics.keys():
            raise ValueError(f"not realize {name} metric")
        
        return self.metrics[name]
        
    @staticmethod
    def euclidean(x: NDArray, y: NDArray) -> float:
        return np.sqrt(np.sum((x - y)**2))

class DecisionTree(BaseEstimator, RegressorMixin):
    def __init__(self,
                 max_depth: int = 3):
        self.max_depth = max_depth
        self.tree: dict | None = None
        self.criterion = Metrics()('euclidean')

    def fit(self, X: NDArray, y: NDArray) -> None:
        self.tree = self._grow_tree(X, y)

    def predict(self, X: NDArray) -> NDArray:
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _grow_tree(self, X: NDArray, y: NDArray, depth: int = 0) -> dict:
        
        n_samples, n_features = X.shape
        if depth >= self.max_depth or len(np.unique(X, axis=0)) == 1:
            return {'value': np.mean(y)}

        best_gain = -np.inf
        best_split = None

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = X[:, feature_idx] > threshold

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                gain = self.criterion(y, np.mean(y)) - (
                    len(y[left_mask]) / len(y) * self.criterion(y[left_mask], np.mean(y[left_mask])) +
                    len(y[right_mask]) / len(y) * self.criterion(y[right_mask], np.mean(y[right_mask]))
                )

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }

        left_tree = self._grow_tree(X[best_split['left_mask']], y[best_split['left_mask']], depth + 1)
        right_tree = self._grow_tree(X[best_split['right_mask']], y[best_split['right_mask']], depth + 1)

        return {
            'feature_idx': best_split['feature_idx'],
            'threshold': best_split['threshold'],
            'left': left_tree,
            'right': right_tree
        }

    def _predict_one(self, x: NDArray, tree: dict) -> float:
        if 'value' in tree:
            return tree['value']
        feature_val = x[tree['feature_idx']]
        if feature_val <= tree['threshold']:
            return self._predict_one(x, tree['left'])
        else:
            return self._predict_one(x, tree['right'])
        

class GradientBoosting(BaseEstimator, RegressorMixin):
    def __init__(self,
                 learning_rate: float = 0.1,
                 n_estimators: int = 100,
                 max_depth: int = 3):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        self.init = 0.0

    def fit(self, X: NDArray, y: NDArray):
        self.init = y.mean()
        preds = np.zeros(len(y)) + self.init

        for _ in range(self.n_estimators):
            residuals = y - preds
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            preds += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
            
        return self

    def predict(self, samples: NDArray) -> NDArray:
        preds = np.zeros(len(samples)) + self.init

        for tree in self.trees:
            preds += self.learning_rate * tree.predict(samples)

        return preds