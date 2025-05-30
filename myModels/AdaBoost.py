from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from numpy.typing import NDArray

class Stump(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = None
        self.error = float('inf')

    def fit(self, X: NDArray, y: NDArray, sample_weight=None):
        n_samples, n_features = X.shape
        
        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                for polarity in [1, -1]:
                    pred = self.__predict(X, feature_idx, threshold, polarity)
                    incorrect = pred != y
                    error = np.sum(sample_weight[incorrect])

                    if error < self.error:
                        self.error = error
                        self.feature_idx = feature_idx
                        self.threshold = threshold
                        self.polarity = polarity
                        self.error = self.error

        return self

    def __predict(self, X, feature_idx, threshold, polarity):
        feature_column = X[:, feature_idx]
        predictions = np.ones(X.shape[0])
        if polarity == 1:
            predictions[feature_column <= threshold] = -1
        else:
            predictions[feature_column > threshold] = -1
        return predictions

    def predict(self, X):
        if self.feature_idx is None or self.threshold is None:
            raise ValueError("Модель не обучена.")
        return self.__predict(X, self.feature_idx, self.threshold, self.polarity)



class AdaBoost(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_estimators: int = 50,
                 estimator: BaseEstimator = Stump()):
        super().__init__()
        
        self.n_estimators = n_estimators
        self.estimator = estimator
        
        self.betas_ = [None] * n_estimators
        self.estimators_ = [None] * n_estimators
        
        self.__replace_labels = None
        
    def fit(self, X: NDArray, y: NDArray) -> BaseEstimator:
        X = X.copy()
        y = y.copy()
        
        n = X.shape[0]
        weights = np.ones(n) / n
        
        y_uni = np.unique(y)
        self.__replace_labels = {y_uni[0]: -1, y_uni[1]: 1}
        y = np.vectorize(self.__replace_labels.get)(y)
        
        for k in range(self.n_estimators):
            self.estimators_[k] = clone(self.estimator)
            self.estimators_[k].fit(X, y, sample_weight=weights)
            
            labels = self.estimators_[k].predict(X)
            
            a = np.sum(weights * (y != labels))
            b = np.sum(weights)
            beta = 0.5 * np.log((b - a) / a)
            
            weights[y != labels] = weights[y != labels] * np.exp(2*beta)
            
            self.betas_[k] = beta
        
        return self
    
    def predict(self, X) -> NDArray:
        reverse_replace_labels = {value: key
                                  for key, value in self.__replace_labels.items()}
        
        pred = np.zeros(X.shape[0])
        for beta, est in zip(self.betas_, self.estimators_):
            pred += beta * est.predict(X)
        return np.vectorize(reverse_replace_labels.get)(np.sign(pred))
        
    def score(self, X, y, sample_weight = None):
        return super().score(X, y, sample_weight)
