from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from numpy.typing import NDArray
from joblib import Parallel, delayed
from typing import Union, Callable
import os

class Kernels:
    def __init__(self):
        self.kernels = {
            'gaussian': Kernels.gaussian
        }
    
    def __call__(self, name: str):
        if name not in self.kernels.keys():
            raise ValueError(f"not realize {name} kernel")
        
        return self.kernels[name]
    
    @staticmethod
    def gaussian(t: float) -> float:
        return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * t**2)
    
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
    
class RBFLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self,
                 kernel: str | Union[str, Callable[[float], float]]= 'gaussian',
                 metric : str | Union[str, Callable[[float, float], float]] = 'euclidean',
                 n_jobs: int | None = None,
                 lambda_: float = 1e-3):
        super().__init__()
        
        if isinstance(kernel, str): self.kernel_ = Kernels()(kernel)
        else: self.kernel_ = kernel
        
        if isinstance(metric, str): self.metric_ = Metrics()(metric)
        else: self.metric_ = metric
        
        if n_jobs is None: self.n_jobs = os.cpu_count()
        else: self.n_jobs = n_jobs
        
        self.lambda_ = lambda_
        
        self.K = None
        self.weights = None
        
    def fit(self, X: NDArray, y: NDArray) -> BaseEstimator:
        
        def K(x: NDArray) -> NDArray:
            distances = np.array([self.metric_(x, xi)**2 for xi in X])
            return np.vectorize(self.kernel_)(distances)
        
        self.K = K
        
        KX = np.array(Parallel(n_jobs=self.n_jobs)(delayed(self.K)(x) for x in X))
        n = KX.shape[0]
        self.weights = np.linalg.lstsq(KX + self.lambda_ * np.eye(n), y)[0]
        
        return self
    
    def predict(self, X: NDArray, y: None = None) -> NDArray:
        KX = np.array(Parallel(n_jobs=self.n_jobs)(delayed(self.K)(x) for x in X))
        return KX @ self.weights
        
        
        
        
    