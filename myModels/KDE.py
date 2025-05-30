from sklearn.base import BaseEstimator
import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable
import os
from joblib import Parallel, delayed

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

class KDE(BaseEstimator):
    def __init__(self,
                 bandwidth: float = 1.0,
                 kernel: str | Union[str, Callable[[float], float]]= 'gaussian',
                 metric : str | Union[str, Callable[[float, float], float]] = 'euclidean',
                 n_jobs: int | None = None):
        super().__init__()
        
        self.bandwidth_ = bandwidth
        
        if isinstance(kernel, str): self.kernel_ = Kernels()(kernel)
        else: self.kernel_ = kernel
        
        if isinstance(metric, str): self.metric_ = Metrics()(metric)
        else: self.metric_ = metric
        
        if n_jobs is None: self.n_jobs = os.cpu_count()
        else: self.n_jobs = n_jobs
        
    def fit(self, X: NDArray, y: None = None) -> BaseEstimator:
        
        h = self.bandwidth_
        n = X.shape[0]
        metric = self.metric_
        kernel = self.kernel_
        
        def density(x: NDArray) -> float:
            distances = np.array([metric(x, xi) for xi in X])
            normalized = distances / h
            kernel_values = kernel(normalized)
            return np.sum(kernel_values) / (n * h)
        
        self.density = density
        
        return self
    
    def score_samples(self, X: NDArray) -> NDArray:
        log_density = np.log(np.array(
            Parallel(n_jobs=self.n_jobs)(delayed(self.density)(x) for x in X)))
        
        return log_density
    
    def score(self, X) -> float:
        log_densities = self.score_samples(X)
        return log_densities.sum()
        