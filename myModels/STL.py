from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import pandas as pd

class STL(BaseEstimator):
    def __init__(self,
                 period: int,
                 seasonal: int,
                 trend: int | None = None,
                 low_pass: int | None = None,
                 seasonal_deg: int = 1,
                 trend_deg: int = 1,
                 low_pass_deg: int = 1):
        super().__init__()
        
        self.Y = None
        self.T = None
        self.S = None
        self.R = None
        
        self.period = period
        self.seasonal = seasonal
        
        if trend is None:
            trend = int(np.ceil(1.5 * self.period / (1 - 1.5 / self.seasonal)))
            trend += ((trend % 2) == 0)
        self.trend = trend
        
        if low_pass is None:
            low_pass = self.period + 1
            low_pass += ((low_pass % 2) == 0)
        self.low_pass = low_pass
        
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.low_pass_deg = low_pass_deg
        
        self.rho = None
        
    def fit(self, X: NDArray, inner_iter: int = 0, outer_iter: int = 0):
        self.Y = np.array(X.copy())
        n = X.shape[0]
        self.T = np.zeros(n)
        self.S = np.zeros(n)
        self.R = np.zeros(n)
        self.rho = np.ones(n)
        
        for _ in range(outer_iter + 1):
            for _ in range(inner_iter):
                self.__inner_loop()
            self.__outer_loop()

        return self
    
    def __LOESS(self,
        x: NDArray,
        y: NDArray,
        q: int = 5,
        d: int = 1,
        rho: NDArray | None = None) -> NDArray:
        n = x.shape[0]
        result = np.zeros(n)
    
        for i in range(n):
            distance = np.abs(x - x[i])
            idx_sorted = np.argsort(distance)
            w = idx_sorted[:q]
            max_d = distance[w[-1]]
            if rho is None: weights = (1 - (np.abs(x[w] - x[i]) / max_d)**3)**3
            else: weights = rho[w] * (1 - (np.abs(x[w] - x[i]) / max_d)**3)**3
            X = np.vander(x[w], d + 1, increasing=True)
            W = np.diag(weights)
            beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y[w])[0]
            result[i] = np.polyval(beta[::-1], x[i])
    
        return result
    
    def __moving_average(self, signal, window_size):
        
        signal = np.asarray(signal)
        pad_width = window_size // 2

        padded = np.pad(signal, pad_width=pad_width, mode='reflect')

        series_padded = pd.Series(padded)
        ma_padded = series_padded.rolling(window=window_size, center=True).mean()
        result = ma_padded[pad_width : pad_width + len(signal)]

        return result.values
    
    def __inner_loop(self):
        #step 1
        y_detrend = self.Y - self.T
        
        n = self.Y.shape[0]
        t = np.arange(n)
        for i in range(self.period):
            #step 2
            idx = np.arange(i, n, self.period)
            if len(idx) == 0: continue
            x_cycle = t[idx]
            y_cycle = y_detrend[idx]

            C = self.__LOESS(x_cycle, y_cycle, self.seasonal, self.seasonal_deg, rho=self.rho[idx])
            
            #step 3
            ma1 = self.__moving_average(C, self.period)
            ma2 = self.__moving_average(ma1, self.period)
            ma3 = self.__moving_average(ma2, 3)
            L = self.__LOESS(np.arange(len(ma3)), ma3, self.low_pass, self.low_pass_deg)
                
            #step 4
            self.S[idx] = C - L
        
        #step 5    
        y_deseasond = self.Y - self.S
    
        #step 6
        self.T = self.__LOESS(t, y_deseasond, self.trend, self.trend_deg, rho=self.rho)
        
    def __outer_loop(self):
        self.R = self.Y - self.T - self.S
        h = 6 * np.median(np.abs(self.R))
        self.rho = (1 - (np.abs(self.R) / h)**2)**2
        self.rho[self.rho < 0.0] = 0.0
        
        
    def get_components(self) -> dict:
        return {
            "observed": self.Y,
            "trend": self.T,
            "seasonal": self.S,
            "residual": self.R
        }

    def plot(self):
        """Визуализация результатов декомпозиции"""
        components = self.get_components()

        plt.figure(figsize=(12, 8))

        titles = ['Observed', 'Trend', 'Seasonal', 'Residual']
        for i, key in enumerate(components):
            plt.subplot(4, 1, i + 1)
            plt.plot(components[key], label=key)
            plt.legend()
            plt.title(titles[i])

        plt.tight_layout()
        
