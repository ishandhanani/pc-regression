import typing
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class PCARegression(BaseEstimator, RegressorMixin):
    """
    Principal Component Regression

    This is a sklearn compatible wrapper class for Principal
    Component Analysis and Linear Regression. I had to run PCR
    multiple times for a homework assignment and decided to build a
    custom wrapper for the entire process. I am currently in the process
    of turning it into a pip installable package.

    Parameters
    ----------
    n_components: int, default=2
        The number of principal components
    """

    def __init__(self, n_components:int=2):
        self.lm = LinearRegression()
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.ss = StandardScaler()

    def __name__(self):
        return self

    def fit(self, X: np.ndarray, y=None):
        scaledX = self.ss.fit_transform(X)
        xpca = self.pca.fit_transform(scaledX)
        self.lm.fit(xpca, y)
        return self

    def predict(self, X: np.ndarray, y=None):
        scaledX = self.ss.fit_transform(X)
        xpca = self.pca.fit_transform(scaledX)
        preds = self.lm.predict(xpca)
        return preds
