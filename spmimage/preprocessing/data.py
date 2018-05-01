from logging import getLogger

import numpy as np

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

logger = getLogger(__name__)


class WhiteningScaler(BaseEstimator, TransformerMixin):
    def __init__(self, copy=True, eps=1e-9, thresholding=None, unbiased=True, apply_zca=False):
        self.copy = copy
        self.eps = eps
        self.thresholding = thresholding
        self.unbiased = unbiased
        self.apply_zca = apply_zca

    def _fit(self, X):
        if sparse.issparse(X):
            raise ValueError("""
WhiteningScaler does not support sparse input. See TruncatedSVD for a possible alternative.
""")

        if self.eps <= 0 and self.thresholding in ['normalize', 'drop_minute']:
            raise ValueError('Threshold eps must be positive: eps={0}.'.format(self.eps))

        X = check_array(X, dtype=[np.float64, np.float32], ensure_2d=True, copy=self.copy)

        # Center data
        mean = np.mean(X, axis=0)
        X -= mean

        # SVD
        n_samples = X.shape[0] - 1 if self.unbiased else X.shape[0]
        _, s, V = np.linalg.svd(X / np.sqrt(n_samples), full_matrices=False)

        if self.thresholding is None:
            if np.any(np.isclose(s, np.zeros(s.shape), atol=1e-10)):
                raise ValueError("""
    Eigenvalues of X' are degenerated: X'=X-np.mean(X,axis=0), \
    try thresholding='normalize' or thresholding='drop_minute'.
    """)
        elif self.thresholding == 'normalize':
            s += self.eps
        elif self.thresholding == 'drop_minute':
            s = s[s > self.eps]
            V = V[:s.shape[0]]
        else:
            raise ValueError('No such parameter: thresholding={0}.'.format(self.thresholding))

        return mean, s, V

    def fit(self, X):
        self.mean_, self.var_, self.unitary_ = self._fit(X)
        return self

    def transform(self, X):
        # Decorrelation & Whitening
        S_inv = np.diag(np.ones(self.var_.shape[0]) / self.var_)
        X_transformed = (X - self.mean_).dot(self.unitary_.T.dot(S_inv))

        # ZCA(Zero-phase Component Analysis) Whitening
        if self.apply_zca:
            return X_transformed.dot(self.unitary_)
        return X_transformed

    def inverse_transform(self, X):
        S = np.diag(self.var_)
        X_itransformed = np.copy(X)
        if self.apply_zca:
            X_itransformed = X_itransformed.dot(self.unitary_.T)
        return X_itransformed.dot(S.dot(self.unitary_)) + self.mean_
