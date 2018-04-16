from logging import getLogger

import numpy as np

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

logger = getLogger(__name__)

class WhiteningScaler(BaseEstimator, TransformerMixin):
    def __init__(self, copy=True, eps=1e-9, thresholding=None,
                unbiased=True, apply_zca=False):
        self.copy = copy
        self.eps = eps
        self.thresholding = thresholding
        self.unbiased = unbiased
        self.apply_zca = apply_zca

    def _reset(self):
        pass

    def _fit(self, X):
        if sparse.issparse(X):
            logger.error("""
WhiteningScaler does not support sparse input. See TruncatedSVD for a possible alternative.
""")

        X = check_array(X, dtype=[np.float64, np.float32],
                        ensure_2d=True, copy=self.copy)

        n_samples = X.shape[0]

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        # SVD
        if self.unbiased:
            _, s, V = np.linalg.svd(X / np.sqrt(n_samples - 1), full_matrices=False)
        else:
            _, s, V = np.linalg.svd(X / np.sqrt(n_samples), full_matrices=False)

        if self.eps <= 0 and (self.thresholding == 'normalize'
                                or self.thresholding == 'drop_minute'):
            logger.error("""
Threshold eps must be positive: eps={0}.
""".format(self.eps))

        if self.thresholding is None:
            if np.any(np.isclose(s, np.zeros(s.shape), atol=1e-10)):
                logger.error("""
Eigenvalues of X' are degenerated: X'=X-np.mean(X,axis=0), \
try normalize=True or drop_minute=True.
""")
        elif self.thresholding == 'normalize':
            s += self.eps
            n_components = s.shape[0]
        elif self.thresholding == 'drop_minute':
            s = s[s > self.eps]
            n_components = s.shape[0]
            V = V[:n_components]
        else:
            logger.error("""
No such parameter: thresholding={0}.
""".format(self.thresholding))

        S_inv = np.diag(np.ones(s.shape[0]) / s)

        # Decorrelation & Whitening
        X = X.dot(V.T.dot(S_inv))

        # ZCA(Zero-phase Component Analysis) Whitening
        if self.apply_zca:
            X = X.dot(V)

        return (X, s, V, n_components)

    def fit(self, X):
        X, s, V, n_components = self._fit(X)

        self.components_ = X
        self.var_ = s
        self.unitary_ = V
        self.n_components = n_components

        return self

    def transform(self, X):
        S_inv = np.diag(np.ones(self.var_.shape[0]) / self.var_)
        X_transformed = (X - self.mean_).dot(self.unitary_.T.dot(S_inv))
        if self.apply_zca:
            return X_transformed.dot(self.unitary_)
        return X_transformed

    def fit_transform(self):
        X, s, V, n_components = self._fit(X)

        self.components_ = X
        self.var_ = s
        self.unitary_ = V
        self.n_components = n_components

        return X

    def inverse_transform(self, X):
        S = np.diag(self.var_)
        X_itransformed = np.copy(X)
        if self.apply_zca:
            X_itransformed = X_itransformed.dot(self.unitary_.T)
        return X_itransformed.dot(S.dot(self.unitary_)) + self.mean_
