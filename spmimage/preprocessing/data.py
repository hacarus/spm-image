from logging import getLogger

import numpy as np

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

logger = getLogger(__name__)

class WhiteningScaler(BaseEstimator, TransformerMixin):
    def __init__(self, copy=True, eps=1e-9, normalize=False,
                drop_minute=False, unbiased=True, apply_zca=False):
        self.copy = copy
        self.eps = eps
        self.normalize = normalize
        self.drop_minute = drop_minute
        self.unbiased = unbiased
        self.apply_zca = apply_zca

    def _reset(self):
        pass

    def fit(self, X):
        if sparse.issparse(X):
            logger.warning("""
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
        
        if self.drop_minute:
            s = s[s < self.eps]
            self.n_components = s.shape
            V = V[:self.n_components]
        elif self.normalize:
            s += self.eps

        self.var_ = s
        self.V = V

        S_inv = np.diag(np.ones(s.shape) / s)

        # Decorrelation & Whitening
        X = X.dot(V.T.dot(S_inv))

        # ZCA(Zero-phase Component Analysis) Whitening
        if self.apply_zca:
            X = X.dot(V)

        return X

    def fit_transform(self):
        pass

    def inverse_transform(self):
        pass
