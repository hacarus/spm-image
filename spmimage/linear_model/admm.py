# coding: utf-8

import numpy as np

from sklearn.linear_model.base import LinearModel, RegressorMixin

class LassoADMM(LinearModel, RegressorMixin):
    """Linear Model trained with L1 prior as regularizer (aka the Lasso)
    The optimization objective for Lasso is::
        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
    Technically the Lasso model is optimizing the same objective function as
    the Elastic Net with ``l1_ratio=1.0`` (no L2 penalty).
    """
    def __init__(self, alpha=1.0, rho=1.0, fit_intercept=False, normalize=False, max_iter=1000):
        self.alpha = alpha
        self.rho = rho
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.threshold = alpha / rho
        self.intercept_ = 0

    def fit(self, X, y, check_input=False):
        if self.alpha == 0:
            warnings.warn("With alpha=0, this algorithm does not converge "
                          "well. You are advised to use the LinearRegression "
                          "estimator", stacklevel=2)

        if check_input:
            # We have to check array X & y
            pass
        # 本来はここでスケーリング
        X = X
        y = y

        n_samples, n_features = X.shape

        inv_matrix = np.linalg.inv(X.T.dot(X) / n_samples + self.rho * np.eye(n_features))
        w_t = X.T.dot(y) / n_samples
        z_t = w_t.copy()
        h_t = np.zeros(len(w_t))

        for t in range(self.max_iter):
            w_t = inv_matrix.dot(X.T.dot(y) / n_samples + (self.rho * z_t) - h_t)
            z_t = self._soft_threshold(w_t + (h_t / self.rho), self.threshold)
            h_t += self.rho * (w_t - z_t)

        self.coef_ = w_t

        return self

    def _soft_threshold(self, X: np.ndarray, thresh: float) -> np.ndarray:
        return np.where(np.abs(X) <= thresh, 0, X - thresh * np.sign(X))
