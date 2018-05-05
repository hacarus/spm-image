from logging import getLogger

import numpy as np

from sklearn.utils import check_array, check_X_y
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel

logger = getLogger(__name__)


def _soft_threshold(X: np.ndarray, thresh: float) -> np.ndarray:
    return np.where(np.abs(X) <= thresh, 0, X - thresh * np.sign(X))


def _cost_function(X, y, w, alpha):
    n_samples = X.shape[0]
    return np.linalg.norm(y - X.dot(w)) / n_samples + alpha * np.sum(np.abs(w))


def _admm(X: np.ndarray, y: np.ndarray, alpha: float, rho: float, tol: float, max_iter: int):
    n_samples, n_features = X.shape
    n_targets = y.shape[1]

    # Initialize ADMM parameters
    w_t = X.T.dot(y) / n_samples
    z_t = w_t.copy()
    h_t = np.zeros(w_t.shape)

    # Calculate inverse matrix
    inv_matrix = np.linalg.inv(X.T.dot(X) / n_samples + rho * np.eye(n_features))
    threshold = alpha / rho

    n_iter_ = []
    # Update ADMM parameters by columns
    for k in range(n_targets):
        for t in range(max_iter):
            # current cost
            pre_cost = _cost_function(X, y[:, k], w_t[:, k], alpha)

            # Update
            w_t[:, k] = inv_matrix.dot(X.T.dot(y[:, k]) / n_samples + (rho * z_t[:, k]) - h_t[:, k])
            z_t[:, k] = _soft_threshold(w_t[:, k] + (h_t[:, k] / rho), threshold)
            h_t[:, k] += rho * (w_t[:, k] - z_t[:, k])

            # after cost
            post_cost = _cost_function(X, y[:, k], w_t[:, k], alpha)
            gap = np.abs(pre_cost - post_cost)
            if gap < tol:
                break
        n_iter_.append(t)

    return np.squeeze(z_t), n_iter_


class LassoADMM(LinearModel, RegressorMixin):
    """Linear Model trained with L1 prior as regularizer (aka the Lasso)
    The optimization objective for Lasso is::
        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
    Technically the Lasso model is optimizing the same objective function as
    the Elastic Net with ``l1_ratio=1.0`` (no L2 penalty).
    """

    def __init__(self, alpha=1.0, rho=1.0, fit_intercept=True,
                 normalize=False, copy_X=True, max_iter=1000,
                 tol=1e-4):
        self.alpha = alpha
        self.rho = rho
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None

    def fit(self, X, y, check_input=False):
        if self.alpha == 0:
            logger.warning("""
With alpha=0, this algorithm does not converge well. You are advised to use the LinearRegression estimator            
""")

        if check_input:
            X, y = check_X_y(X, y, accept_sparse='csc',
                             order='F', dtype=[np.float64, np.float32],
                             copy=self.copy_X and self.fit_intercept,
                             multi_output=True, y_numeric=True)
            y = check_array(y, order='F', copy=False, dtype=X.dtype.type,
                            ensure_2d=False)

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(X, y, fit_intercept=self.fit_intercept,
                                                                  normalize=self.normalize,
                                                                  copy=self.copy_X and not check_input)

        if y.ndim == 1:
            y = y[:, np.newaxis]

        self.coef_, self.n_iter_ = _admm(X, y, self.alpha, self.rho, self.tol, self.max_iter)
        if y.shape[1] == 1:
            self.n_iter_ = self.n_iter_[0]

        self._set_intercept(X_offset, y_offset, X_scale)

        # workaround since _set_intercept will cast self.coef_ into X.dtype
        self.coef_ = np.asarray(self.coef_, dtype=X.dtype)

        return self
