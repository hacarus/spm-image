from logging import getLogger
from abc import abstractmethod

import numpy as np

from sklearn.utils import check_array, check_X_y
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel

logger = getLogger(__name__)


def _soft_threshold(X: np.ndarray, thresh: float) -> np.ndarray:
    return np.where(np.abs(X) <= thresh, 0, X - thresh * np.sign(X))


def _cost_function(X, y, w, z, alpha):
    n_samples = X.shape[0]
    return np.linalg.norm(y - X.dot(w)) / n_samples + alpha * np.sum(np.abs(z))


def _admm(X: np.ndarray, y: np.ndarray, D: np.ndarray, alpha: float, rho: float, tol: float, max_iter: int):
    """Alternate Direction Multiplier Method(ADMM) for Generalized Lasso.

    Minimizes the objective function::

            1 / (2 * n_samples) * ||y - Xw||^2_2 + alpha * ||z||_1

    where::

            Dw = z

    To solve this problem, ADMM uses augmented Lagrangian

            1 / (2 * n_samples) * ||y - Xw||^2_2 + alpha * ||z||_1
            + h^T (Dw - z) + rho / 2 * ||Dw - z||^2_2

    where h is Lagrange multiplier and rho is tuning parameter.
    """
    n_samples, n_features = X.shape
    n_targets = y.shape[1]

    # Initialize ADMM parameters
    w_t = X.T.dot(y) / n_samples
    z_t = D.dot(w_t.copy())
    h_t = np.zeros(w_t.shape)

    # Calculate inverse matrix
    inv_matrix = np.linalg.inv(X.T.dot(X) / n_samples + rho * D.T.dot(D))
    threshold = alpha / rho

    n_iter_ = []
    # Update ADMM parameters by columns
    for k in range(n_targets):
        # initial cost
        cost = _cost_function(X, y[:, k], w_t[:, k], z_t[:, k], alpha)
        for t in range(max_iter):
            # Update
            w_t[:, k] = inv_matrix.dot(X.T.dot(y[:, k]) / n_samples + rho * D.T.dot(z_t[:, k] - h_t[:, k] / rho))
            z_t[:, k] = _soft_threshold(D.dot(w_t[:, k]) + h_t[:, k] / rho, threshold)
            h_t[:, k] += rho * (D.dot(w_t[:, k]) - z_t[:, k])

            # after cost
            pre_cost = cost
            cost = _cost_function(X, y[:, k], w_t[:, k], z_t[:, k], alpha)
            gap = np.abs(cost - pre_cost)
            if gap < tol:
                break
        n_iter_.append(t)

    return np.squeeze(z_t), n_iter_


class GeneralizedLasso(LinearModel, RegressorMixin):
    """Alternate Direction Multiplier Method(ADMM) for Generalized Lasso.
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

        n_features = X.shape[1]
        D = self.generate_transform_matrix(n_features)
        sparse_coef, self.n_iter_ = _admm(X, y, D, self.alpha, self.rho, self.tol, self.max_iter)

        if np.linalg.matrix_rank(D) < n_features:
            self.coef_ = np.linalg.pinv(D).dot(sparse_coef)
        else:
            self.coef_ = np.linalg.inv(D).dot(sparse_coef)
        self.coef_ = np.squeeze(self.coef_)

        if y.shape[1] == 1:
            self.n_iter_ = self.n_iter_[0]

        self._set_intercept(X_offset, y_offset, X_scale)

        # workaround since _set_intercept will cast self.coef_ into X.dtype
        self.coef_ = np.asarray(self.coef_, dtype=X.dtype)

        return self

    @abstractmethod
    def generate_transform_matrix(self, n_features: int) -> np.ndarray:
        """
        :return:
        """


class LassoADMM(GeneralizedLasso):
    """Linear Model trained with L1 prior as regularizer (aka the Lasso)
    The optimization objective for Lasso is::
        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
    """

    def __init__(self, alpha=1.0, rho=1.0, fit_intercept=True,
                 normalize=False, copy_X=True, max_iter=1000,
                 tol=1e-4):
        super().__init__(alpha=alpha, rho=rho, fit_intercept=fit_intercept,
                         normalize=normalize, copy_X=copy_X, max_iter=max_iter,
                         tol=tol)

    def generate_transform_matrix(self, n_features: int) -> np.ndarray:
        return np.eye(n_features)


class FusedLassoADMM(GeneralizedLasso):
    """Fused Lasso minimises the following objective function.

1/(2 * n_samples) * ||y - Xw||^2_2 + \lambda_1 \sum_{j=1}^p |w_j| + \lambda_2 \sum_{j=2}^p |w_j - w_{j-1}|
    """

    def __init__(self, alpha=1.0, sparse_coef=1.0, fused_coef=1.0, rho=1.0, fit_intercept=True,
                 normalize=False, copy_X=True, max_iter=1000,
                 tol=1e-4):
        super().__init__(alpha=alpha, rho=rho, fit_intercept=fit_intercept,
                         normalize=normalize, copy_X=copy_X, max_iter=max_iter,
                         tol=tol)
        self.sparse_coef = sparse_coef
        self.fused_coef = fused_coef

    def generate_transform_matrix(self, n_features: int) -> np.ndarray:
        fused = np.eye(n_features) - np.eye(n_features, k=-1)
        fused[0, 0] = 0
        return self.sparse_coef * np.eye(n_features) + self.fused_coef * fused
