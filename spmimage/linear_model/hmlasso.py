from logging import getLogger

import numpy as np
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

logger = getLogger(__name__)


def _preprocess_data(X, y, fit_intercept, normalize=False, copy=True, check_input=True):
    """
    Apply preprocess to data.

    Parameters
    ----------
    X : np.ndarray, shape = (n_samples, n_features)
        Data.

    y : np.ndarray, shape = (n_samples, ) or (n_samples, n_targets)
        Target.

    fit_intercept : boolean
        Whether to calculate the intercept for this model.
        If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).

    normalize : boolean, optional (default=False)
        This parameter is ignored when fit_intercept is set to False.
        If True, the regressors X will be standardized before regression.

    copy : boolean, optional (default=True)
        If True, X will be copied; else, it may be overwritten.

    check_input : boolean, optional (default=True)
        Allow to bypass several input checking.

    Returns
    -------
    X : np.ndarray, shape = (n_samples, n_features)
        Processed data.

    y : np.ndarray, shape = (n_samples, ) or (n_samples, n_targets)
        Processed target.

    X_offset : np.ndarray, shape = (n_features, )
        Mean of data.

    y_offset : np.ndarray or int
        Mean of target.

    X_scale : np.ndarray, shape = (n_features, )
        Standard deviation of data.
    """
    if check_input:
        X = check_array(X, copy=copy, accept_sparse='csc',
                        dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
    if copy:
        X = X.copy()

    y = np.asarray(y, dtype=X.dtype)

    if fit_intercept:
        # standardize
        if normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X_offset, X_scale = scaler.mean_, scaler.var_
            X_scale = np.sqrt(X_scale)
            X_scale[X_scale == 0] = 1
        else:
            scaler = StandardScaler(with_std=False)
            X = scaler.fit_transform(X)
            X_offset = scaler.mean_
            X_scale = np.ones(X.shape[1], dtype=X.dtype)
        y_offset = np.average(y, axis=0)
        y = y - y_offset
    else:
        X_offset = np.zeros(X.shape[1], dtype=X.dtype)
        X_scale = np.ones(X.shape[1], dtype=X.dtype)
        if y.ndim == 1:
            y_offset = X.dtype.type(0)
        else:
            y_offset = np.zeros(y.shape[1], dtype=X.dtype)

    return X, y, X_offset, y_offset, X_scale


def _proximal_map(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Proximal mapping.

    Parameters
    ----------
    x : np.ndarray, shape = (n, )

    Returns
    -------
    : np.ndarray, shape = (n, )
    """
    return np.sign(x) * np.maximum((np.abs(x) - alpha), 0)


def _symm(x: np.ndarray) -> np.ndarray:
    """
    Return symmetric matrix.

    Parameters
    ----------
    x : np.ndarray, shape = (n, n)

    Returns
    -------
    : np.ndarray, shape = (n, n)
    """
    return 0.5 * (x + x.T)


def _projection(x: np.ndarray, eps: float) -> np.ndarray:
    """
    Projection onto PSD.

    Parameters
    ----------
    x : np.ndarray, shape = (n, n)
        symmetric matrix

    Returns
    -------
    : np.ndarray, shape = (n, n)
        symmetric positive definite matrix
    """
    w, v = np.linalg.eigh(x)
    return v.dot(np.diag(np.maximum(w, eps)).dot(v.T))


def _cost_function_qpl1(x: np.ndarray, Q: np.ndarray, p: np.ndarray, alpha: float) -> float:
    """
    Return 1 / 2 * x^TQx + p^Tx + alpha * ||x||_1.
    """
    return 0.5 * x.dot(Q.dot(x)) + p.dot(x) + alpha * np.sum(np.abs(x))


def _admm_qpl1(Q: np.ndarray, p: np.ndarray, alpha: float, mu: float, tol: float, max_iter: int) -> (np.ndarray, int):
    """
    Alternate Direction Multiplier Method (ADMM) for quadratic programming with l1 regularization.

    Minimizes the objective function::
            1 / 2 * x^TQx + p^Tx + alpha * ||z||_1

    To solve this problem, ADMM uses augmented Lagrangian
            1 / 2 * x^TQx + p^Tx + alpha * ||z||_1
            + h^T (x - z) + mu / 2 * ||x - z||^2_2
    where h is Lagrange multiplier and mu is tuning parameter.
    """
    n_features = p.shape[0]

    coef_matrix = Q + mu * np.eye(n_features)
    inv_matrix = np.linalg.inv(coef_matrix)
    alpha_mu = alpha / mu
    inv_mu = 1 / mu
    x = -np.copy(p)
    z = -np.copy(p)
    h = np.zeros_like(p)
    cost = _cost_function_qpl1(x, Q, p, alpha)
    t = 0
    for t in range(max_iter):
        x = inv_matrix.dot(mu * z - p - h)
        z = _proximal_map(x + inv_mu * h, alpha_mu)
        h = h + mu * (x - z)

        pre_cost = cost
        cost = _cost_function_qpl1(x, Q, p, alpha)
        gap = np.abs(cost - pre_cost)
        if gap < tol:
            break
    return x, t


def _cost_function_psd(A: np.ndarray, S: np.ndarray, W: np.ndarray) -> float:
    """
    Return 1 / 2 * ||W * (A - S)||^2_F.
    """
    return 0.5 * np.sum((W * (A - S)) ** 2)


def _admm_psd(S: np.ndarray, W: np.ndarray, mu: float, tol: float, max_iter: int, eps: float) -> np.ndarray:
    """
    Alternate Direction Multiplier Method (ADMM) for following minimization.

    Minimizes the objective function::
            1 / 2 * ||W * (A - S)||^2_F
    where
            A ≥ O

    To solve this problem, ADMM uses augmented Lagrangian
            1 / 2 * ||W * B||^2_F - <Lambda, A - B - S> + mu / 2 * ||A - B - S||^2_F
    where Lambda is Lagrange multiplier and mu is tuning parameter.
    """
    # initialize
    A = np.zeros_like(S)
    B = np.zeros_like(S)
    Lambda = np.zeros_like(S)

    cost = _cost_function_psd(B, S, W)
    weight = W * W / mu + np.eye(S.shape[0])
    for _ in range(max_iter):
        A = _projection(_symm(B + S + Lambda), eps)
        B = (A - S - Lambda) / weight
        Lambda = Lambda - (A - B - S)

        pre_cost = cost
        cost = _cost_function_psd(B, S, W)
        gap = np.abs(cost - pre_cost)
        if gap < tol:
            break
    return A


def _update(
        S: np.ndarray, W: np.ndarray, mu_cov: float, tol_cov: float, max_iter_cov: int, eps: float,
        p: np.ndarray, alpha: float, mu_coef: float, tol_coef: float, max_iter_coef: int
) -> (np.ndarray, int):
    """
    Update.
    """
    Sigma = _admm_psd(S=S, W=W, mu=mu_cov, tol=tol_cov, max_iter=max_iter_cov, eps=eps)
    coef = _admm_qpl1(Q=Sigma, p=p, alpha=alpha, mu=mu_coef, tol=tol_coef, max_iter=max_iter_coef)
    return coef


class HMLasso(LinearModel, RegressorMixin):
    """
    Lasso with High Missing Rate.

    Parameters
    ----------
    alpha : float, optional (default=1.0)
        Constant that multiplies the L1 term.

    mu_coef : float, optional (default=1.0)
        Constant that used in augmented Lagrangian function for Lasso.

    mu_cov : float, optional (default=1.0)
        Constant that used in augmented Lagrangian function for covariance estimation.

    normalize : boolean, optional (default=False)
        This parameter is ignored when fit_intercept is set to False.
        If True, the regressors X will be standardized before regression.

    copy_X : boolean, optional (default=True)
        If True, X will be copied; else, it may be overwritten.

    tol_coef : float, optional (default=1e-4)
        The tolerance for Lasso.

    tol_cov : float, optional (default=1e-4)
        The tolerance for covariance estimation.

    max_iter_coef : int, optional (default=1000)
        The maximum number of iterations of Lasso.

    max_iter_cov : int, optional (default=100)
        The maximum number of iterations of covariance estimation.

    eps : float, optional (default=1e-8)
        small positive value used in projection onto PSD.

    Attributes
    ----------
    coef_ : array, shape (n_features,) | (n_targets, n_features)
        parameter vector

    intercept_ : float | array, shape (n_targets,)
        independent term in decision function.

    n_iter_ : int | array-like, shape (n_targets,)
        number of iterations run by admm solver to reach
        the specified tolerance.
    """
    def __init__(
            self,
            alpha: float = 1.0,
            mu_coef: float = 1.0,
            mu_cov: float = 1.0,
            normalize: bool = False,
            copy_X: bool = True,
            tol_coef: float = 1e-4,
            tol_cov: float = 1e-4,
            max_iter_coef: int = 1000,
            max_iter_cov: int = 100,
            eps: float = 1e-8
    ):
        self.alpha = alpha
        self.mu_coef = mu_coef
        self.mu_cov = mu_cov
        self.normalize = normalize
        self.copy_X = copy_X
        self.tol_coef = tol_coef
        self.tol_cov = tol_cov
        self.max_iter_coef = max_iter_coef
        self.max_iter_cov = max_iter_cov
        self.eps = eps

        self.coef_ = None
        self.n_iter_ = None

    def _set_intercept(self, X_offset, y_offset, X_scale):
        """
        Set the intercept_
        """
        self.coef_ = self.coef_ / X_scale
        self.intercept_ = y_offset - np.dot(X_offset, self.coef_.T)

    _preprocess_data = staticmethod(_preprocess_data)

    def fit(self, X: np.ndarray, y: np.ndarray, check_input: bool = False):
        """
        fit.

        Parameters
        ----------
        X : np.ndarray, shape = (n_samples, n_features)
            Missed data.

        y : np.ndarray, shape = (n_samples, ) or (n_samples, n_targets)
            Target.

        check_input : boolean, optional (default=False)
            Allow to bypass several input checking.
        """
        if self.alpha == 0:
            logger.warning("""
                With alpha=0, this algorithm does not converge well. 
                You are advised to use the LinearRegression estimator.
            """)

        if check_input:
            X, y = check_X_y(
                X, y, accept_sparse='csc', accept_large_sparse=False,
                order='F', dtype=[np.float64, np.float32],
                copy=False, force_all_finite='allow-nan',
                multi_output=True, y_numeric=True
            )
            y = check_array(
                y, order='F', copy=False, dtype=X.dtype.type,
                ensure_2d=False
            )
        X, y, X_offset, y_offset, X_scale = self._preprocess_data(X, y, fit_intercept=True,
                                                                  normalize=self.normalize,
                                                                  copy=self.copy_X and not check_input)

        n_samples, n_features = X.shape[:2]

        missed = np.isnan(X)
        not_missed = (~missed).astype(np.float)
        R = not_missed.T.dot(not_missed)

        # centralize
        X[missed] = 0
        if y.ndim == 1:
            y = y[:, np.newaxis]
        n_targets = y.shape[1]

        S = X.T.dot(X) / R
        rho = X.T.dot(y) / np.diag(R).reshape(-1, 1)

        # Update by columns
        w_t = np.empty((n_targets, n_features), dtype=X.dtype)
        n_iter_ = np.empty((n_targets,), dtype=int)
        if n_targets == 1:
            w_t, n_iter_[0] = _update(
                S=S, W=R/n_samples, mu_cov=self.mu_cov, tol_cov=self.tol_cov, max_iter_cov=self.max_iter_cov,
                eps=self.eps, p=-rho[:, 0], alpha=self.alpha, mu_coef=self.mu_coef, tol_coef=self.tol_coef,
                max_iter_coef=self.max_iter_coef
            )
        else:
            results = Parallel(n_jobs=-1, backend='threading')(
                delayed(_update)(
                    S=S, W=R/n_samples, mu_cov=self.mu_cov, tol_cov=self.tol_cov, max_iter_cov=self.max_iter_cov,
                    eps=self.eps, p=-rho[:, k], alpha=self.alpha, mu_coef=self.mu_coef, tol_coef=self.tol_coef,
                    max_iter_coef=self.max_iter_coef
                )
                for k in range(n_targets)
            )
            for k in range(n_targets):
                w_t[k], n_iter_[k] = results[k]

        self.coef_, self.n_iter_ = np.squeeze(w_t), n_iter_.tolist()

        if y.shape[1] == 1:
            self.n_iter_ = self.n_iter_[0]

        self._set_intercept(X_offset, y_offset, X_scale)

        # workaround since _set_intercept will cast self.coef_ into X.dtype
        self.coef_ = np.asarray(self.coef_, dtype=X.dtype)

        return self
