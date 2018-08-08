from logging import getLogger
from abc import abstractmethod

import numpy as np
from scipy import sparse

from sklearn.utils import check_array, check_X_y
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel, _pre_fit
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.model_selection import check_cv
from sklearn.externals.joblib import Parallel, delayed

logger = getLogger(__name__)


def _soft_threshold(X: np.ndarray, thresh: float) -> np.ndarray:
    return np.where(np.abs(X) <= thresh, 0, X - thresh * np.sign(X))


def _cost_function(X, y, w, z, alpha):
    n_samples = X.shape[0]
    return np.linalg.norm(y - X.dot(w)) / n_samples + alpha * np.sum(np.abs(z))


def admm_path(X, y, Xy=None, alphas=None, eps=1e-3, n_alphas=100, rho=1.0, fit_intercept=True,
              normalize=False, copy_X=True, max_iter=1000, tol=1e-04):
    _, n_features = X.shape
    multi_output = False
    n_iters = []

    if y.ndim != 1:
        multi_output = True
        _, n_outputs = y.shape

    if alphas is None:
        alphas = _alpha_grid(X, y, Xy=Xy, l1_ratio=1.0, eps=eps, n_alphas=n_alphas)
    else:
        alphas = np.sort(alphas)[::-1]
        n_alphas = len(alphas)

    if not multi_output:
        coefs = np.zeros((n_features, n_alphas), dtype=X.dtype)
    else:
        coefs = np.zeros((n_features, n_outputs, n_alphas), dtype=X.dtype)

    for i, alpha in enumerate(alphas):
        clf = LassoADMM(alpha=alpha, rho=rho, fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, max_iter=max_iter, tol=tol)
        clf.fit(X, y)
        coefs[..., i] = clf.coef_
        n_iters.append(clf.n_iter_)

    return alphas, coefs, n_iters

def _path_residuals(X, y, train, test, path, path_params, alphas=None,
                    X_order=None, dtype=None):
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]

    # admm_path dosen't have fit_intercept and normalize as args.
    # so, use default value in LassoADMMCV for fit_intercept and normalize
    fit_intercept = True
    normalize = False

    if y.ndim == 1:
        precompute = 'auto'
    else:
        # No Gram variant of multi-task exists right now.
        # Fall back to default enet_multitask
        precompute = False

    X_train, y_train, X_offset, y_offset, X_scale, precompute, Xy = \
        _pre_fit(X_train, y_train, None, precompute, normalize, fit_intercept,
                 copy=False)

    path_params = path_params.copy()
    path_params['Xy'] = Xy
    path_params['alphas'] = alphas

    # Do the ordering and type casting here, as if it is done in the path,
    # X is copied and a reference is kept here
    X_train = check_array(X_train, 'csc', dtype=dtype, order=X_order)
    alphas, coefs, _ = path(X_train, y_train, **path_params)
    del X_train, y_train

    if y.ndim == 1:
        # Doing this so that it becomes coherent with multioutput.
        coefs = coefs[np.newaxis, :, :]
        y_offset = np.atleast_1d(y_offset)
        y_test = y_test[:, np.newaxis]

    if normalize:
        nonzeros = np.flatnonzero(X_scale)
        coefs[:, nonzeros] /= X_scale[nonzeros][:, np.newaxis]

    intercepts = y_offset[:, np.newaxis] - np.dot(X_offset, coefs)
    if sparse.issparse(X_test):
        n_order, n_features, n_alphas = coefs.shape
        # Work around for sparse matrices since coefs is a 3-D numpy array.
        coefs_feature_major = np.rollaxis(coefs, 1)
        feature_2d = np.reshape(coefs_feature_major, (n_features, -1))
        X_test_coefs = safe_sparse_dot(X_test, feature_2d)
        X_test_coefs = X_test_coefs.reshape(X_test.shape[0], n_order, -1)
    else:
        X_test_coefs = safe_sparse_dot(X_test, coefs)
    residues = X_test_coefs - y_test[:, :, np.newaxis]
    residues += intercepts
    this_mses = ((residues ** 2).mean(axis=0)).mean(axis=0)

    return this_mses


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
    inv_matrix_XTy = inv_matrix.dot(X.T).dot(y) / n_samples
    inv_matrix_DT = inv_matrix.dot(rho * D.T)
    threshold = alpha / rho

    n_iter_ = []
    # Update ADMM parameters by columns
    for k in range(n_targets):
        # initial cost
        cost = _cost_function(X, y[:, k], w_t[:, k], z_t[:, k], alpha)
        for t in range(max_iter):
            # Update
            w_t[:, k] = inv_matrix_XTy[:, k] \
                        + inv_matrix_DT.dot(z_t[:, k] - h_t[:, k] / rho)
            Dw_t = D.dot(w_t[:, k])
            z_t[:, k] = _soft_threshold(Dw_t + h_t[:, k] / rho, threshold)
            h_t[:, k] += rho * (Dw_t - z_t[:, k])

            # after cost
            pre_cost = cost
            cost = _cost_function(X, y[:, k], w_t[:, k], z_t[:, k], alpha)
            gap = np.abs(cost - pre_cost)
            if gap < tol:
                break
        n_iter_.append(t)
    return np.squeeze(w_t), n_iter_


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
        self.coef_, self.n_iter_ = _admm(X, y, D, self.alpha, self.rho, self.tol, self.max_iter)

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


class LassoADMMCV(LinearModel, RegressorMixin):

    path = staticmethod(admm_path)
    
    def __init__(self, eps=1e-3, n_alphas=100, alphas=None, rho=1.0,  fit_intercept=True,
                 normalize=False, copy_X=True, max_iter=1000, tol=1e-4, cv=None, verbose=False, n_jobs=1):
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.rho = rho
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X, y):
        y = check_array(y, copy=False, dtype=[np.float64, np.float32],
                        ensure_2d=False)
        if y.shape[0] == 0:
            raise ValueError("y has 0 samples: %r" % y)

        model = LassoADMM()
        
        copy_X = self.copy_X and self.fit_intercept

        if isinstance(X, np.ndarray) or sparse.isspmatrix(X):
            # Keep a reference to X
            reference_to_old_X = X
            # Let us not impose fortran ordering so far: it is
            # not useful for the cross-validation loop and will be done
            # by the model fitting itself
            X = check_array(X, 'csc', copy=False)
            if sparse.isspmatrix(X):
                if (hasattr(reference_to_old_X, "data") and
                   not np.may_share_memory(reference_to_old_X.data, X.data)):
                    # X is a sparse matrix and has been copied
                    copy_X = False
            elif not np.may_share_memory(reference_to_old_X, X):
                # X has been copied
                copy_X = False
            del reference_to_old_X
        else:
            X = check_array(X, 'csc', dtype=[np.float64, np.float32],
                            order='F', copy=copy_X)
            copy_X = False

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have inconsistent dimensions (%d != %d)"
                             % (X.shape[0], y.shape[0]))

        
        # All LinearModelCV parameters except 'cv' are acceptable
        path_params = self.get_params()
        
        copy_X = self.copy_X and self.fit_intercept

        if isinstance(X, np.ndarray) or sparse.isspmatrix(X):
            # Keep a reference to X
            reference_to_old_X = X
            # Let us not impose fortran ordering so far: it is
            # not useful for the cross-validation loop and will be done
            # by the model fitting itself
            X = check_array(X, 'csc', copy=False)
            if sparse.isspmatrix(X):
                if (hasattr(reference_to_old_X, "data") and
                   not np.may_share_memory(reference_to_old_X.data, X.data)):
                    # X is a sparse matrix and has been copied
                    copy_X = False
            elif not np.may_share_memory(reference_to_old_X, X):
                # X has been copied
                copy_X = False
            del reference_to_old_X
        else:
            X = check_array(X, 'csc', dtype=[np.float64, np.float32],
                            order='F', copy=copy_X)
            copy_X = False

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have inconsistent dimensions (%d != %d)"
                             % (X.shape[0], y.shape[0]))

        
        # All LinearModelCV parameters except 'cv' are acceptable
        path_params = self.get_params()
        
        path_params.pop('cv', None)
        path_params.pop('n_jobs', None)
        path_params.pop('verbose', None)
        
        alphas = self.alphas

        if alphas is None:
            alphas = _alpha_grid(X, y, l1_ratio=1.0, eps=self.eps,
                                 n_alphas=self.n_alphas)
        else:
            alphas = np.sort(alphas)[::-1]
            
        # We want n_alphas to be the number of alphas used for each l1_ratio.
        n_alphas = len(alphas)
        path_params.update({'n_alphas': n_alphas})

        # init cross-validation generator
        cv = check_cv(self.cv)

        # Compute path for all folds and compute MSE to get the best alpha
        folds = list(cv.split(X, y))
        best_mse = np.inf

        # We do a double for loop folded in one, in order to be able to
        # iterate in parallel on l1_ratio and folds
        jobs = (delayed(_path_residuals)(X, y, train, test, self.path,
                                         path_params, alphas=alphas,
                                         X_order='F', dtype=X.dtype.type)
                for train, test in folds)        
        mse_paths = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                             backend="threading")(jobs)
        mse_paths = np.reshape(mse_paths, (len(folds), -1))
        mean_mse = np.mean(mse_paths, axis=0)
        self.mse_path_ = np.squeeze(np.rollaxis(mse_paths, 1, 0))

        best_alpha = alphas[np.argmin(mean_mse)]
        
        self.alpha_ = best_alpha
        if self.alphas is None:
            self.alphas_ = np.asarray(alphas)
        # Remove duplicate alphas in case alphas is provided.
        else:
            self.alphas_ = np.asarray(alphas)

        # Refit the model with the parameters selected
        common_params = dict((name, value)
                             for name, value in self.get_params().items()
                             if name in model.get_params())
        model.set_params(**common_params)
        model.alpha = best_alpha
        model.fit(X, y)
        
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_
        self.n_iter_ = model.n_iter_

        return self
