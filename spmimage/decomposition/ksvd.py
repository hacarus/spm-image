from logging import getLogger

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.decomposition.dict_learning import SparseCodingMixin, sparse_encode
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from .dict_learning import sparse_encode_with_mask

logger = getLogger(__name__)


def _ksvd(Y: np.ndarray, n_components: int, n_nonzero_coefs: int, max_iter: int, tol: float,
          dict_init: np.ndarray = None, mask: np.ndarray = None, n_jobs: int = 1, method: str = None):
    """_ksvd
    Finds a dictionary that can be used to represent data using a sparse code.
    Solves the optimization problem:
        argmin \sum_{i=1}^M || y_i - w_iH ||_2^2 such that ||w_i||_0 <= k_0 for all 1 <= i <= M
        ({w_i}_{i=1}^M, H)

    **Note**
    Y ~= WH = code * dictionary

    Parameters:
    ------------
        Y : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.
        n_components : int,
            number of dictionary elements to extract
        n_nonzero_coefs : int,
            number of non-zero elements of sparse coding
        max_iter : int,
            maximum number of iterations to perform
        tol : float,
            tolerance for numerical error
        dict_init : array of shape (n_components, n_features),
            initial values for the dictionary, for warm restart
        mask : array-like, shape (n_samples, n_features),
            value at (i,j) in mask is not 1 indicates value at (i,j) in Y is missing
        n_jobs : int, optional
            Number of parallel jobs to run.
    Returns:
    ---------
        code : array of shape (n_samples, n_components)
            The sparse code factor in the matrix factorization.
        dictionary : array of shape (n_components, n_features),
            The dictionary factor in the matrix factorization.
        errors : array
            Vector of errors at each iteration.
        n_iter : int
            Number of iterations run. Returned only if `return_n_iter` is
            set to True.
    """

    W = np.zeros((Y.shape[0], n_components))
    if dict_init is None:
        H = np.random.rand(n_components, Y.shape[1])
        H = np.dot(H, np.diag(1. / np.sqrt(np.diag(np.dot(H.T, H)))))
    else:
        H = dict_init

    errors = [np.linalg.norm(Y - W.dot(H), 'fro')]
    k = -1
    for k in range(max_iter):
        if mask is None:
            W = sparse_encode(Y, H, algorithm='omp', n_nonzero_coefs=n_nonzero_coefs, n_jobs=n_jobs)
        else:
            W = sparse_encode_with_mask(Y, H, mask, algorithm='omp', n_nonzero_coefs=n_nonzero_coefs, n_jobs=n_jobs)

        for j in range(n_components):
            x = W[:, j] != 0
            if np.sum(x) == 0:
                continue

            if method == 'approximate':
                H[j, :] = 0
                error = Y[x, :] - np.dot(W[x, :], H)
                g = W[x, j].T
                d = error.T.dot(g)
                d /= np.linalg.norm(d)
                g = error.dot(d)
                W[x, j] = g.T
                H[j, :] = d
            else:
                # normal ksvd
                W[x, j] = 0
                error = Y[x, :] - np.dot(W[x, :], H)
                U, s, V = np.linalg.svd(error)
                W[x, j] = U[:, 0] * s[0]
                H[j, :] = V.T[:, 0]

        errors.append(np.linalg.norm(Y - W.dot(H), 'fro'))
        if np.abs(errors[-1] - errors[-2]) < tol:
            break

    return W, H, errors, k + 1


class KSVD(BaseEstimator, SparseCodingMixin):
    """ K-SVD
    Finds a dictionary that can be used to represent data using a sparse code.
    Solves the optimization problem:
        argmin \sum_{i=1}^M || y_i - Ax_i ||_2^2 such that ||x_i||_0 <= k_0 for all 1 <= i <= M
        (A,{x_i}_{i=1}^M)

    Parameters
    ----------
        n_components : int,
            number of dictionary elements to extract
        max_iter : int,
            maximum number of iterations to perform
        tol : float,
            tolerance for numerical error
        missing_value : float,
            missing value in the data
        transform_algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}
            Algorithm used to transform the data
            lars: uses the least angle regression method (linear_model.lars_path)
            lasso_lars: uses Lars to compute the Lasso solution
            lasso_cd: uses the coordinate descent method to compute the
            Lasso solution (linear_model.Lasso). lasso_lars will be faster if
            the estimated components are sparse.
            omp: uses orthogonal matching pursuit to estimate the sparse solution
            threshold: squashes to zero all coefficients less than alpha from
            the projection ``dictionary * X'``
            .. versionadded:: 0.17
               *lasso_cd* coordinate descent method to improve speed.
        transform_n_nonzero_coefs : int, ``0.1 * n_features`` by default
            Number of nonzero coefficients to target in each column of the
            solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
            and is overridden by `alpha` in the `omp` case.
        transform_alpha : float, 1. by default
            If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
            penalty applied to the L1 norm.
            If `algorithm='threshold'`, `alpha` is the absolute value of the
            threshold below which coefficients will be squashed to zero.
            If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
            the reconstruction error targeted. In this case, it overrides
            `n_nonzero_coefs`.
        n_jobs : int,
            number of parallel jobs to run
        split_sign : bool, False by default
            Whether to split the sparse feature vector into the concatenation of
            its negative part and its positive part. This can improve the
            performance of downstream classifiers.
        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
        method : {'approximate': Approximate KSVD, 'normal': normal KSVD}, 'approximate' by default

    Attributes
    ----------
        components_ : array, [n_components, n_features]
            dictionary atoms extracted from the data
        error_ : array
            vector of errors at each iteration
        n_iter_ : int
            Number of iterations run.
    **References:**
        Elad, Michael, and Michal Aharon.
        "Image denoising via sparse and redundant representations over learned dictionaries."
        IEEE Transactions on Image processing 15.12 (2006): 3736-3745.
    ----------

    """

    def __init__(self, n_components=None, max_iter=1000, tol=1e-8,
                 missing_value=None, transform_algorithm='omp',
                 transform_n_nonzero_coefs=None,
                 transform_alpha=None, n_jobs=1,
                 split_sign=False, random_state=None, method='approximate'):
        self._set_sparse_coding_params(n_components, transform_algorithm,
                                       transform_n_nonzero_coefs,
                                       transform_alpha, split_sign, n_jobs)
        self.max_iter = max_iter
        self.tol = tol
        self.missing_value = missing_value
        self.random_state = random_state
        self.method = method
        self.components_ = None

    def fit(self, X, y=None):
        """Fit the model from data in X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.
        y : Ignored
        Returns
        -------
        self : object
            Returns the object itself
        """

        # Input validation on an array, list, sparse matrix or similar.
        # By default, the input is converted to an at least 2D numpy array. If the dtype of the array is object, attempt converting to float, raising on failure.
        X = check_array(X)
        n_samples, n_features = X.shape
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        mask = None
        if self.missing_value is not None:
            mask = np.where(X == self.missing_value, 0, 1)

        # initialize dictionary
        dict_init = None
        if self.components_ is not None and self.components_.shape == (n_components, n_features):
            # Warm Start
            dict_init = self.components_

        code, self.components_, self.error_, self.n_iter_ = _ksvd(
            X, n_components, self.transform_n_nonzero_coefs,
            max_iter=self.max_iter, tol=self.tol,
            dict_init=dict_init, mask=mask, n_jobs=self.n_jobs, method=self.method)

        return self

    def transform(self, X):
        """Encode the data as a sparse combination of the dictionary atoms.
        Coding method is determined by the object parameter
        `transform_algorithm`.
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Test data to be transformed, must have the same number of
            features as the data used to train the model.
        Returns
        -------
        code : array, shape (n_samples, n_components)
            Transformed data
        """
        if self.missing_value is not None:
            check_is_fitted(self, 'components_')

            X = check_array(X)

            mask = np.where(X == self.missing_value, 0, 1)

            code = sparse_encode_with_mask(
                X, self.components_, mask, algorithm=self.transform_algorithm,
                n_nonzero_coefs=self.transform_n_nonzero_coefs,
                alpha=self.transform_alpha, n_jobs=self.n_jobs)

            if self.split_sign:
                # feature vector is split into a positive and negative side
                n_samples, n_features = code.shape
                split_code = np.empty((n_samples, 2 * n_features))
                split_code[:, :n_features] = np.maximum(code, 0)
                split_code[:, n_features:] = -np.minimum(code, 0)
                code = split_code
            return code

        else:
            return super().transform(X)
