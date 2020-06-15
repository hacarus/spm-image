import numpy as np

from sklearn.utils import check_array
from sklearn.decomposition import sparse_encode


def sparse_encode_with_mask(X, dictionary, mask, **kwargs):
    """sparse_encode_with_mask
    Finds a sparse coding that represent data with given dictionary.

    X ~= code * dictionary

    Parameters:
    ------------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        dictionary : array of shape (n_components, n_features),
            The dictionary factor

        mask : array-like, shape (n_samples, n_features),
            value at (i,j) in mask is not 1 indicates value at (i,j) in X is missing

        verbose : bool
            Degree of output the procedure will print.

        **kwargs :
            algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}
                lars: uses the least angle regression method (linear_model.lars_path)
                lasso_lars: uses Lars to compute the Lasso solution
                lasso_cd: uses the coordinate descent method to compute the
                Lasso solution (linear_model.Lasso). lasso_lars will be faster if
                the estimated components are sparse.
                omp: uses orthogonal matching pursuit to estimate the sparse solution
                threshold: squashes to zero all coefficients less than regularization
                from the projection dictionary * data'
            n_nonzero_coefs : int,
                number of non-zero elements of sparse coding
            n_jobs : int, optional
                Number of parallel jobs to run.

    Returns:
    ---------
        code : array of shape (n_components, n_features)
            The sparse codes
    """
    code = np.zeros((X.shape[0], dictionary.shape[0]))
    for idx in range(X.shape[0]):
        code[idx, :] = sparse_encode(X[idx, :][mask[idx, :] == 1].reshape(1, -1),
                                     dictionary[:, mask[idx, :] == 1],
                                     **kwargs)
    return code


def _shrinkage_map_for_l21_norm(X, gamma):
    """
    Shrinkage mapping for l2,1-norm minimization.
    """
    norm_X = np.linalg.norm(X, axis=0).reshape(1, -1)
    norm_X[norm_X == 0] = gamma
    return np.maximum(1 - gamma / norm_X, 0) * X


def sparse_encode_with_l21_norm(X, dictionary, max_iter=30, alpha=1.0, tau=1.0, check_input=True):
    """
    Finds a sparse coding that represent data with given dictionary.

    X ~= code * dictionary

    Minimizes the following objective function:

            1 / (2 * n_samples) * ||X - WD||^2_F + alpha * ||W||_2,1

    To solve this problem, ADMM uses augmented Lagrangian

            1 / (2 * n_samples) * ||X - WD||^2_F + alpha * ||Y||_2,1
            + U^T (W - Y) + tau / (2 * n_samples) * ||W - Y||^2_F

    where U is Lagrange multiplier and tau is tuning parameter.

    Parameters:
    ------------
        X : array-like, shape (n_samples, n_features)
            Training matrix

        dictionary : array of shape (n_components, n_features)
            The dictionary factor

        max_iter : int, optional (default=1000)
            Maximum number of iterations

        alpha : float, optional (default=1.0)
            The penalty applied to the L2-1 norm

        check_input : boolean, optional (default=True)
            If False, the input arrays X and dictionary will not be checked.

        tau : float, optional (default=1.0)
            The penalty applied to the augmented Lagrangian function

    Returns:
    ---------
        Y : array of shape (n_components, n_features)
            The sparse codes
    """
    if check_input:
        dictionary = check_array(dictionary)
        X = check_array(X)

    n_components = dictionary.shape[0]
    n_samples = X.shape[0]
    tau /= n_samples
    inv_matrix = np.linalg.inv(dictionary @ dictionary.T / n_samples + tau * np.identity(n_components))
    XD = X @ dictionary.T / n_samples
    tau_inv = 1 / tau
    alpha_tau = alpha * tau_inv

    # initialize
    W = XD @ inv_matrix
    Y = W.copy()
    U = np.zeros_like(W)

    for _ in range(max_iter):
        W = (XD + tau * Y - U) @ inv_matrix
        Y = _shrinkage_map_for_l21_norm(W + tau_inv * U, alpha_tau)
        U = U + tau * (W - Y)
    return Y
