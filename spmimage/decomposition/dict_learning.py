import numpy as np
import sklearn
from sklearn.externals.joblib import Parallel, delayed, cpu_count
from sklearn.utils import (check_array, check_random_state, gen_even_slices,
                     gen_batches, _get_n_jobs)
from ..linear_model import matching_pursuit


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
        code[idx, :] = sklearn.decomposition.sparse_encode(
            X[idx, :][mask[idx, :] == 1].reshape(1, -1),
            dictionary[:, mask[idx, :] == 1], **kwargs)
    return code

def sparse_encode(X, dictionary, gram=None, cov=None, algorithm='lasso_lars',
                  n_nonzero_coefs=None, alpha=None, copy_cov=True, init=None,
                  max_iter=1000, n_jobs=1, check_input=True, verbose=0):
    """Sparse coding
    Each row of the result is the solution to a sparse coding problem.
    The goal is to find a sparse array `code` such that::
        X ~= code * dictionary
    Read more in the :ref:`User Guide <SparseCoder>`.
    Parameters
    ----------
        X : array of shape (n_samples, n_features)
            Data matrix
        dictionary : array of shape (n_components, n_features)
            The dictionary matrix against which to solve the sparse coding of
            the data. Some of the algorithms assume normalized rows for meaningful
            output (particularly for 'omp' and 'mp').
        gram : array, shape=(n_components, n_components)
            Precomputed Gram matrix, dictionary * dictionary'
        cov : array, shape=(n_components, n_samples)
            Precomputed covariance, dictionary' * X
        algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold', 'mp'}
            lars: uses the least angle regression method (linear_model.lars_path)
            lasso_lars: uses Lars to compute the Lasso solution
            lasso_cd: uses the coordinate descent method to compute the
            Lasso solution (linear_model.Lasso). lasso_lars will be faster if
            the estimated components are sparse.
            omp: uses orthogonal matching pursuit to estimate the sparse solution
            threshold: squashes to zero all coefficients less than alpha from
            the projection dictionary * X'
            mp: uses matching pursuit to estimate the sparse solution
        n_nonzero_coefs : int, 0.1 * n_features by default
            Number of nonzero coefficients to target in each column of the
            solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
            and is overridden by `alpha` in the `omp` case.
        alpha : float, 1. by default
            If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
            penalty applied to the L1 norm.
            If `algorithm='threshold'`, `alpha` is the absolute value of the
            threshold below which coefficients will be squashed to zero.
            If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
            the reconstruction error targeted. In this case, it overrides
            `n_nonzero_coefs`.
        copy_cov : boolean, optional
            Whether to copy the precomputed covariance matrix; if False, it may be
            overwritten.
        init : array of shape (n_samples, n_components)
            Initialization value of the sparse codes. Only used if
            `algorithm='lasso_cd'`.
        max_iter : int, 1000 by default
            Maximum number of iterations to perform if `algorithm='lasso_cd'`.
        n_jobs : int or None, optional (default=None)
            Number of parallel jobs to run.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
        check_input : boolean, optional
            If False, the input arrays X and dictionary will not be checked.
        verbose : int, optional
            Controls the verbosity; the higher, the more messages. Defaults to 0.
    Returns
    -------
        code : array of shape (n_samples, n_components)
            The sparse codes
    """
    if algorithm != 'mp':
        return sklearn.decomposition.sparse_encode(
            X, dictionary, gram, cov, algorithm, n_nonzero_coefs, alpha, copy_cov,
            init, max_iter, n_jobs, check_input, verbose)
    elif check_input:
        dictionary = check_array(dictionary)
        X = check_array(X)

    n_samples, n_features = X.shape
    n_components = dictionary.shape[0]

    if n_nonzero_coefs is None:
        n_nonzero_coefs = min(max(n_features / 10, 1), n_components)

    if n_jobs == 1:
        code = matching_pursuit(dictionary=dictionary, signal=X, n_nonzero_coefs=n_nonzero_coefs)
        return code

    # Enter parallel code block
    code = np.empty((n_samples, n_components))
    slices = list(gen_even_slices(n_samples, _get_n_jobs(n_jobs)))

    code_views = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(matching_pursuit)(dictionary=dictionary, signal=X[this_slice], n_nonzero_coefs=n_nonzero_coefs) 
        for this_slice in slices)
    for this_slice, this_view in zip(slices, code_views):
        code[this_slice] = this_view
    return code
