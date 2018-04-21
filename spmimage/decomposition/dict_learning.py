import numpy as np

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
