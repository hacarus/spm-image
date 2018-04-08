import numpy as np

from sklearn.decomposition import sparse_encode
from sklearn.externals.joblib import Parallel, delayed


def sparse_encode_with_mask(X, dictionary, mask, n_jobs=1, verbose=False, **kwargs):
    W = np.zeros((X.shape[0], dictionary.shape[0]))
    codes = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(sparse_encode)(
            X[idx, :][mask[idx, :] == 1].reshape(1, -1),
            dictionary[:, mask[idx, :] == 1],
            **kwargs
        ) for idx in range(X.shape[0]))
    for idx, code in zip(range(X.shape[0]), codes):
        W[idx, :] = code
    return W
