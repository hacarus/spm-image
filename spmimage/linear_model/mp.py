"""Matching pursuit algorithms
"""

import numpy as np
from sklearn.utils import check_array
from sklearn.preprocessing import normalize

def matching_pursuit(dictionary, signal, n_nonzero_coefs=None, copy_dictionary=True, copy_signal=True, tol=None):
    """Matching Pursuit (MP)
    
    Solves n_targets Matching Pursuit problems.
    The goal is to find a sparse array `coefs` such that::
        signal ~= coefs * dictionary
    Parameters
    ----------
    dictionary : array, shape (n_components, n_features)
        Input data. Columns are assumed to have unit norm.

    signal : array, shape (n_samples, n_features)
        Input targets.

    n_nonzero_coefs : int
        Targeted number of non-zero elements

    copy_X : bool, optional
        Whether the input data must be copied by the algorithm. A false
        value is only helpful if it is already Fortran-ordered, otherwise a
        copy is made anyway.

    copy_y : bool, optional
        Whether the covariance vector Xy must be copied by the algorithm.
        If False, it may be overwritten.

    tol : float
        Maximum norm of the residual.
    """
    normalized = normalize(dictionary)
    if abs(np.linalg.norm(normalized - dictionary, 'fro')) > len(dictionary) * 1e-8:
        raise ValueError("All columns of the dictionary must have unit norm.")
    if tol is not None and tol < 0:
        raise ValueError("Epsilon cannot be negative")
    dictionary = check_array(dictionary, order='F', copy=copy_dictionary)
    if copy_signal:
        signal = signal.copy()

    n_active = 0

    n_samples = signal.shape[0]
    n_components = dictionary.shape[0]
    n_features = dictionary.shape[1]
    coefs = np.zeros((n_samples, n_components))
    while(True):
        inners = dictionary.dot(signal.T)
        max_ids = np.argmax(np.abs(inners), axis=0)
        for i, max_id in enumerate(max_ids):
            coefs[i, max_id] += inners[max_id, i]
            signal[i] -= inners[max_id, i] * dictionary[max_id]
        n_active += 1
        if tol is not None and np.linalg.norm(signal) <= tol:
                break
        if n_active == n_features or n_active == n_nonzero_coefs:
            break
    return coefs
