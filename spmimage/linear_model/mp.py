"""Matching pursuit algorithms
"""

import numpy as np
from sklearn.utils import check_array

def matching_pursuit(dictionary, signal, copy_dictionary=True, copy_signal=True, tol=None):
    """Matching Pursuit (MP)
    
    Solves n_targets Matching Pursuit problems.

    Parameters
    ----------
    dictionary : array, shape (n_samples, n_features)
        Input data. Columns are assumed to have unit norm.

    signal : array, shape (n_samples,) or (n_samples, n_targets)
        Input targets.

    copy_X : bool, optional
        Whether the input data must be copied by the algorithm. A false
        value is only helpful if it is already Fortran-ordered, otherwise a
        copy is made anyway.

    copy_y : bool, optional
        Whether the covariance vector Xy must be copied by the algorithm.
        If False, it may be overwritten.
    """
    dictionary = check_array(dictionary, order='F', copy=copy_dictionary)
    if copy_signal:
        signal = signal.copy()
    if signal.ndim == 1:
        n_targets = 1
    else:
        n_targets = signal.shape[1]

    n_active = 0

    n_samples = dictionary.shape[0]
    n_features = dictionary.shape[1]
    coefs = np.zeros((n_targets, n_samples))
    while(True):
        inners = dictionary.dot(signal.T)
        max_ids = np.argmax(np.abs(inners), axis=0)
        for col, max_id in enumerate(max_ids):
            coefs[col, max_id] = inners[max_id, col]
            signal[col] -= inners[max_id, col] * dictionary[max_id]
        n_active += 1
        if tol is not None:
            if np.linalg.norm(signal) <= tol:
                break
        if n_active == n_features:
            break
    return coefs
