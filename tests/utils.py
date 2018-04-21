from typing import Tuple

import numpy as np


def generate_dictionary_and_samples(n_samples: int, n_features: int, n_components: int, n_nonzero_coefs: int) \
        -> Tuple[np.ndarray, np.ndarray]:
    # random dictionary base
    A0 = np.random.randn(n_components, n_features)
    A0 = np.dot(A0, np.diag(1. / np.sqrt(np.diag(np.dot(A0.T, A0)))))

    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        # select n_nonzero_coefs components from dictionary
        X[i, :] = np.dot(np.random.randn(n_nonzero_coefs),
                         A0[np.random.permutation(range(n_components))[:n_nonzero_coefs], :])
    return A0, X
