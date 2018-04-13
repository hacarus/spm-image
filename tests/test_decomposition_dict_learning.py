import unittest
from typing import Tuple

from sklearn.decomposition import sparse_encode
from spmimage.decomposition import sparse_encode_with_mask

import numpy as np


def generate_dictionary_and_samples(n_samples: int, n_features: int, n_components: int, k0: int) \
        -> Tuple[np.ndarray, np.ndarray]:
    # random dictionary base
    A0 = np.random.randn(n_components, n_features)
    A0 = np.dot(A0, np.diag(1. / np.sqrt(np.diag(np.dot(A0.T, A0)))))

    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        # select k0 components from dictionary
        X[i, :] = np.dot(np.random.randn(k0), A0[np.random.permutation(range(n_components))[:k0], :])
    return A0, X


class Testdict_learning(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_sparse_encode_with_mask_ones(self):
        k0 = 3
        n_samples = 64
        n_features = 32
        n_components = 10
        max_iter = 100

        A0, X = generate_dictionary_and_samples(n_samples, n_features, n_components, k0)
        mask = np.ones(X.shape)

        W1 = sparse_encode(X, A0, algorithm='omp', n_nonzero_coefs=k0, n_jobs=1)
        W2 = sparse_encode_with_mask(X, A0, mask, algorithm='omp', n_nonzero_coefs=k0, n_jobs=1)

        # check error of learning
        self.assertTrue(abs(np.linalg.norm(X-W1.dot(A0), 'fro') - np.linalg.norm(X-W2.dot(A0), 'fro')) < 1e-8)

    def test_sparse_encode_with_mask_ones(self):
        k0 = 5
        n_samples = 128
        n_features = 64
        n_components = 32
        max_iter = 100

        A0, X = generate_dictionary_and_samples(n_samples, n_features, n_components, k0)
        mask = np.random.rand(X.shape[0], X.shape[1])
        mask = np.where(mask < 0.8, 1, 0)

        W = sparse_encode_with_mask(X, A0, mask, algorithm='omp', n_nonzero_coefs=k0, n_jobs=1)

        # check error of learning
        #print(np.linalg.norm(mask*(X-W.dot(A0)), 'fro'))
        self.assertTrue(np.linalg.norm(mask*(X-W.dot(A0)), 'fro') < 50)

if __name__ == '__main__':
    unittest.main()
