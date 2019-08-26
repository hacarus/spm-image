import unittest

from sklearn.preprocessing import normalize
from sklearn.decomposition import sparse_encode
from spmimage.decomposition import sparse_encode_with_mask
import spmimage

import numpy as np

from tests.utils import generate_dictionary_and_samples


class TestDictLearning(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_sparse_encode_with_no_mask(self):
        k0 = 3
        n_samples = 64
        n_features = 32
        n_components = 10

        A0, X = generate_dictionary_and_samples(n_samples, n_features, n_components, k0)
        mask = np.ones(X.shape)

        W1 = sparse_encode(X, A0, algorithm='omp', n_nonzero_coefs=k0)
        W2 = sparse_encode_with_mask(X, A0, mask, algorithm='omp', n_nonzero_coefs=k0)

        # check if W1 and W2 is almost same
        self.assertTrue(abs(np.linalg.norm(X - W1.dot(A0), 'fro') - np.linalg.norm(X - W2.dot(A0), 'fro')) < 1e-8)

    def test_sparse_encode_with_mask(self):
        k0 = 5
        n_samples = 128
        n_features = 64
        n_components = 32

        A0, X = generate_dictionary_and_samples(n_samples, n_features, n_components, k0)
        mask = np.random.rand(X.shape[0], X.shape[1])
        mask = np.where(mask < 0.8, 1, 0)

        W = sparse_encode_with_mask(X, A0, mask, algorithm='omp', n_nonzero_coefs=k0)

        # check error of learning
        # print(np.linalg.norm(mask*(X-W.dot(A0)), 'fro'))
        self.assertTrue(np.linalg.norm(mask * (X - W.dot(A0)), 'fro') < 50)

    def test_sparse_encode(self):
        k0 = 3
        n_samples = 64
        n_features = 32
        n_components = 10

        A0, X = generate_dictionary_and_samples(n_samples, n_features, n_components, k0)
        A0 = normalize(A0)

        W1 = sparse_encode(X, A0, algorithm='omp', n_nonzero_coefs=k0)
        W2 = spmimage.decomposition.sparse_encode(X, A0, algorithm='omp', n_nonzero_coefs=k0)

        # check if W1 and W2 is almost same
        self.assertTrue(abs(np.linalg.norm(X - W1.dot(A0), 'fro') - np.linalg.norm(X - W2.dot(A0), 'fro')) < 1e-8)

    def test_sparse_encode_mp(self):
        k0 = 3
        n_samples = 64
        n_features = 32
        n_components = 10

        A0, X = generate_dictionary_and_samples(n_samples, n_features, n_components, k0)
        A0 = normalize(A0)

        W = spmimage.decomposition.sparse_encode(X, A0, algorithm='mp', n_nonzero_coefs=k0)

        # check error of learning
        # print(np.linalg.norm(X - W.dot(A0), 'fro'))
        self.assertTrue(np.linalg.norm(X - W.dot(A0), 'fro') < np.linalg.norm(X))

    def test_parallel_sparse_encode_mp(self):
        k0 = 3
        n_samples = 64
        n_features = 32
        n_components = 10

        A0, X = generate_dictionary_and_samples(n_samples, n_features, n_components, k0)
        A0 = normalize(A0)

        W = spmimage.decomposition.sparse_encode(X, A0, algorithm='mp', n_nonzero_coefs=k0, n_jobs=2)

        # check error of learning
        # print(np.linalg.norm(X - W.dot(A0), 'fro'))
        self.assertTrue(np.linalg.norm(X - W.dot(A0), 'fro') < np.linalg.norm(X))


if __name__ == '__main__':
    unittest.main()
