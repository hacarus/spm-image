from unittest import TestCase
from typing import Tuple

from spmimage.decomposition import KSVD

import numpy as np


class TestKSVD(TestCase):

    @staticmethod
    def generate_input(n_samples: int, n_features: int, n_components: int, k0: int) \
            -> Tuple[np.ndarray, np.ndarray]:
        # random dictionary base
        A0 = np.random.randn(n_components, n_features)
        A0 = np.dot(A0, np.diag(1. / np.sqrt(np.diag(np.dot(A0.T, A0)))))

        X = np.zeros((n_samples, n_features))
        for i in range(n_samples):
            # select k0 components from dictionary
            X[i, :] = np.dot(np.random.randn(k0), A0[np.random.permutation(range(n_components))[:k0], :])
        return A0, X

    def test_ksvd(self):
        np.random.seed(0)
        k0 = 4
        n_samples = 512
        n_features = 32
        n_components = 24
        max_iter = 500

        A0, X = self.generate_input(n_samples, n_features, n_components, k0)
        model = KSVD(n_components=n_components, k0=k0, max_iter=max_iter)
        model.fit(X)

        # check error of learning
        self.assertTrue(model.error_[-1] < 10)
        self.assertTrue(model.n_iter_ <= max_iter)

        # check estimated dictionary
        norm = np.linalg.norm(model.components_ - A0, ord='fro')
        self.assertTrue(norm < 15)

        # check reconstructed data
        code = model.transform(X)
        reconstructed = np.dot(code, model.components_)
        reconstruct_error = np.linalg.norm(reconstructed - X, ord='fro')
        self.assertTrue(reconstruct_error < 15)
