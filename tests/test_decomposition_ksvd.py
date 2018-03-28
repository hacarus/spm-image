from unittest import TestCase
from typing import Tuple

from spmimage.decomposition import KSVD

import numpy as np


class TestKSVD(TestCase):

    def generate_input(self, dict_size: Tuple[int, int], k0: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        # random dictionary base
        A0 = np.random.randn(*dict_size)
        X = np.zeros((dict_size[0], n_samples))
        for i in range(n_samples):
            # select k0 components from dictionary
            X[:, i] = np.dot(A0[:, np.random.permutation(range(dict_size[1]))[:k0]], np.random.randn(k0))
        return A0, X.T

    def test_ksvd(self):
        np.random.seed(0)
        k0 = 4
        n_samples = 512
        dict_size = (24, 32)
        max_iter = 100
        A0, X = self.generate_input(dict_size, k0, n_samples)
        model = KSVD(n_components=dict_size[1], k0=k0, max_iter=max_iter)
        model.fit(X)

        norm = np.linalg.norm(model.components_ - A0.T, ord='fro')

        self.assertTrue(model.error_[-1] < 75)
        self.assertTrue(norm < 50)
        self.assertTrue(model.n_iter_ <= max_iter)

        code = model.transform(X)
        reconstructed = np.dot(code, model.components_)
        reconstruct_error =  np.linalg.norm(reconstructed - X, ord='fro')
        print(reconstruct_error)

