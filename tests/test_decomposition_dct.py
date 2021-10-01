import unittest

import numpy as np
import numpy.testing as npt

from spmimage.decomposition import generate_dct_dictionary
from spmimage.decomposition.dct import zig_zag_index


class TestDCT(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_zig_zag_index(self):
        n = 5
        true_matrix = np.array([[0., 1., 5., 6., 14.],
                                [2., 4., 7., 13., 15.],
                                [3., 8., 12., 16., 21.],
                                [9., 11., 17., 20., 22.],
                                [10., 18., 19., 23., 24.]])

        M = np.empty((n, n))
        for k in range(n * n):
            M[zig_zag_index(k, n)] = k

        npt.assert_array_equal(M, true_matrix)

    def test_dct_complete(self):
        n_components = 4
        patch_size = 2
        D = generate_dct_dictionary(n_components, patch_size)
        D22 = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [0.707, -0.707, 0.707, -0.707],
            [0.707, 0.707, -0.707, -0.707],
            [0.5, -0.5, -0.5, 0.5]
        ])
        npt.assert_array_almost_equal(D, D22, 3)

    def test_dct_less(self):
        patch_size = 5

        n_components = 4
        D = generate_dct_dictionary(n_components, patch_size)
        npt.assert_array_equal((n_components, patch_size ** 2), D.shape)

        n_components = 15
        D = generate_dct_dictionary(n_components, patch_size)
        npt.assert_array_equal((n_components, patch_size ** 2), D.shape)

        n_components = 24
        D = generate_dct_dictionary(n_components, patch_size)
        npt.assert_array_equal((n_components, patch_size ** 2), D.shape)


if __name__ == '__main__':
    unittest.main()
