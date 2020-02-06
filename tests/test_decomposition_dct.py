import unittest

import numpy as np
import numpy.testing as npt

from spmimage.decomposition import generate_dct_dictionary


class TestKSVD(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_small_dct_dict(self):
        patch_size = 2
        sqrt_dict_size = 2
        D = generate_dct_dictionary(patch_size, sqrt_dict_size)
        D22 = np.array([[1., 1., 1., 1.],
                        [0.5, -0.5, 0.5, -0.5],
                        [0.5, 0.5, -0.5, -0.5],
                        [0.25, -0.25, -0.25, 0.25]]
                       )
        npt.assert_array_almost_equal(D, D22)

    def test_shape_dct_dict(self):
        patch_size = 5
        sqrt_dict_size = 7
        D = generate_dct_dictionary(patch_size, sqrt_dict_size)
        npt.assert_array_equal((7 ** 2, 5 ** 2), D.shape)


if __name__ == '__main__':
    unittest.main()
