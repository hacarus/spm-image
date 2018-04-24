import unittest

import numpy as np
from spmimage.preprocessing import WhiteningScaler
from numpy.testing import assert_array_almost_equal


class TestWhitening(unittest.TestCase):
    def setUp(self):
        self.identity_matrix = np.identity(3, dtype=int)

    def test_fit_error(self):
        X = self.identity_matrix
        self.assertEqual(np.linalg.matrix_rank(X - np.mean(X, axis=0)), 2)

        model = WhiteningScaler()
        self.assertRaises(ValueError, model.fit, X)

        model = WhiteningScaler(eps=-1, thresholding='normalize')
        self.assertRaises(ValueError, model.fit, X)

        model = WhiteningScaler(eps=-1, thresholding='drop_minute')
        self.assertRaises(ValueError, model.fit, X)

    def test_fit_normalize(self):
        X = self.identity_matrix

        model = WhiteningScaler(eps=1e-6, thresholding='normalize')
        actual = model.fit_transform(X)
        expected = np.array([[-1.15206e+00, -7.80281e-02, 2.04709e-10],
                             [6.43604e-01, -9.58699e-01, 5.91895e-11],
                             [5.08455e-01, 1.03673e+00, -5.42803e-11]])
        # assert_array_almost_equal(actual, expected, decimal=4)
        assert_array_almost_equal(np.cov(actual.T), np.diag([1, 1, 0]), decimal=4)
        assert_array_almost_equal(X, model.inverse_transform(actual))

    def test_fit_drop_minute(self):
        X = self.identity_matrix

        model = WhiteningScaler(eps=1e-8, thresholding='drop_minute')
        actual = model.fit_transform(X)
        expected = np.array([[-1.15206, -0.07803],
                             [0.64361, -0.95870],
                             [0.50846, 1.03673]])
        # assert_array_almost_equal(actual, expected, decimal=4)
        assert_array_almost_equal(np.cov(actual.T), np.diag([1, 1]), decimal=4)
        assert_array_almost_equal(X, model.inverse_transform(actual))

    def test_fit_normalize_biased(self):
        X = self.identity_matrix

        model = WhiteningScaler(eps=1e-8, thresholding='normalize', unbiased=False)
        actual = model.fit_transform(X)
        expected = np.array([[-1.4142e+00, -1.4345e-17, 1.8453e-08],
                             [7.0711e-01, -1.2247e+00, -3.8987e-09],
                             [7.0711e-01, 1.2247e+00, 6.7568e-09]])
        # assert_array_almost_equal(actual, expected, decimal=4)
        assert_array_almost_equal(np.cov(actual.T), np.diag([1.5, 1.5, 0]), decimal=4)
        assert_array_almost_equal(np.cov(actual.T) * (2 / 3), np.diag([1, 1, 0]), decimal=4)
        assert_array_almost_equal(X, model.inverse_transform(actual))

    def test_fit_normalize_drop_minute_zca(self):
        X = self.identity_matrix

        model = WhiteningScaler(eps=1e-8, thresholding='drop_minute', apply_zca=True)
        actual = model.fit_transform(X)
        expected = np.array([[0.94281, -0.4714, -0.4714],
                             [-0.47140, 0.94281, -0.47140],
                             [-0.47140, -0.47140, 0.94281]])
        assert_array_almost_equal(actual, expected, decimal=4)
        assert_array_almost_equal(np.cov(actual.T),
                                  np.array([[0.6667, -0.3333, -0.3333],
                                            [-0.3333, 0.6667, -0.3333],
                                            [-0.3333, -0.3333, 0.6667]]),
                                  decimal=4)
        assert_array_almost_equal(X, model.inverse_transform(actual))


if __name__ == '__main__':
    unittest.main()
