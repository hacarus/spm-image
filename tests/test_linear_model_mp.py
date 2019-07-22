import unittest

from spmimage.linear_model import matching_pursuit
import numpy as np
from sklearn.preprocessing import normalize
from numpy.testing import assert_array_almost_equal


class TestMatchingPursuit(unittest.TestCase):
    def setUp(self):
        np.random.seed()

    def test_matching_pursuit(self):
        X = normalize(np.array([[1., 2.], [3., 4.], [5., 6.]]))
        beta = np.array([[0., 1., 0.], [3., 0., 0.]])
        y = beta.dot(X)
        predict = matching_pursuit(dictionary=X, signal=y, tol=1e-6)
        assert_array_almost_equal(beta, predict)
