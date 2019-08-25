import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal


class TestLassoPPD(unittest.TestCase):
    def setUp(self) -> None:
        # NOTE: we should import LassoPPD here, otherwise `test_experimental.py` will fail.
        from spmimage.experimental import enable_ppd
        from spmimage.linear_model import LassoPPD

        self.lasso = LassoPPD(alpha=1, max_iter=10, params=[1])
        self.fused = LassoPPD(alpha=1, max_iter=10, params=[1, -1])
        self.trend = LassoPPD(alpha=1, max_iter=10, params=[1, -2, 1])

    def test_Du(self):
        u = np.ones(10)
        assert_array_almost_equal(u, self.lasso._Du(u))
        assert_array_almost_equal(np.zeros(9), self.fused._Du(u))
        assert_array_almost_equal(np.zeros(8), self.trend._Du(u))

    def test_DTv(self):
        v = np.ones(10)
        v1 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1])
        v2 = np.array([1, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1])
        assert_array_almost_equal(v, self.lasso._DTv(v))
        assert_array_almost_equal(v1, self.fused._DTv(v))
        assert_array_almost_equal(v2, self.trend._DTv(v))

    def test_lasso_admm_zero(self):
        # Check that lasso by admm can handle zero data without crashing
        y = np.random.rand(100)
        self.lasso.fit(None, y)
        self.fused.fit(None, y)
        self.trend.fit(None, y)


if __name__ == '__main__':
    unittest.main()
