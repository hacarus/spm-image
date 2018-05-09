import unittest

import numpy as np
from spmimage.linear_model import LassoADMM, FusedLassoADMM
from numpy.testing import assert_array_almost_equal


def build_dataset(n_samples=50, n_features=200, n_informative_features=10,
                  n_targets=1):
    """
    build an ill-posed linear regression problem with many noisy features and
    comparatively few samples
    this is the same dataset builder as in sklearn implementation
    (see https://github.com/scikit-learn/scikit-learn/blob/master
                    /sklearn/linear_model/tests/test_coordinate_descent.py)
    """
    random_state = np.random.RandomState(0)
    if n_targets > 1:
        w = random_state.randn(n_features, n_targets)
    else:
        w = random_state.randn(n_features)
    w[n_informative_features:] = 0.0
    X = random_state.randn(n_samples, n_features)
    y = np.dot(X, w)
    X_test = random_state.randn(n_samples, n_features)
    y_test = np.dot(X_test, w)
    return X, y, X_test, y_test


class TestLassoADMM(unittest.TestCase):
    def test_lasso_admm_zero(self):
        # Check that lasso by admm can handle zero data without crashing
        X = [[0], [0], [0]]
        y = [0, 0, 0]
        clf = LassoADMM(alpha=0.1).fit(X, y)
        pred = clf.predict([[1], [2], [3]])
        assert_array_almost_equal(clf.coef_, [0])
        assert_array_almost_equal(pred, [0, 0, 0])

    def test_lasso_admm_toy(self):
        # Test LassoADMM for various parameters of alpha and rho, using
        # the same test case as Lasso implementation of sklearn.
        # (see https://github.com/scikit-learn/scikit-learn/blob/master
        #               /sklearn/linear_model/tests/test_coordinate_descent.py)
        # Actually, the parameters alpha = 0 should not be allowed. However,
        # we test it as a border case.
        # WARNING:
        #   LassoADMM can't check the case which is not converged
        #   because LassoADMM doesn't check dual gap yet.
        #   This problem will be fixed in future.

        X = np.array([[-1.], [0.], [1.]])
        Y = [-1, 0, 1]  # just a straight line
        T = [[2.], [3.], [4.]]  # test sample

        clf = LassoADMM(alpha=1e-8)
        clf.fit(X, Y)
        pred = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [1], decimal=3)
        assert_array_almost_equal(pred, [2, 3, 4], decimal=3)

        clf = LassoADMM(alpha=0.1)
        clf.fit(X, Y)
        pred = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [.85], decimal=3)
        assert_array_almost_equal(pred, [1.7, 2.55, 3.4], decimal=3)

        clf = LassoADMM(alpha=0.5)
        clf.fit(X, Y)
        pred = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [.254], decimal=3)
        assert_array_almost_equal(pred, [0.508, 0.762, 1.016], decimal=3)

        clf = LassoADMM(alpha=1)
        clf.fit(X, Y)
        pred = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [.0], decimal=3)
        assert_array_almost_equal(pred, [0, 0, 0], decimal=3)

        # this is the same test case as the case alpha=1e-8
        # because the default rho parameter equals 1.0
        clf = LassoADMM(alpha=1e-8, rho=1.0)
        clf.fit(X, Y)
        pred = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [1], decimal=3)
        assert_array_almost_equal(pred, [2, 3, 4], decimal=3)

        clf = LassoADMM(alpha=0.5, rho=0.3, max_iter=50)
        clf.fit(X, Y)
        pred = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [0.249], decimal=3)
        assert_array_almost_equal(pred, [0.498, 0.746, 0.995], decimal=3)

        # tests for max_iter parameter(default = 1000)
        clf = LassoADMM(alpha=0.5, rho=0.3, max_iter=100)
        clf.fit(X, Y)
        pred = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [0.249], decimal=3)
        assert_array_almost_equal(pred, [0.498, 0.746, 0.995], decimal=3)

        clf = LassoADMM(alpha=0.5, rho=0.3, max_iter=500)
        clf.fit(X, Y)
        pred = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [0.249], decimal=3)
        assert_array_almost_equal(pred, [0.498, 0.746, 0.995], decimal=3)

        clf = LassoADMM(alpha=0.5, rho=0.3, max_iter=1000)
        clf.fit(X, Y)
        pred = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [0.249], decimal=3)
        assert_array_almost_equal(pred, [0.498, 0.746, 0.995], decimal=3)

        clf = LassoADMM(alpha=0.5, rho=0.5)
        clf.fit(X, Y)
        pred = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [0.249], decimal=3)
        assert_array_almost_equal(pred, [0.498, 0.746, 0.995], decimal=3)

    def test_lasso_admm(self):
        X, y, X_test, y_test = build_dataset()

        clf = LassoADMM(alpha=0.05, tol=1e-8).fit(X, y)
        self.assertGreater(clf.score(X_test, y_test), 0.99)
        self.assertLess(clf.n_iter_, 150)

        clf = LassoADMM(alpha=0.05, fit_intercept=False).fit(X, y)
        self.assertGreater(clf.score(X_test, y_test), 0.99)

        # normalize doesn't seem to work well
        clf = LassoADMM(alpha=0.144, rho=0.1, normalize=True).fit(X, y)
        self.assertGreater(clf.score(X_test, y_test), 0.60)


class TestFusedLassoADMM(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_fused_lasso_alpha(self):
        X = np.random.normal(0.0, 1.0, (8, 4))
        beta = np.array([4, 4, 0, 0])
        y = X.dot(beta)
        T = np.array([[5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]])  # test sample

        # small regularization parameter
        clf = FusedLassoADMM(alpha=1e-8).fit(X, y)
        actual = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [3.997, 3.998, -0.037, 0.005], decimal=3)
        assert_array_almost_equal(actual, [43.774, 75.626, 107.478], decimal=3)
        self.assertLess(clf.n_iter_, 150)

        # default
        clf = FusedLassoADMM(alpha=1).fit(X, y)
        actual = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [3.197, 1.599, 0.799, 0.4], decimal=3)
        assert_array_almost_equal(actual, [34.691, 58.673, 82.654], decimal=3)
        self.assertLess(clf.n_iter_, 150)

        # all coefs will be zero
        clf = FusedLassoADMM(alpha=10).fit(X, y)
        actual = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [0, 0, 0, 0], decimal=3)
        assert_array_almost_equal(actual, [3.72582929, 3.72582929, 3.72582929], decimal=3)
        self.assertLess(clf.n_iter_, 20)

    def test_fused_lasso_coef(self):
        X = np.random.normal(0.0, 1.0, (8, 4))
        beta = np.array([4, 4, 0, 0])
        y = X.dot(beta)
        T = np.array([[5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]])  # test sample

        # small fused_coef
        clf = FusedLassoADMM(alpha=1e-8, fused_coef=1e-4).fit(X, y)
        actual = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [3.999e+00, 3.999e+00, -7.296e-03, 9.860e-04], decimal=3)
        assert_array_almost_equal(actual, [43.95, 75.916, 107.883], decimal=3)
        self.assertLess(clf.n_iter_, 100)

        # large fused_coef
        clf = FusedLassoADMM(alpha=1e-8, fused_coef=10).fit(X, y)
        actual = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [3.916, 3.669, -0.107, 0.196], decimal=3)
        assert_array_almost_equal(actual, [42.499, 73.198, 103.896], decimal=3)
        self.assertLess(clf.n_iter_, clf.max_iter)

        # small sparse_coef
        clf = FusedLassoADMM(alpha=1e-8, sparse_coef=1e-4).fit(X, y)
        actual = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [3.999e+00, 3.999e+00, -1.100e-02, 2.126e-03], decimal=3)
        assert_array_almost_equal(actual, [43.931, 75.885, 107.839], decimal=3)
        self.assertLess(clf.n_iter_, 100)

        # large fused_coef
        clf = FusedLassoADMM(alpha=1e-8, sparse_coef=10).fit(X, y)
        actual = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [3.912, 3.826, -0.336, 0.124], decimal=3)
        assert_array_almost_equal(actual, [41.367, 71.471, 101.575], decimal=3)
        self.assertLess(clf.n_iter_, clf.max_iter)

    def test_simple_lasso(self):
        X, y, X_test, y_test = build_dataset()

        # check if FusedLasso generates same result of LassoAdmm when fused_coef is zero
        clf = FusedLassoADMM(alpha=0.05, sparse_coef=1, fused_coef=0, tol=1e-8).fit(X, y)
        self.assertGreater(clf.score(X_test, y_test), 0.99)
        self.assertLess(clf.n_iter_, 150)


if __name__ == '__main__':
    unittest.main()
