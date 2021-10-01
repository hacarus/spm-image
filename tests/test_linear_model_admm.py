import unittest

import numpy as np
from spmimage.linear_model import (LassoADMM,
                                   FusedLassoADMM,
                                   TrendFilteringADMM,
                                   QuadraticTrendFilteringADMM)
from spmimage.linear_model.admm import admm_path
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

        clf = LassoADMM(alpha=0.5, rho=0.5)
        clf.fit(X, Y)
        pred = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [0.249], decimal=3)
        assert_array_almost_equal(pred, [0.498, 0.746, 0.995], decimal=3)

    def test_lasso_admm_toy_multi(self):
        # for issue #39
        X = np.eye(4)
        y = np.array([[1, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0],
                      [0, 0, 1]])

        clf = LassoADMM(alpha=0.05, tol=1e-8).fit(X, y)
        assert_array_almost_equal(clf.coef_[0], [0.3, 0.3, -0.3, -0.3],
                                  decimal=3)
        assert_array_almost_equal(clf.coef_[1], [0.3, -0.3, 0.3, -0.3],
                                  decimal=3)
        assert_array_almost_equal(clf.coef_[2], [-0.3, 0.3, -0.3, 0.3],
                                  decimal=3)

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

    def test_lasso_admm_multi(self):
        X, y, X_test, y_test = build_dataset(n_targets=3)

        clf = LassoADMM(alpha=0.05, tol=1e-8).fit(X, y)
        self.assertGreater(clf.score(X_test, y_test), 0.99)
        self.assertLess(clf.n_iter_[0], 150)

        clf = LassoADMM(alpha=0.05, fit_intercept=False).fit(X, y)
        self.assertGreater(clf.score(X_test, y_test), 0.99)

        # normalize doesn't seem to work well
        clf = LassoADMM(alpha=0.144, rho=0.1, normalize=True).fit(X, y)
        self.assertGreater(clf.score(X_test, y_test), 0.60)


class TestFusedLassoADMM(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X = np.random.normal(0.0, 1.0, (8, 4))

    def test_fused_lasso_alpha(self):
        beta = np.array([4, 4, 0, 0])
        y = self.X.dot(beta)
        T = np.array([[5., 6., 7., 8.],
                      [9., 10., 11., 12.],
                      [13., 14., 15., 16.]])  # test sample

        # small regularization parameter
        clf = FusedLassoADMM(alpha=1e-8).fit(self.X, y)
        actual = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [3.998, 3.998, -0.022, 0.003],
                                  decimal=3)
        assert_array_almost_equal(actual, [43.862, 75.773, 107.683], decimal=3)
        self.assertLess(clf.n_iter_, 100)

        # default
        clf = FusedLassoADMM(alpha=1).fit(self.X, y)
        actual = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [2.428, 1.506, 0., 0.], decimal=3)
        assert_array_almost_equal(actual, [22.69, 38.427, 54.164], decimal=3)
        self.assertLess(clf.n_iter_, 100)

        # all coefs will be zero
        clf = FusedLassoADMM(alpha=10).fit(self.X, y)
        actual = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [0, 0, 0, 0], decimal=3)
        assert_array_almost_equal(actual, [3.724, 3.722, 3.72], decimal=3)

        self.assertLess(clf.n_iter_, 20)

    def test_fused_lasso_coef(self):
        beta = np.array([4, 4, 0, 0])
        y = self.X.dot(beta)
        T = np.array([[5., 6., 7., 8.],
                      [9., 10., 11., 12.],
                      [13., 14., 15., 16.]])  # test sample

        # small trend_coef
        clf = FusedLassoADMM(alpha=1e-8, trend_coef=1e-4).fit(self.X, y)
        actual = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [3.999, 3.999, -0.007, 0.001],
                                  decimal=3)
        assert_array_almost_equal(actual, [43.95, 75.916, 107.883], decimal=3)
        self.assertLess(clf.n_iter_, 100)

        # large trend_coef
        clf = FusedLassoADMM(alpha=1e-8, trend_coef=10).fit(self.X, y)
        actual = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [3.938, 3.755, -0.079, 0.141],
                                  decimal=3)
        assert_array_almost_equal(actual, [42.862, 73.885, 104.908], decimal=3)
        self.assertLess(clf.n_iter_, clf.max_iter)

        # small sparse_coef
        clf = FusedLassoADMM(alpha=1e-8, sparse_coef=1e-4).fit(self.X, y)
        actual = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [3.999, 3.999, -0.011, 0.002],
                                  decimal=3)
        assert_array_almost_equal(actual, [43.931, 75.885, 107.839], decimal=3)
        self.assertLess(clf.n_iter_, 100)

        # large sparse_coef
        clf = FusedLassoADMM(alpha=1e-8, sparse_coef=10).fit(self.X, y)
        actual = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [3.913, 3.837, -0.349, 0.113],
                                  decimal=3)
        assert_array_almost_equal(actual, [41.265, 71.32, 101.374], decimal=3)
        self.assertLess(clf.n_iter_, clf.max_iter)

    def test_simple_lasso(self):
        X, y, X_test, y_test = build_dataset()

        # check if FusedLasso generates the same result of LassoAdmm
        # when trend_coef is zero
        clf = FusedLassoADMM(alpha=0.05, sparse_coef=1,
                             trend_coef=0, tol=1e-8).fit(X, y)
        self.assertGreater(clf.score(X_test, y_test), 0.99)
        self.assertLess(clf.n_iter_, 150)


class TestTrendFilteringADMM(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X = np.random.normal(0.0, 1.0, (8, 5))

    def test_generate_transform_matrix(self):
        D = np.array([[-1, 2, -1, 0, 0],
                      [0, -1, 2, -1, 0],
                      [0, 0, -1, 2, -1]])
        clf = TrendFilteringADMM(sparse_coef=1, trend_coef=0)
        assert_array_almost_equal(np.eye(5), clf.generate_transform_matrix(5))

        clf = TrendFilteringADMM(sparse_coef=0, trend_coef=1)
        assert_array_almost_equal(D, clf.generate_transform_matrix(5))

        clf = TrendFilteringADMM(sparse_coef=1, trend_coef=1)
        assert_array_almost_equal(np.vstack([np.eye(5), D]),
                                  clf.generate_transform_matrix(5))

    def test_trend_filtering(self):
        beta = np.array([0., 10., 20., 10., 0.])
        y = self.X.dot(beta)

        # small regularization parameter
        clf = TrendFilteringADMM(alpha=1e-8).fit(self.X, y)
        assert_array_almost_equal(np.round(clf.coef_), [0, 10, 20, 10, 0])

        # default
        clf = TrendFilteringADMM(alpha=0.01).fit(self.X, y)
        assert_array_almost_equal(clf.coef_, [0.015, 10.017, 19.932,
                                              9.989, 0.033], decimal=3)

        # all coefs will be zero
        clf = TrendFilteringADMM(alpha=1e5).fit(self.X, y)
        assert_array_almost_equal(clf.coef_, [0., 0., 0., 0., 0.], decimal=1)


class TestQuadraticTrendFilteringADMM(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X = np.random.normal(0.0, 1.0, (8, 5))

    def test_generate_transform_matrix(self):
        D = np.array([[1., -3., 3., -1., 0.],
                      [0., 1., -3., 3., -1.]])

        clf = QuadraticTrendFilteringADMM(sparse_coef=1, trend_coef=0)
        assert_array_almost_equal(np.eye(5), clf.generate_transform_matrix(5))

        clf = QuadraticTrendFilteringADMM(sparse_coef=0, trend_coef=1)
        assert_array_almost_equal(D, clf.generate_transform_matrix(5))

        clf = QuadraticTrendFilteringADMM(sparse_coef=1, trend_coef=1)
        assert_array_almost_equal(np.vstack([np.eye(5), D]),
                                  clf.generate_transform_matrix(5))

        # boundary check
        assert_array_almost_equal(np.eye(1), clf.generate_transform_matrix(1))
        assert_array_almost_equal(np.eye(2), clf.generate_transform_matrix(2))
        assert_array_almost_equal(np.eye(3), clf.generate_transform_matrix(3))

    def test_trend_filtering(self):
        beta = np.array([0., 1., 2., 1., 0.])
        y = self.X.dot(beta)

        # small regularization parameter
        clf = QuadraticTrendFilteringADMM(alpha=1e-8).fit(self.X, y)
        assert_array_almost_equal(np.round(clf.coef_), [0, 1, 2, 1, 0])

        # all coefs will be zero
        clf = QuadraticTrendFilteringADMM(alpha=1e5).fit(self.X, y)
        assert_array_almost_equal(clf.coef_, [0., 0., 0., 0., 0.], decimal=1)


class TestAdmmPath(unittest.TestCase):
    def test_admm_path_alphas(self):
        # check if input alphas are sorted
        X, y, X_test, y_test = build_dataset()

        alphas = [0.1, 0.3, 0.5, -0.1, -0.2]
        actual_alphas, _, _ = admm_path(X, y, alphas=alphas)
        assert_array_almost_equal(actual_alphas, [0.5, 0.3, 0.1, -0.1, -0.2])

    def test_admm_path_coefs(self):
        # check if we can get correct coefs

        X = np.array([[-1.], [0.], [1.]])
        y = np.array([-1, 0, 1])  # just a straight line

        _, coefs_actual, _ = admm_path(X, y,
                                       alphas=[1e-8, 0.1, 0.5, 1], rho=1.0)
        assert_array_almost_equal(coefs_actual[0], [-1.31072000e-04,
                                                    2.53888000e-01,
                                                    8.49673483e-01,
                                                    9.99738771e-01], decimal=3)


if __name__ == '__main__':
    unittest.main()
