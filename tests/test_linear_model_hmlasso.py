import unittest

import numpy as np
from spmimage.linear_model import HMLasso
from numpy.testing import assert_array_almost_equal


def build_dataset(n_samples=50, n_features=100, n_informative_features=10,
                  n_targets=1):
    """
    build an ill-posed linear regression problem with many noisy features and
    comparatively few samples
    """
    random_state = np.random.RandomState(0)
    if n_targets > 1:
        w = random_state.randn(n_features, n_targets)
    else:
        w = random_state.randn(n_features)
    w[n_informative_features:] = 0.0
    X = random_state.randn(n_samples, n_features)
    y = np.dot(X, w)
    rand = random_state.rand(n_samples, n_features)
    X[rand > 0.99] = np.nan
    X_test = random_state.randn(n_samples, n_features)
    y_test = np.dot(X_test, w)
    return X, y, X_test, y_test


class TestHMLasso(unittest.TestCase):
    def test_lasso_admm_zero(self):
        # Check that lasso by admm can handle zero data without crashing
        X = [[0], [0], [0]]
        y = [0, 0, 0]
        clf = HMLasso(alpha=0.1).fit(X, y)
        pred = clf.predict([[1], [2], [3]])
        assert_array_almost_equal(clf.coef_, [0])
        assert_array_almost_equal(pred, [0, 0, 0])

    def test_lasso_admm_toy(self):
        # Test HMLasso for various parameters of alpha and mu_coef, using
        # the same test case as Lasso implementation of sklearn.
        # (see https://github.com/scikit-learn/scikit-learn/blob/master
        #               /sklearn/linear_model/tests/test_coordinate_descent.py)
        # Actually, the parameters alpha = 0 should not be allowed. However,
        # we test it as a border case.
        # WARNING:
        #   HMLasso can't check the case which is not converged
        #   because HMLasso doesn't check dual gap yet.
        #   This problem will be fixed in future.

        X = np.array([[-1.], [0.], [1.]])
        Y = [-1, 0, 1]  # just a straight line
        T = [[2.], [3.], [4.]]  # test sample

        clf = HMLasso(alpha=1e-8, tol_coef=1e-8)
        clf.fit(X, Y)
        pred = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [1], decimal=3)
        assert_array_almost_equal(pred, [2, 3, 4], decimal=3)

        clf = HMLasso(alpha=0.1, tol_coef=1e-8)
        clf.fit(X, Y)
        pred = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [.85], decimal=3)
        assert_array_almost_equal(pred, [1.7, 2.55, 3.4], decimal=3)

        clf = HMLasso(alpha=0.5, tol_coef=1e-8)
        clf.fit(X, Y)
        pred = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [.25], decimal=3)
        assert_array_almost_equal(pred, [0.5, 0.75, 1.0], decimal=3)

        clf = HMLasso(alpha=1, tol_coef=1e-8)
        clf.fit(X, Y)
        pred = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [.0], decimal=3)
        assert_array_almost_equal(pred, [0, 0, 0], decimal=3)

        # this is the same test case as the case alpha=1e-8
        # because the default mu_coef parameter equals 1.0
        clf = HMLasso(alpha=1e-8, mu_coef=1.0, tol_coef=1e-8)
        clf.fit(X, Y)
        pred = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [1], decimal=3)
        assert_array_almost_equal(pred, [2, 3, 4], decimal=3)

        clf = HMLasso(alpha=0.5, mu_coef=0.5, tol_coef=1e-8)
        clf.fit(X, Y)
        pred = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [0.25], decimal=3)
        assert_array_almost_equal(pred, [0.5, 0.75, 1.0], decimal=3)

    def test_lasso_admm_toy_multi(self):
        # for issue #39
        X = np.eye(4)
        y = np.array([[1, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0],
                      [0, 0, 1]])

        clf = HMLasso(alpha=0.05, tol_coef=1e-8).fit(X, y)
        assert_array_almost_equal(clf.coef_[0], [0.29999988, 0.29999988, -0.29999988, -0.29999988], decimal=3)
        assert_array_almost_equal(clf.coef_[1], [0.29999988, -0.29999988, 0.29999988, -0.29999988], decimal=3)
        assert_array_almost_equal(clf.coef_[2], [-0.29999988, 0.29999988, -0.29999988, 0.29999988], decimal=3)

    def test_lasso_admm(self):
        X, y, X_test, y_test = build_dataset()

        clf = HMLasso(alpha=0.05, mu_cov=0.1, tol_coef=1e-8).fit(X, y)
        self.assertGreater(clf.score(X_test, y_test), 0.9)

        clf = HMLasso(alpha=0.05, mu_cov=0.1).fit(X, y)
        self.assertGreater(clf.score(X_test, y_test), 0.9)

        clf = HMLasso(alpha=0.144, mu_cov=0.1, normalize=True).fit(X, y)
        self.assertGreater(clf.score(X_test, y_test), 0.9)

    def test_lasso_admm_multi(self):
        X, y, X_test, y_test = build_dataset(n_targets=3)

        clf = HMLasso(alpha=0.05, mu_cov=0.1, tol_coef=1e-8).fit(X, y)
        self.assertGreater(clf.score(X_test, y_test), 0.9)

        clf = HMLasso(alpha=0.05, mu_cov=0.1, ).fit(X, y)
        self.assertGreater(clf.score(X_test, y_test), 0.9)

        clf = HMLasso(alpha=0.144, mu_cov=0.1, normalize=True).fit(X, y)
        self.assertGreater(clf.score(X_test, y_test), 0.9)


if __name__ == '__main__':
    unittest.main()
