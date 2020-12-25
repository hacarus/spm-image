from logging import getLogger

from abc import abstractmethod, ABC

from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel

from typing import Tuple
import numpy as np

logger = getLogger(__name__)


class PreconditionedPrimalDual(LinearModel, RegressorMixin, ABC):
    """
    Abstract class for Preconditioned Primal Dual algorithm
    """

    def __init__(self,
                 alpha: float = 1.0,
                 max_iter: int = 1000):
        """
        Lasso Preconditioned Primal Dual algorithm

        Parameters
        ----------
        alpha : float
            A regularization parametfer.
        max_iter : int
            The maximum number of iterations.
        """
        self.alpha = alpha
        self.max_iter = max_iter

    @abstractmethod
    def _Du(self, u: np.ndarray):
        """
        calc Du, where D is a matrix and u is a vector

        Parameters
        --------------
        u : np.ndarray
            a vector
        """
        raise NotImplementedError()

    @abstractmethod
    def _DTv(self, v: np.ndarray):
        """
        calc D^Tv, where D is a matrix and v is a vector

        Parameters
        --------------
        v : np.ndarray
            a vector
        """
        raise NotImplementedError()

    @abstractmethod
    def _step_size(self):
        """
        define tau and sigma
        """
        raise NotImplementedError()

    @abstractmethod
    def _init_dual(self, y, sigma):
        """
        define dual vector
        """
        raise NotImplementedError()

    def fit(self, X, y, check_input=False):
        """
        Parameters
        ----------
        X : ignored
        y : np.ndarray
            input data

        Attribute
        ---------
        self._coef
            estimated result

        Returns
        --------
        self : PreconditionedPrimalDual
            for method chain
        """
        if self.alpha == 0:
            logger.warning(
                """With alpha=0, this algorithm does not converge well. You are advised to use the LinearRegression estimator""")
            raise ValueError()

        self.shape = y.shape
        self.dim = np.prod(self.shape)
        tau, sigma = self._step_size()

        # initialize
        u_pre = np.copy(y)
        dual = self._init_dual(u_pre, sigma)

        # main loop
        for _ in range(self.max_iter):
            u = u_pre - (tau * (self._DTv(dual[:-self.dim]) + self.alpha * dual[-self.dim:]))
            u_bar = 2 * u - u_pre
            dual[:-self.dim] += sigma[:-self.dim] * self._Du(u_bar)
            dual[-self.dim:] += sigma[-self.dim:] * self.alpha * (u_bar - y)
            dual = np.clip(dual, -1, 1)
            u_pre = u

        self.coef_ = u_pre
        return self


class LassoPPD(PreconditionedPrimalDual):
    def __init__(self,
                 alpha: float = 1.0,
                 max_iter: int = 1000,
                 params: np.ndarray = None):
        """
        Lasso Preconditioned Primal Dual algorithm

        Parameters
        ----------
        alpha : float
            A regularization parametfer.
        max_iter : int
            The maximum number of iterations.
        coef : np.ndarray, default, None
            lasso coef
        """
        super().__init__(alpha, max_iter)
        if params is None:
            self.params = np.array([1])
        else:
            self.params = np.array(params)

    def _Du(self, u: np.ndarray) -> np.ndarray:
        """
        calc Du

        Parameters
        --------------
        u : np.ndarray
            a vector

        Return
        --------------
        x : np.ndarray
            x = Du
        """
        x = np.zeros(len(u) - len(self.params) + 1)
        for i, k in enumerate(self.params):
            x += k * u[i:len(u) - len(self.params) + i + 1]
        return x

    def _DTv(self, v: np.ndarray) -> np.ndarray:
        """
        calc D^Tv

        Parameters
        --------------
        v : np.ndarray
            a vector

        Return
        --------------
        x : np.ndarray
            x = Dv
        """
        x = np.zeros(len(v) + len(self.params) - 1)
        for i, k in enumerate(self.params):
            x[i:len(v) + i] += k * v
        return x

    def _step_size(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        define tau and sigma

        Parameters
        ---------------
        dim : int
            the dimension of input array

        Return
        -------
        tau : np.ndarray
            step parameter tau

        sigma : np.ndarray
            step parameter sigma
        """
        tau = np.ones(self.dim) * np.sum(np.abs(self.params))
        for i in range(len(self.params) - 1):
            tau[i] = np.sum(np.abs(self.params[:i + 1]))
            tau[-i] = np.sum(np.abs(self.params[-i - 1:]))
        tau += self.alpha
        tau = 1. / tau

        sigma = np.ones(2 * self.dim - len(self.params) + 1)
        sigma[:self.dim - len(self.params) + 1] *= np.sum(np.abs(self.params))
        sigma[self.dim - len(self.params) + 1:] *= self.alpha
        sigma = 1. / sigma
        return tau, sigma

    def _init_dual(self, y, sigma) -> np.ndarray:
        """
        define dual vector

        Return
        ------
        dual : np.ndarray
        """
        dual = np.zeros(2 * self.dim - len(self.params) + 1)
        dual[:-self.dim] = sigma[:-self.dim] * self._Du(y)
        dual = np.clip(dual, -1, 1)
        return dual
