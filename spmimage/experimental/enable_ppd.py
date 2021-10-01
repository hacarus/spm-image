"""Enables LassoPPD
The API and results of this estimator might change without any deprecation cycle.
Importing this file dynamically sets :class:`spmimage.linear_model.LassoPPD`
as an attribute of the impute module::
    >>> # explicitly require this experimental feature
    >>> from spmimage.experimental import enable_ppd import   # noqa
    >>> # now you can import normally from impute
    >>> from spmimage.linear_model import LassoPPD
"""

from ..linear_model._ppd import LassoPPD
from .. import linear_model

linear_model.LassoPPD = LassoPPD
linear_model.__all__ += ['LassoPPD']
