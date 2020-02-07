import itertools
from typing import Tuple

import numpy as np


def zig_zag_index(k: int, n: int) -> Tuple[int, int]:
    """
    get k-th index i and j on (n, n)-matrix according to zig-zag scan.

    Parameters:
    -----------
        k : int
            a ranking of element, which we want to know the index i and j

        n : int
            a size of square matrix

    Returns:
    -----------
        (i, j) : Tuple[int, int]
            the tuple which represents the height and width index of k-th elements

    Reference
    ----------
    https://medium.com/100-days-of-algorithms/day-63-zig-zag-51a41127f31
    """
    # upper side of interval
    if k >= n * (n + 1) // 2:
        i, j = zig_zag_index(n * n - 1 - k, n)
        return n - 1 - i, n - 1 - j

    # lower side of interval
    i = int((np.sqrt(1 + 8 * k) - 1) / 2)
    j = k - i * (i + 1) // 2
    return (j, i - j) if i & 1 else (i - j, j)


def generate_dct_atom(u, v, n) -> np.ndarray:
    """
    generate an (u, v)-th atom of DCT dictionary with size n by n.

    Parameters:
    -----------
        u : int
            an index for height

        v : int
            an index for width

        n : int
            a size of DCT

    Returns:
    -----------
        atom : np.ndarray
            (n, n) matrix which represents (u,v)-th atom of DCT dictionary
    """
    atom = np.empty((n, n))
    for i, j in itertools.product(range(n), range(n)):
        atom[i, j] = np.cos(((i+0.5)*u*np.pi)/n) * np.cos(((j+0.5)*v*np.pi)/n)
    return atom

def generate_dct_dictionary(n_components: int, patch_size: int) -> np.ndarray:
    """generate_dct_dictionary
    Generate a DCT dictionary.
    An atom is a (patch_size, patch_size) image, and total number of atoms is
    n_components.

    The result D is a matrix whose shape is (n_components, patch_size ** 2).
    Note that, a row of the result D shows an atom (flatten).

    Parameters:
    ------------
        n_components: int
            a number of atom, where n_components <= patch_size ** 2.

        patch_size : int
            size of atom of DCT dictionary

    Returns:
    ------------
        D : np.ndarray, shape (n_components, patch_size ** 2)
            DCT dictionary
    """
    D = np.empty((n_components, patch_size ** 2))

    if n_components > patch_size ** 2:
        raise ValueError("n_components must be smaller than patch_size ** 2")

    elif n_components == patch_size ** 2:
        for i, j in itertools.product(range(patch_size), range(patch_size)):
            D[i*patch_size + j] = generate_dct_atom(i, j, patch_size).flatten()
    else:
        for k in range(n_components):
            i, j = zig_zag_index(k, patch_size)
            D[k, :] = generate_dct_atom(i, j, patch_size).flatten()
    return D
