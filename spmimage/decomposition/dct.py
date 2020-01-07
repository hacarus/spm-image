import numpy as np


def generate_dct_dictionary(patch_size: int, sqrt_dict_size: int) -> np.ndarray:
    """generate_dct_dictionary
    Generate a DCT dictionary.
    An atom is a (patch_size, patch_size) image, and total number of atoms is 
    sqrt_dict_size * sqrt_dict_size.
    The result D is a matrix whose shape is (sqrt_dict_size^2, patch_size^2).
    Note that, a row of the result D shows an atom (flatten).

    Parameters:
    ------------
        patch_size : int
            height and width of an atom of DCT dictionary

        sqrt_dict_size : int
            Total number of DCT atoms is a square number.
            This parameter fix the number of atoms in the Dictionary.

    Returns:
    ------------
        D : np.ndarray, shape (sqrt_dict_size^2, patch_size^2)
            DCT dictionary
    """
    D1 = np.zeros((sqrt_dict_size, patch_size))
    for k in np.arange(sqrt_dict_size):
        for i in np.arange(patch_size):
            D1[k, i] = np.cos(i * k * np.pi / float(sqrt_dict_size))
        if k != 0:
            D1[k, :] -= D1[k, :].mean()
    return np.kron(D1, D1)
