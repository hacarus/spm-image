import numpy as np
from itertools import product

from typing import Tuple

__all__ = [
    'extract_simple_patches_2d',
    'reconstruct_from_simple_patches_2d',
]


def extract_simple_patches_2d(image: np.ndarray, patch_size: Tuple[int, int]) -> np.ndarray:
    """Reshape a 2D image into a collection of patches without duplication of extracted range.
    """
    i_h, i_w = image.shape
    p_h, p_w = patch_size

    patches = []

    n_h = int(i_h / p_h)
    n_w = int(i_w / p_w)

    for i in range(n_h):
        for j in range(n_w):
            patch = image[p_h * i:p_h * i + p_h, p_w * j:p_w * j + p_w]
            patches.append(patch.flatten())

    n_patch = len(patches)
    return np.asarray(patches).flatten().reshape(n_patch, p_h, p_w)


def reconstruct_from_simple_patches_2d(patches: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """Reconstruct the image from all of its patches.
    """
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    image = np.zeros(image_size)

    n_h = int(i_h / p_h)
    n_w = int(i_w / p_w)
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        image[p_h * i:p_h * i + p_h, p_w * j:p_w * j + p_w] += p
    return image
