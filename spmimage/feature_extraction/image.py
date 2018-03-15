from logging import getLogger

import numpy as np
from itertools import product

from typing import Tuple

__all__ = [
    'extract_simple_patches_2d',
    'reconstruct_from_simple_patches_2d',
]

logger = getLogger(__name__)


def extract_simple_patches_2d(image: np.ndarray, patch_size: Tuple[int, int]) -> np.ndarray:
    """Reshape a 2D image into a collection of patches without duplication of extracted range.
    """

    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size

    if i_h % p_h != 0 or i_w % p_w != 0:
        logger.warning(
            'image %s divided by patch %s is not zero and some parts will be lost', image.shape[:2], patch_size)

    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]

    patches = []

    n_h = int(i_h / p_h)
    n_w = int(i_w / p_w)

    for i in range(n_h):
        for j in range(n_w):
            patch = image[p_h * i:p_h * i + p_h, p_w * j:p_w * j + p_w]
            patches.append(patch.flatten())

    n_patches = len(patches)

    patches_ret = np.asarray(patches).flatten().reshape(-1, p_h, p_w, n_colors)
    if patches_ret.shape[-1] == 1:
        return patches_ret.reshape((n_patches, p_h, p_w))
    else:
        return patches_ret


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
