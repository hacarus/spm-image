from unittest import TestCase

from spmimage.feature_extraction.image import extract_simple_patches_2d, reconstruct_from_simple_patches_2d

import numpy as np


class TestImage(TestCase):
    def test_extract_simple_patches_2d_gray(self):
        image = np.arange(16).reshape((4, 4))

        actual = extract_simple_patches_2d(image, (2, 2,))
        self.assertEqual((4, 2, 2), actual.shape)

        self.assertTrue((np.array([0, 1, 4, 5]) == actual[0].flatten()).all())
        self.assertTrue((np.array([2, 3, 6, 7]) == actual[1].flatten()).all())
        self.assertTrue((np.array([8, 9, 12, 13]) == actual[2].flatten()).all())
        self.assertTrue((np.array([10, 11, 14, 15]) == actual[3].flatten()).all())

    def test_extract_simple_patches_2d_color(self):
        image = np.arange(48).reshape((4, 4, 3))

        actual = extract_simple_patches_2d(image, (2, 2,))
        self.assertEqual((4, 2, 2, 3), actual.shape)

        self.assertTrue((np.array([0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]) == actual[0].flatten()).all())
        self.assertTrue((np.array([6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23]) == actual[1].flatten()).all())
        self.assertTrue((np.array([24, 25, 26, 27, 28, 29, 36, 37, 38, 39, 40, 41]) == actual[2].flatten()).all())
        self.assertTrue((np.array([30, 31, 32, 33, 34, 35, 42, 43, 44, 45, 46, 47]) == actual[3].flatten()).all())

    def test_reconstruct_from_simple_patches_2d_gray(self):
        patches = np.stack((
            np.array([[0, 1], [4, 5]]),
            np.array([[2, 3], [6, 7]]),
            np.array([[8, 9], [12, 13]]),
            np.array([[10, 11], [14, 15]]),
        ))
        actual = reconstruct_from_simple_patches_2d(patches, (4, 4))
        self.assertTrue((np.arange(16).reshape((4, 4)) == actual).all())

    def test_reconstruct_from_simple_patches_2d_color(self):
        patches = np.stack((
            np.array([[[0, 1, 2], [3, 4, 5]], [[12, 13, 14], [15, 16, 17]]]),
            np.array([[[6, 7, 8], [9, 10, 11]], [[18, 19, 20], [21, 22, 23]]]),
            np.array([[[24, 25, 26], [27, 28, 29]], [[36, 37, 38], [39, 40, 41]]]),
            np.array([[[30, 31, 32], [33, 34, 35]], [[42, 43, 44], [45, 46, 47]]]),
        ))
        actual = reconstruct_from_simple_patches_2d(patches, (4, 4, 3))
        self.assertTrue((np.arange(48).reshape((4, 4, 3)) == actual).all())
