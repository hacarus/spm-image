from unittest import TestCase

from spmimage.feature_extraction.image import extract_simple_patches_2d

import numpy as np


class TestImage(TestCase):
    def test_extract_simple_patches_2d_numeric(self):
        image = np.arange(16).reshape((4, 4))

        actual = extract_simple_patches_2d(image, (2, 2,))
        self.assertEqual((4, 2, 2), actual.shape)

        self.assertTrue((np.array([0, 1, 4, 5]) == actual[0].flatten()).all())
        self.assertTrue((np.array([2, 3, 6, 7]) == actual[1].flatten()).all())
        self.assertTrue((np.array([8, 9, 12, 13]) == actual[2].flatten()).all())
        self.assertTrue((np.array([10, 11, 14, 15]) == actual[3].flatten()).all())
