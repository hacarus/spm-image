import unittest

import spmimage.linear_model
import importlib


class TestExperimental(unittest.TestCase):

    def test_ppd_disabled(self):
        # if you see this method fail, please check if you import LassoPPD in global in other test cases
        importlib.reload(spmimage.linear_model)
        with self.assertRaises(ImportError) as ctx:
            from spmimage.linear_model import LassoPPD
        self.assertIsNotNone(ctx.exception)

    def test_ppd_enabled(self):
        importlib.reload(spmimage.linear_model)
        from spmimage.experimental import enable_ppd
        from spmimage.linear_model import LassoPPD
        self.assertIsNotNone(LassoPPD)


if __name__ == '__main__':
    unittest.main()
