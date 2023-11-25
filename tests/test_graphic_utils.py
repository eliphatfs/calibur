import unittest
import calibur
import numpy


class TestGraphicUtilities(unittest.TestCase):

    def test_bilinear(self):
        im = numpy.array([0, 1, 0, 0], dtype=numpy.float32).reshape(2, 2, 1)
        self.assertListEqual(
            calibur.graphic_utils.sample2d(im, [[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(-1).tolist(),
            [0, 0, 0, 1]
        )
        self.assertListEqual(
            calibur.graphic_utils.sample2d(im, [
                [0.25, 0.5], [0.5, 0.5], [0.5, 0.75], [0.75, 0.75]
            ]).reshape(-1).tolist(),
            [0.0, 0.25, 0.5, 1]
        )

    def test_point_in_triangle_2d(self):
        v0 = [0.0, 1.0]
        v1 = [1.0, 0.0]
        v2 = [0.0, 0.0]
        pts = [[0.2, 0.5], [0.3, -0.1], [0.8, 0.5], [0.1, 0.7]]
        self.assertListEqual(
            calibur.graphic_utils.point_in_tri2d(pts, v0, v1, v2).reshape(-1).tolist(),
            [True, False, False, True]
        )
