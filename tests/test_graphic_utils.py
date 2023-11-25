import unittest
import calibur
import numpy
import trimesh


class TestGraphicUtilities(unittest.TestCase):

    def test_bilinear(self):
        im = numpy.array([0, 1, 0, 0], dtype=numpy.float32).reshape(2, 2, 1)
        self.assertListEqual(
            calibur.sample2d(im, [[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(-1).tolist(),
            [0, 0, 0, 1]
        )
        self.assertListEqual(
            calibur.sample2d(im, [
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
            calibur.point_in_tri2d(pts, v0, v1, v2).reshape(-1).tolist(),
            [True, False, False, True]
        )

    def test_compute_normals(self):
        spot_tris = calibur.resources.get_spot()
        mesh = trimesh.Trimesh(spot_tris.reshape(-1, 3), numpy.arange(len(spot_tris) * 3).reshape(-1, 3))
        nors = calibur.compute_tri3d_normals(spot_tris)
        self.assertTrue(numpy.allclose(mesh.face_normals, nors))

    def test_compute_normals_transformed(self):
        spot_tris = calibur.resources.get_spot()
        mesh = trimesh.Trimesh(spot_tris.reshape(-1, 3), numpy.arange(len(spot_tris) * 3).reshape(-1, 3))
        pose = numpy.diag([1, -1, -1, 1])
        tris_trs = calibur.transform_point(spot_tris, pose)
        nors = calibur.compute_tri3d_normals(tris_trs)
        mesh.apply_transform(pose)
        self.assertTrue(numpy.allclose(mesh.face_normals, nors))
