import unittest
import calibur
import trimesh
import numpy
from calibur.conventions import CC


class Projection(unittest.TestCase):
    def test_fov_focal_invariance(self):
        fov = calibur.projection.focal_to_fov(50, 35)
        focal = calibur.projection.fov_to_focal(fov, 35)
        self.assertAlmostEqual(focal, 50.0)

    def test_focal_xy_invariance(self):
        fov = numpy.radians(47.1)
        self.assertAlmostEqual(calibur.projection.fov_y_to_x(fov, 1.0), fov)

    def test_space_invariance(self):
        pts = numpy.random.randn(128, 3)
        cam_pose = trimesh.transformations.compose_matrix(angles=[0.124, 1.32, 0.94], translate=[1, 2, 4])
        r = 256
        cx, cy = 100, 200
        f = 300
        
        view_mtx = numpy.linalg.inv(cam_pose)
        pts_v = calibur.graphic_utils.transform_point(pts, view_mtx)
        p_mtx = calibur.projection.projection_gl_persp(r, r, cx, cy, f, f, 0.1, 100.0)
        pts_ndc = calibur.graphic_utils.transform_point(pts_v, p_mtx)
        vp_1 = calibur.viewport.gl_ndc_to_dx_viewport(pts_ndc, r, r)
        
        cam_pose_cv = calibur.conventions.convert_pose(cam_pose, CC.GL, CC.CV)
        extrinsics_cv = numpy.linalg.inv(cam_pose_cv)
        intrinsics_cv = calibur.projection.intrinsic_cv(cx, cy, f, f)
        vp_2 = calibur.graphic_utils.transform_point(pts, extrinsics_cv) @ intrinsics_cv.T

        vp_1 = calibur.GraphicsNDArray(vp_1)
        vp_2 = calibur.GraphicsNDArray(vp_2)
        self.assertTrue(numpy.allclose(vp_1.xy, vp_2.xy / vp_2.z))
