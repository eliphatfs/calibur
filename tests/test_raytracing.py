import cv2
import unittest
import numpy
import trimesh
import calibur.raytracing as rtx
import calibur.resources as resx
import calibur.rays as rays
import calibur.conventions as conv
from calibur.shading import SampleEnvironments
from calibur.conventions import CC


class TestRaytracing(unittest.TestCase):

    def test_blender_cube_render(self):
        """
        Renders what blender starts with.
        But using a HDRI environment light.
        """
        cam_pose = trimesh.transformations.compose_matrix(
            angles=[1.1093189716339111, 0.0, 0.8149281740188599],
            translate=[7.358891487121582, -6.925790786743164, 4.958309173583984]
        ).astype(numpy.float32)
        cam_pose_cv = conv.convert_pose(cam_pose, CC.Blender, CC.CV)
        mesh = resx.get_blender_cube()
        bvh = rtx.BVH(numpy.array(mesh.triangles, dtype=numpy.float32))
        pixel_per_mm = round(300 / 25.4)
        f = 50 * pixel_per_mm
        cx = 18 * pixel_per_mm
        cy = 12 * pixel_per_mm
        sx = 36 * pixel_per_mm
        sy = 24 * pixel_per_mm
        rays_o, rays_d = rays.get_cam_rays_cv(cam_pose_cv, f, f, cx, cy, sy, sx)
        hit_id, hit_d, hit_u, hit_v = bvh.raycast(rays_o, rays_d)
        face_normals = numpy.where(hit_id[..., None] >= 0, mesh.face_normals[hit_id], 0).astype(numpy.float32)
        shaded = SampleEnvironments.eucalyptus_grove.shade(face_normals)
        cv2.imshow("shaded", cv2.cvtColor(shaded.reshape(sy, sx, 3), cv2.COLOR_RGB2BGR))
        cv2.waitKey()


if __name__ == '__main__':
    unittest.main()
