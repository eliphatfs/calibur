import os
import cv2
import unittest
import numpy
import trimesh
import calibur.raytracing as rtx
import calibur.resources as resx
import calibur.rays as rays
import calibur.conventions as conv
import calibur.projection as projection
import calibur.viewport as viewport
from calibur.ndarray_extension import GraphicsNDArray
from calibur.graphic_utils import transform_point
from calibur.generic_utils import cat
from calibur.shading import SampleEnvironments
from calibur.conventions import CC


class TestRaytracing(unittest.TestCase):

    def setUp(self) -> None:
        os.makedirs("test_outputs", exist_ok=True)
        self.blender_start_cam_pose = trimesh.transformations.compose_matrix(
            angles=[1.1093189716339111, 0.0, 0.8149281740188599],
            translate=[7.358891487121582, -6.925790786743164, 4.958309173583984]
        ).astype(numpy.float32)
        self.blender_cube = resx.get_blender_cube()
        
        pixel_per_mm = 1920 / 36
        f = 50 * pixel_per_mm
        cx = 18 * pixel_per_mm
        aspect = 16 / 9
        cy = 18 / aspect * pixel_per_mm
        sx = round(36 * pixel_per_mm)
        sy = round(36 / aspect * pixel_per_mm)
        self.blender_start_intrinsics = f, cx, cy, sx, sy

    def test_blender_cube_render(self):
        """
        Renders what blender starts with.
        But using a HDRI environment light.
        """
        cam_pose = self.blender_start_cam_pose
        cam_pose_cv = conv.convert_pose(cam_pose, CC.Blender, CC.CV)
        mesh = self.blender_cube
        bvh = rtx.BVH(numpy.array(mesh.triangles, dtype=numpy.float32))
        f, cx, cy, sx, sy = self.blender_start_intrinsics
        rays_o, rays_d = rays.get_cam_rays_cv(cam_pose_cv, f, f, cx, cy, sy, sx)
        hit_id, hit_d, hit_u, hit_v = bvh.raycast(rays_o, rays_d)
        face_normals = numpy.where(hit_id[..., None] >= 0, mesh.face_normals[hit_id], 0).astype(numpy.float32)
        shaded = SampleEnvironments.eucalyptus_grove.shade(face_normals)
        cv2.imwrite(
            "test_outputs/blender_init_eg.png",
            cv2.cvtColor(numpy.clip(shaded.reshape(sy, sx, 3), 0, 1) * 255, cv2.COLOR_RGB2BGR)
        )

    def test_blender_cube_viewport_visibility(self):
        """
        Renders the camera-space normals by casting rays in GL NDC of what blender starts with.
        """
        cam_pose = self.blender_start_cam_pose
        cam_pose_gl = conv.convert_pose(cam_pose, CC.Blender, CC.GL)
        # view matrix is inverse of camera pose
        mesh = self.blender_cube.copy().apply_transform(numpy.linalg.inv(cam_pose_gl))
        # mesh now in view space
        f, cx, cy, sx, sy = self.blender_start_intrinsics
        proj = projection.projection_gl_persp(sx, sy, cx, cy, f, f, 0.1, 100.0)
        tris_ndc = GraphicsNDArray(transform_point(mesh.triangles, proj))
        tris_vp = viewport.gl_ndc_to_dx_viewport(tris_ndc, sx, sy, 0.1, 100.0)
        rays_o, rays_d = rays.get_dx_viewport_rays(sy, sx, 0.1)
        bvh = rtx.BVH(numpy.array(tris_vp, dtype=numpy.float32))
        hit_id, hit_d, hit_u, hit_v = bvh.raycast(rays_o, rays_d)
        face_normals = numpy.where(hit_id[..., None] >= 0, mesh.face_normals[hit_id], 0).astype(numpy.float32)
        colors = face_normals * numpy.array([0.5, 0.5, 0.5], dtype=numpy.float32) + 0.5
        cv2.imwrite(
            "test_outputs/blender_init_nor.png",
            cv2.cvtColor(numpy.clip(colors.reshape(sy, sx, 3), 0, 1) * 255, cv2.COLOR_RGB2BGR)
        )


if __name__ == '__main__':
    unittest.main()
