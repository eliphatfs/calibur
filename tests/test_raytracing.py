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
from calibur.shading import SampleEnvironments
from calibur.conventions import CC, WorldConventions
from calibur.render_pipelines import SimpleRayTraceRP


class TestRaytracing(unittest.TestCase):

    def setUp(self) -> None:
        os.makedirs("test_outputs", exist_ok=True)
        self.blender_start_cam_pose = trimesh.transformations.compose_matrix(
            angles=[1.1093189716339111, 0.0, 0.8149281740188599],
            translate=[7.358891487121582, -6.925790786743164, 4.958309173583984]
        ).astype(numpy.float32)
        self.blender_cube = resx.get_blender_cube()
        
        self.pixel_per_mm = 1920 / 36
        f = 50
        cx = 18
        aspect = 16 / 9
        cy = 18 / aspect
        sx = 36
        sy = 36 / aspect
        self.blender_start_intrinsics = f, cx, cy, sx, sy

    def render_with_blender_init_cam(self, mesh: trimesh.Trimesh, env=SampleEnvironments.eucalyptus_grove):
        cam_pose = self.blender_start_cam_pose
        cam_pose_cv = conv.convert_pose(cam_pose, CC.Blender, CC.CV)
        rp = SimpleRayTraceRP().set_geometry(mesh.triangles, mesh.face_normals)
        f, cx, cy, sx, sy = self.blender_start_intrinsics
        ss = self.pixel_per_mm
        shaded = rp.render(env, cam_pose_cv, f, f, cx, cy, sx, sy, ss)
        return cv2.cvtColor(shaded * 255, cv2.COLOR_RGB2BGR)

    def test_blender_cube_render(self):
        """
        Renders what blender starts with.
        But using a HDRI environment light.
        """
        shaded = self.render_with_blender_init_cam(self.blender_cube)
        cv2.imwrite("test_outputs/blender_init_eg.png", shaded)

    def test_spot_quad_render(self):
        mesh_pose = conv.convert_pose(numpy.eye(4, dtype=numpy.float32), WorldConventions.Blender, WorldConventions.Unity)
        # the mesh is in Unity coordinates, and we compute a pose such that in Blender it is identity
        # thus we need to convert Blender pose I to Unity pose
        mesh = resx.get_spot().copy().apply_transform(mesh_pose)
        shaded = self.render_with_blender_init_cam(mesh)
        cv2.imwrite("test_outputs/spot.png", shaded)

    def test_monkey_render(self):
        mesh_pose = conv.convert_pose(numpy.eye(4, dtype=numpy.float32), WorldConventions.Blender, WorldConventions.GL)
        mesh = resx.get_monkey().copy().apply_transform(mesh_pose)
        shaded = self.render_with_blender_init_cam(mesh, SampleEnvironments.grace_cathedral)
        cv2.imwrite("test_outputs/monkey.png", shaded)

    def test_monkey_glb_render(self):
        mesh_pose = conv.convert_pose(numpy.eye(4, dtype=numpy.float32), WorldConventions.Blender, WorldConventions.GLTF)
        mesh = resx.get_monkey_glb().copy().apply_transform(mesh_pose)
        shaded = self.render_with_blender_init_cam(mesh)
        cv2.imwrite("test_outputs/monkey_glb.png", shaded)

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
        h, w = round(sy * self.pixel_per_mm), round(sx * self.pixel_per_mm)
        proj = projection.projection_gl_persp(sx, sy, cx, cy, f, f, 0.1, 100.0)
        tris_ndc = GraphicsNDArray(transform_point(mesh.triangles, proj))
        tris_vp = viewport.gl_ndc_to_dx_viewport(tris_ndc, w, h, 0.1, 100.0)
        rays_o, rays_d = rays.get_dx_viewport_rays(h, w, 0.1)
        bvh = rtx.BVH(numpy.array(tris_vp, dtype=numpy.float32))
        hit_id, hit_d, hit_u, hit_v = bvh.raycast(rays_o, rays_d)
        face_normals = numpy.where(hit_id[..., None] >= 0, mesh.face_normals[hit_id], 0).astype(numpy.float32)
        colors = face_normals * numpy.array([0.5, 0.5, 0.5], dtype=numpy.float32) + 0.5
        cv2.imwrite(
            "test_outputs/blender_init_nor.png",
            cv2.cvtColor(numpy.clip(colors.reshape(h, w, 3), 0, 1) * 255, cv2.COLOR_RGB2BGR)
        )


if __name__ == '__main__':
    unittest.main()
