import numpy
from . import rays, raytracing as rtx
from .shading import SHEnvironment
from .ndarray_extension import NDArray


class SimpleRayTraceRP(object):

    def set_geometry(self, tris: NDArray, tri_nors: NDArray):
        self.tri_nors = numpy.array(tri_nors, numpy.float32)
        self.bvh = rtx.BVH(numpy.array(tris, dtype=numpy.float32))
        return self

    def render(
        self,
        env: SHEnvironment, cam_pose_cv: NDArray,
        fx: float, fy: float, cx: float, cy: float, sx: float, sy: float,
        pixel_per_unit: float = 1.0,
    ):
        ss = pixel_per_unit
        h, w = round(sy * ss), round(sx * ss)
        rays_o, rays_d = rays.get_cam_rays_cv(
            cam_pose_cv, fx * ss, fy * ss, cx * ss, cy * ss, h, w
        )
        hit_id, hit_d, hit_u, hit_v = self.bvh.raycast(rays_o, rays_d)
        face_normals = numpy.where(hit_id[..., None] >= 0, self.tri_nors[hit_id], 0)
        return numpy.clip(env.shade(face_normals).reshape(h, w, 3), 0, 1)
