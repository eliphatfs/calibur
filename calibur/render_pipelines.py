import numpy
from . import rays, raytracing as rtx
from .shading import Environment
from .ndarray_extension import NDArray
from .graphic_utils import compute_tri3d_normals


class SimpleRayTraceRP(object):
    """
    A minimal CPU ray tracing render pipeline for debugging cameras.
    """

    def set_geometry(self, tris: NDArray):
        """
        Set geometry triangles. Assumes CCW winding.

        :param tris: ``(N, 3, 3)`` triangles, array of 3 vertices.
        :returns: *self* after preprocessing the geometry.
        """
        tris = numpy.array(tris, dtype=numpy.float32)
        self.tri_nors = compute_tri3d_normals(tris)
        self.bvh = rtx.BVH(tris)
        return self

    def render(
        self,
        env: Environment, cam_pose_cv: NDArray,
        fx: float, fy: float, cx: float, cy: float, sx: float, sy: float,
        pixel_per_unit: float = 1.0,
    ):
        """
        Renders the geometry using an environment.

        :param env: The :py:class:`Environment` to shade the world.
        :param cam_pose_cv: ``(4, 4)`` camera pose in :py:const:`CameraConventions.CV` conventions.
        :param fx: Focal length on the `x` (horizontal) axis.
        :param fy: Focal length on the `y` (vertical) axis.
        :param cx: Principal point `x` (horizontal) axis. Origin is top-left of frame.
        :param cy: Principal point `y` (vertical) axis. Origin is top-left of frame.
        :param sx: Sensor size `x` (horizontal), or *width*.
        :param sy: Sensor size `y` (vertical), or *height*.
        :param pixel_per_unit:
            If the previous focal lengths and sizes are in physical units like *mm*,
            a value can be specified for pixel density per unit.
            Defaults to ``1.0``.

        :returns:
            ``(H, W, 3)`` RGB image ranging in ``[0, 1]``,
            where ``[H W] = round([sy sx] * pixel_per_unit)``.
        """
        ss = pixel_per_unit
        h, w = round(sy * ss), round(sx * ss)
        rays_o, rays_d = rays.get_cam_rays_cv(
            cam_pose_cv, fx * ss, fy * ss, cx * ss, cy * ss, h, w
        )
        hit_id, hit_d, hit_u, hit_v = self.bvh.raycast(rays_o, rays_d)
        face_normals = numpy.where(hit_id[..., None] >= 0, self.tri_nors[hit_id], 0)
        return numpy.clip(env.shade(face_normals).reshape(h, w, 3), 0, 1)
