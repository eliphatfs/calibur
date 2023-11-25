import numpy


def get_dx_viewport_rays(h, w, z_start):
    """
    Shoot rays from ``z_start`` towards z +inf in viewport space.

    Viewport *xy* lies in ``(0-w, 0-h)``,
    where ``0/w/h`` represent borders (instead of center of border pixels) of viewport.

    The origin is top-left (vulkan/directx convention).

    :returns: a tuple ``(rays_o, rays_d)``, both of shape ``(h * w, 3)``.
    """
    x_base = numpy.arange(w, dtype=numpy.float32) + 0.5
    y_base = numpy.arange(h, dtype=numpy.float32) + 0.5
    rays_d = numpy.array([0., 0., 1.], dtype=numpy.float32)
    rays_d = numpy.broadcast_to(rays_d, [h * w, 3])
    rays_ox, rays_oy = numpy.meshgrid(x_base, y_base, indexing='xy')
    rays_o = numpy.stack([rays_ox, rays_oy, numpy.full_like(rays_ox, z_start)], axis=-1)
    rays_o = rays_o.reshape(rays_d.shape)
    return rays_o, rays_d


def get_view_ray_directions_cv(h, w, fx, fy, cx, cy, norm=False):
    """
    :param h: ``(int)`` height of camera.
    :param w: ``(int)`` width of camera.
    :param intrinsics: ``[fx, fy, cx, cy]`` CV intrinsics of camera.
    :param norm: whether the ray directions should be normalized.

    :returns: ``(h, w, 3)`` directions of the rays in OpenCV camera coordinate
    """
    x = numpy.arange(w, dtype=numpy.float32) + 0.5
    y = numpy.arange(h, dtype=numpy.float32) + 0.5
    x, y = numpy.meshgrid(x, y, indexing='xy')
    # (h, w, *3*)
    directions = numpy.stack([(x - cx) / fx, (y - cy) / fy, numpy.ones_like(x)], axis=-1)
    if norm:
        norm = numpy.linalg.norm(directions, axis=-1, keepdims=True)
        directions = directions / norm
    return directions


def transform_ray_directions_cv(directions, c2w, norm=False):
    """
    :meta private:
    :param directions: ``(N, 3)`` precomputed ray directions in camera coordinate.
    :param c2w:
        ``(3+, 4)`` transformation matrix from camera coordinate to world coordinate,
        or the pose of the camera. Camera is in OpenCV convention.
    :param norm: whether to normalize the resulting directions.
    :returns: a tuple ``(rays_o, rays_d)``, both of shape ``(N, 3)``.
    """
    rays_d = directions.reshape(-1, 3) @ c2w[:3, :3].T  # (*, 3)
    rays_o = numpy.broadcast_to(c2w[:3, 3], rays_d.shape)  # (*, 3)
    if norm:
        norm = numpy.linalg.norm(rays_d, axis=-1, keepdims=True)
        rays_d = rays_d / norm
    return rays_o, rays_d


def get_cam_rays_cv(c2w, fx, fy, cx, cy, h, w):
    """
    Get camera rays (origin, dir) from extrinsics and intrinsics in OpenCV convention.

    :returns: a tuple ``(rays_o, rays_d)``.

        * **rays_o** -- ``(h * w, 3)``, the origin of the rays in world coordinates.
        * **rays_d** -- ``(h * w, 3)``, the normalized direction of the rays in world coordinates.
    """
    directions = get_view_ray_directions_cv(
        h, w, fx, fy, cx, cy, norm=False
    )
    rays_o, rays_d = transform_ray_directions_cv(directions, c2w, norm=True)
    return rays_o, rays_d
