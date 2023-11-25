import numpy
from .generic_utils import supercat, unbind
from .ndarray_extension import NDArray, cast_graphics


def homogeneous(coords: NDArray):
    """
    Concatenate a `1` dimension to the coords to transform it into homogeneous coordinates.
    """
    return supercat([coords, numpy.ones(1, coords.dtype)], dim=-1)


def transform_point(xyz: NDArray, matrix: NDArray) -> NDArray:
    """
    Transforms points with transformation matrices.
    The function is not limited to 3D coords.

    The output is normalized to `w=1` in homogeneous coordinates.

    :param xyz: ``(..., D)``
    :param matrix: ``(..., D + 1, D + 1)``
    :returns: ``(..., D)``
    """
    h = homogeneous(xyz)
    xyzw = numpy.matmul(h, matrix.swapaxes(-1, -2))
    xyz = xyzw[..., :-1] / xyzw[..., -1:]
    return xyz


def transform_vector(xyz: NDArray, matrix: NDArray) -> NDArray:
    """
    Transforms directions with transformation matrices.
    The function is not limited to 3D coords.

    :param xyz: ``(..., D)``
    :param matrix: ``(..., D + 1, D + 1)``
    :returns: ``(..., D)``
    """
    return numpy.matmul(xyz, matrix[:-1, :-1].swapaxes(-1, -2))


@cast_graphics
def sample2d(im: NDArray, xy: NDArray) -> NDArray:
    """
    Bilinear sampling with UV coordinate in Blender convention.

    The origin ``(0, 0)`` is the bottom-left corner of the image as in most UV conventions.

    :param im: ``(H, W, ?)`` image.
    :param xy: ``(..., 2)``, should lie in ``[0, 1]`` mostly (out-of-bounds values are clamped).
    :returns: ``(..., ?)`` sampled points.
    """
    x, y = xy.x, xy.y
    x = x * im.shape[1] - 0.5
    y = (1 - y) * im.shape[0] - 0.5

    x0 = numpy.floor(x).astype(int)
    x1 = x0 + 1
    y0 = numpy.floor(y).astype(int)
    y1 = y0 + 1

    x0s = numpy.clip(x0, 0, im.shape[1] - 1)
    x1s = numpy.clip(x1, 0, im.shape[1] - 1)
    y0s = numpy.clip(y0, 0, im.shape[0] - 1)
    y1s = numpy.clip(y1, 0, im.shape[0] - 1)

    Ia = numpy.squeeze(im[y0s, x0s], axis=-2)
    Ib = numpy.squeeze(im[y1s, x0s], axis=-2)
    Ic = numpy.squeeze(im[y0s, x1s], axis=-2)
    Id = numpy.squeeze(im[y1s, x1s], axis=-2)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


@cast_graphics
def sign2d(p1: NDArray, p2: NDArray, p3: NDArray) -> NDArray:
    """
    :meta private:
    """
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)


def point_in_tri2d(pt: NDArray, v1: NDArray, v2: NDArray, v3: NDArray) -> NDArray:
    """
    Decide whether points lie in 2D triangles. Arrays broadcast.

    :param pt: ``(..., 2)`` points to decide.
    :param v1: ``(..., 2)`` first vertex in triangles.
    :param v2: ``(..., 2)`` second vertex in triangles.
    :param v3: ``(..., 2)`` third vertex in triangles.
    :returns: ``(..., 1)`` boolean result of points lie in triangles.
    """
    d1 = sign2d(pt, v1, v2)
    d2 = sign2d(pt, v2, v3)
    d3 = sign2d(pt, v3, v1)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    # TODO: implement top-left rule for edge cases to support transparency
    return ~(has_neg & has_pos)


def magnitude(vecs: NDArray):
    """
    Compute the magnitudes or lengths of vecs.

    :param vecs: ``(..., D)``.
    :returns: ``(..., 1)``.
    """
    return numpy.linalg.norm(vecs, axis=-1, keepdims=True)


def normalized(vecs: NDArray):
    """
    Safely normalize a batch of vecs.

    :param vecs: ``(..., D)``.
    :returns: ``(..., D)``, the same shape as input.
    """
    return vecs / (magnitude(vecs) + 1e-12)


def compute_tri3d_normals(tris: NDArray):
    """
    Compute face normals of triangles.

        Assumes the CCW order as forward face.

    :param tris: ``(..., 3, 3)`` where the last dimension is xyz and the second last is 3 vertices.
    :returns: ``(..., 3)`` of normals of each triangle.
    """
    v0, v1, v2 = unbind(tris, axis=-2)
    u = v1 - v0
    v = v2 - v0
    return normalized(numpy.cross(u, v))
