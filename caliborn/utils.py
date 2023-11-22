from typing import Sequence
from numpy import ndarray
import numpy
from . import Q


# Shorthand
cat = numpy.concatenate


def supercat(tensors: Sequence[ndarray], dim: int = 0):
    """
    Similar to `numpy.concatenate`, but supports broadcasting. For example:

    [M, 32], [N, 1, 64] -- supercat 2 --> [N, M, 96]
    """
    ndim = max(x.ndim for x in tensors)
    tensors = [x.reshape(*[1] * (ndim - x.ndim), *x.shape) for x in tensors]
    shape = [max(x.size(i) for x in tensors) for i in range(ndim)]
    shape[dim] = -1
    tensors = [numpy.broadcast_to(x, shape) for x in tensors]
    return cat(tensors, dim)


def homogeneous(coords: ndarray):
    """
    Concatenate a `1` dimension to the coords to transform it into homogeneous coordinates.
    """
    return supercat([coords, numpy.ones(1, coords.dtype)], dim=-1)


def transform_point(xyz: ndarray, matrix: ndarray) -> ndarray:
    """
    xyz: [..., D]
    matrix: [..., D + 1, D + 1]
    -> [..., D]

    The output is normalized to `w=1` in homogeneous coordinates.
    The function is not limited to 3D coords.
    """
    h = homogeneous(xyz)
    xyzw = numpy.matmul(h, matrix)
    xyz = xyzw[..., :-1] / xyzw[..., -1:]
    return xyz


def transform_vector(xyz: ndarray, matrix: ndarray) -> ndarray:
    """
    xyz: [..., D]
    matrix: [..., D + 1, D + 1]
    -> [..., D + 1]

    The function is not limited to 3D coords.
    """
    return numpy.matmul(xyz, matrix[:-1, :-1])


def sample2d(im: ndarray, xy: ndarray) -> ndarray:
    """
    Bilinear sampling with UV coordinate in Blender convention.
    xy should lie in [0, 1] mostly (out-of-bounds values are clamped).
    The origin (0, 0) is the bottom-left corner of the image.

    im: [H, W, ?]
    xy: [..., 2]
    -> [..., ?]
    """
    x, y = xy[Q.x], xy[Q.y]
    x = numpy.asarray(x) * im.shape[1] - 0.5
    y = (1 - numpy.asarray(y)) * im.shape[0] - 0.5

    x0 = numpy.floor(x).astype(int)
    x1 = x0 + 1
    y0 = numpy.floor(y).astype(int)
    y1 = y0 + 1

    x0 = numpy.clip(x0, 0, im.shape[1] - 1)
    x1 = numpy.clip(x1, 0, im.shape[1] - 1)
    y0 = numpy.clip(y0, 0, im.shape[0] - 1)
    y1 = numpy.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def sign2d(p1: ndarray, p2: ndarray, p3: ndarray) -> ndarray:
    return (p1[Q.x] - p3[Q.x]) * (p2[Q.y] - p3[Q.y]) - (p2[Q.x] - p3[Q.x]) * (p1[Q.y] - p3[Q.y])


def point_in_tri2d(pt: ndarray, v1: ndarray, v2: ndarray, v3: ndarray) -> ndarray:
    d1 = sign2d(pt, v1, v2)
    d2 = sign2d(pt, v2, v3)
    d3 = sign2d(pt, v3, v1)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    # TODO: implement top-left rule for edge cases to support transparency
    return ~(has_neg & has_pos)
