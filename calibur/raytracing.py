import numpy
from typing import Optional
from .generic_utils import NumPyWarning, unbind


class BVH(object):
    def __init__(
        self,
        triangles: numpy.ndarray,
        centroids: Optional[numpy.ndarray] = None,
        indices: Optional[numpy.ndarray] = None
    ) -> None:
        """
        Construct BVH from triangles.

        :param triangles: ``(N, 3, 3)`` array of 3 vertices.
        :param centroids: See note below.
        :param indices: See note below.

        ..

            Centroids and indices should only be passed if it is not the root node,
            which should not be used other than by the BVH itself in principle.
        """
        assert triangles.ndim == 3, triangles.shape
        assert triangles.shape[-1] == triangles.shape[-2] == 3, triangles.shape
        if centroids is None:
            centroids = triangles.mean(-2)
        if indices is None:
            indices = numpy.arange(len(triangles), dtype=numpy.int32)
        assert list(centroids.shape) == list(triangles.shape)[:2], [centroids.shape, triangles.shape]
        assert list(indices.shape) == list(triangles.shape)[:1], [indices.shape, triangles.shape]
        self.leaves = None
        self.children = []
        bound_min = triangles.min((0, 1))
        bound_max = triangles.max((0, 1))
        self.bounds = bound_min, bound_max
        if len(triangles) <= 8:
            self.leaves = triangles, indices
        else:
            box_size = bound_max - bound_min
            split_axis = numpy.argmax(box_size)
            med = numpy.median(centroids[..., split_axis])
            split_1 = centroids[..., split_axis] < med
            split_2 = ~split_1
            if split_1.all() or split_2.all():
                self.leaves = triangles, indices
            else:
                for mask in (split_1, split_2):
                    next_tris = triangles[mask]
                    next_centroids = centroids[mask]
                    next_indices = indices[mask]
                    self.children.append(BVH(next_tris, next_centroids, next_indices))

    def _update(self, hit_i, hit_d, hit_u, hit_v, intersect_mask, i, t, u, v):
        update_mask = t < hit_d[intersect_mask]
        hit_i[intersect_mask] = numpy.where(update_mask, i, hit_i[intersect_mask])
        hit_d[intersect_mask] = numpy.where(update_mask, t, hit_d[intersect_mask])
        hit_u[intersect_mask] = numpy.where(update_mask, u, hit_u[intersect_mask])
        hit_v[intersect_mask] = numpy.where(update_mask, v, hit_v[intersect_mask])

    def raycast(self, rays_o: numpy.ndarray, rays_d: numpy.ndarray, far=32767.):
        """
        Cast a batch of rays to the triangles in the BVH.

        :param rays_o: ``(N, 3)`` ray origins.
        :param rays_d: ``(N, 3)`` ray directions (normalized).
        :returns: a tuple ``(hit_i, hit_d, hit_u, hit_v)``.

            * **hit_i** -- ``(N) int32``, index into given triangles, ``-1`` if missing.
            * **hit_d** -- ``(N) float32``, ray travel distances.
            * **hit_u** -- ``(N) float32``, triangle barycentric ``u``.
            * **hit_v** -- ``(N) float32``, triangle barycentric ``v``.

                For barycentric coordinates, ``P = w v1 + u v2 + v v3``.
        """
        epsilon = 1e-6
        bound_min, bound_max = self.bounds
        hit_i = numpy.full([len(rays_o)], -1, dtype=numpy.int32)
        hit_d = numpy.full([len(rays_o)], far, dtype=numpy.float32)
        hit_u = numpy.zeros([len(rays_o)], dtype=numpy.float32)
        hit_v = numpy.zeros([len(rays_o)], dtype=numpy.float32)
        with NumPyWarning(divide='ignore'):
            t1 = (bound_min - rays_o) / rays_d
            t2 = (bound_max - rays_o) / rays_d
            t_min_axes = numpy.minimum(t1, t2).T
            t_min = numpy.maximum.reduce(list(t_min_axes))
            t_max_axes = numpy.maximum(t1, t2).T
            t_max = numpy.minimum.reduce(list(t_max_axes))
            intersect_mask = (t_max > 0) & (t_min <= t_max)
            subrays_o = rays_o[intersect_mask]
            subrays_d = rays_d[intersect_mask]
            if self.leaves is not None:
                tris, inds = self.leaves  # M, 3, 3; M
                subrays_ot = numpy.broadcast_to(subrays_o[:, None], [len(subrays_o), len(inds), 3])
                subrays_dt = numpy.broadcast_to(subrays_d[:, None], [len(subrays_o), len(inds), 3])
                # N, M, 3
                v0, v1, v2 = unbind(tris[None], -2)  # 1, M, 3
                e1 = v1 - v0
                e2 = v2 - v0
                cross = numpy.cross(subrays_dt, e2)
                det = numpy.einsum('nmc,nmc->nm', e1, cross)
                inv_det = numpy.reciprocal(det)
                s = subrays_ot - v0
                u = inv_det * numpy.einsum('nmc,nmc->nm', s, cross)
                scross1 = numpy.cross(s, e1)
                v = inv_det * numpy.einsum('nmc,nmc->nm', subrays_dt, scross1)
                t = inv_det * numpy.einsum('nmc,nmc->nm', e2, scross1)
                hit_mask = (
                    (numpy.abs(det) > epsilon) & (u >= 0) & (v >= 0) & (u + v <= 1)
                )
                t = numpy.where(hit_mask, t, far)
                sel_tri = numpy.argmin(t, axis=1)  # N, M -> N
                t_n = numpy.arange(len(t))
                sel_t = t[t_n, sel_tri]  # N
                sel_u = u[t_n, sel_tri]
                sel_v = v[t_n, sel_tri]
                fill_idx = inds[sel_tri]
                self._update(hit_i, hit_d, hit_u, hit_v, intersect_mask, fill_idx, sel_t, sel_u, sel_v)
        for child in self.children:
            child: BVH
            i, t, u, v = child.raycast(subrays_o, subrays_d, far)
            self._update(hit_i, hit_d, hit_u, hit_v, intersect_mask, i, t, u, v)
        return hit_i, hit_d, hit_u, hit_v
