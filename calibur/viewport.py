from .generic_utils import cat
from .ndarray_extension import cast_graphics, NDArray, treemap_cast_graphics
from .projection import linear_depth_gl


@cast_graphics
def gl_ndc_to_dx_ndc(ndc: NDArray):
    """
    DX NDC differs to GL in that depth lies in ``[0, 1]`` instead of ``[-1, 1]``.

    This is a simple transform compressing the *Z* dimension of NDC.
    """
    return cat([ndc.x, ndc.y, ndc.z * 0.5 + 0.5], axis=-1)


@treemap_cast_graphics
def gl_ndc_to_gl_viewport(ndc: NDArray, w: float, h: float, near=None, far=None):
    """
    GL NDC to GL viewport coordinates.
    
        Notice that GL viewport origin is the *bottom-left* corner.

    If ``near`` and ``far`` are not None, the resulting *Z* is linear depth.
    Otherwise *Z* is kept unchanged.
    """
    if near is None and far is None:
        return cat([(ndc.x + 1) * (w / 2), (ndc.y + 1) * (h / 2), ndc.z], axis=-1)
    else:
        assert near is not None and far is not None, [near, far]
        z = linear_depth_gl(ndc.z, near, far)
        return cat([(ndc.x + 1) * (w / 2), (ndc.y + 1) * (h / 2), z], axis=-1)


@treemap_cast_graphics
def gl_viewport_to_dx_viewport(vp: NDArray, h: float):
    """
    GL viewport to DX/Vulkan viewport coordinates.

        They differ in that DX viewport origin is *top-left* compared to *bottom-left* in GL.
    """
    return cat([vp.x, h - vp.y, vp.z], axis=-1)


def gl_ndc_to_dx_viewport(ndc: NDArray, w: float, h: float, near=None, far=None):
    """
    GL NDC to DX viewport coordinates.

    If ``near`` and ``far`` are not None, the resulting *Z* is linear depth.
    Otherwise *Z* is kept unchanged.
    """
    return gl_viewport_to_dx_viewport(gl_ndc_to_gl_viewport(ndc, w, h, near, far), h)
