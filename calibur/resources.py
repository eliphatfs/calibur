import numpy
from .generic_utils import get_relative_path


archive = numpy.load(get_relative_path("data.npz"))


def get_blender_cube() -> numpy.ndarray:
    """
    :returns: triangles ``(N, 3, 3)``.
    """
    return archive["blender_cube"].astype(numpy.float32)


def get_spot() -> numpy.ndarray:
    """
    The spot mesh from Keenan's 3D Model Repository
    "Robust fairing via conformal curvature flow"
    The mesh is in Unity coordinates (Z forward, Y up).
    The animal looks into -Z.
    
    :returns: triangles ``(N, 3, 3)``.
    """
    return archive["spot"].astype(numpy.float32)


def get_monkey() -> numpy.ndarray:
    """
    The blender monkey head mesh.
    The mesh is in GL coordinates (-Z forward, Y up).
    The animal looks into Z.
    
    :returns: triangles ``(N, 3, 3)``.
    """
    return archive["monkey"].astype(numpy.float32)
