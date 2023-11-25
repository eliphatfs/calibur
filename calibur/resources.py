import trimesh
from .generic_utils import get_relative_path


def get_blender_cube() -> trimesh.Trimesh:
    return trimesh.load(get_relative_path("blender_cube.obj"))


def get_spot() -> trimesh.Trimesh:
    """
    The spot mesh from Keenan's 3D Model Repository
    "Robust fairing via conformal curvature flow"
    The mesh is in Unity coordinates (Z forward, Y up).
    The animal looks into -Z.
    """
    return trimesh.load(get_relative_path("spot_quadrangulated.obj"))


def get_monkey() -> trimesh.Trimesh:
    """
    The blender monkey head mesh.
    The mesh is in GL coordinates (-Z forward, Y up).
    The animal looks into Z.
    """
    return trimesh.load(get_relative_path("monkey.obj"))


def get_monkey_glb() -> trimesh.Trimesh:
    """
    The blender monkey head mesh in GLB format.
    The mesh is in GLTF coordinates (-Z forward, Y up).
    The animal looks into Z.
    """
    return trimesh.load(get_relative_path("monkey.obj"))
