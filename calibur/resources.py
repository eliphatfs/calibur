import trimesh
from .generic_utils import get_relative_path


def get_blender_cube() -> trimesh.Trimesh:
    return trimesh.load(get_relative_path("blender_cube.obj"))
