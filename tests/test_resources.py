import os
import unittest
import numpy
import trimesh
import calibur.resources as resx


def pathof(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", name)


class TestResources(unittest.TestCase):

    def test_blender_cube(self):
        self.assertTrue(numpy.allclose(
            trimesh.load(pathof("blender_cube.obj")).triangles,
            resx.get_blender_cube(), atol=1e-3
        ))

    def test_monkey(self):
        self.assertTrue(numpy.allclose(
            trimesh.load(pathof("monkey.obj")).triangles,
            resx.get_monkey(), atol=1e-3
        ))

    def test_spot(self):
        self.assertTrue(numpy.allclose(
            trimesh.load(pathof("spot_quadrangulated.obj"), force='mesh').triangles,
            resx.get_spot(), atol=1e-3
        ))
