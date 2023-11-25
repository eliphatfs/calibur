import numpy
from .graphic_utils import homogeneous, sample2d
from .generic_utils import get_relative_path
from .conventions import convert_pose, WorldConventions


class Environment(object):
    """
    Abstract base class for an shading environment.

    Available implementations:
    :py:class:`NormalCaptureEnvironment`,
    :py:class:`SHEnvironment`,
    :py:class:`SampleEnvironments`.
    """

    def shade(self, normals: numpy.ndarray):
        """
        :param normals: ``(N, 3)`` world-space normals.
        :returns: ``(N, 3)`` RGB values in ``[0, 1]`` range.
        """
        raise NotImplementedError


class NormalCaptureEnvironment(Environment):
    """
    A special 'environment' to capture the world-space normals of rendered object.
    """

    def shade(self, normals):
        return normals * numpy.array([0.5, 0.5, 0.5], dtype=numpy.float32) + 0.5


class SHEnvironment(Environment):
    _sh_int_64: numpy.ndarray = numpy.load(get_relative_path("sh_int_64.npz"))["coeff"]

    def __init__(self, sh: numpy.ndarray, world_convention) -> None:
        """
        The environment lighting method by Ramamoorthi et al.
        "An Efficient Representation for Irradiance Environment Maps" (2001).
        The environment light only depends on surface normal,
        and is represented by Spherical Harmonics (SH) coefficients.
        
        :param sh: ``(9, 3)`` SH coefficients of the environment map.
        :param world_convention: target world convention for shading.

        Two built-in environments can be found at :py:class:`SampleEnvironments`.
        """
        self.sh = numpy.array(sh, dtype=numpy.float32)
        c1, c2, c3, c4, c5 = 0.429043, 0.511664, 0.743125, 0.886227, 0.247708
        L00, L1m1, L10, L11, L2m2, L2m1, L20, L21, L22 = self.sh
        # Ix: a point in world_convention
        # Ax: a point in GL convention
        A = convert_pose(numpy.eye(4, dtype=numpy.float32), WorldConventions.GL, world_convention)
        # x^TQx => x^TA^TQAx
        self.quadratic = numpy.array([
            [c1 * L22,  c1 * L2m2,  c1 * L21, c2 * L11],
            [c1 * L2m2, -c1 * L22, c1 * L2m1, c2 * L1m1],
            [c1 * L21,  c1 * L2m1,  c3 * L20, c2 * L10],
            [c2 * L11,  c2 * L1m1,  c2 * L10, c4 * L00 - c5 * L20]
        ], dtype=numpy.float32)  # 4, 4, 3
        self.quadratic = numpy.einsum("ap,pqc,qb->abc", A.T, self.quadratic, A)
        assert list(self.quadratic.shape) == [4, 4, 3]

    @classmethod
    def from_image(cls, img: numpy.ndarray, world_convention):
        """
        Box-filtered integration of equirect environment map into SH coefficients.

        :param img: ``(H, W, C)``, preferably ``W = 2H``.
        :param world_convention: target world convention for shading.

        Experimental.
        """
        img = img.astype(numpy.float32)
        gx = numpy.linspace(0, 1, 64, dtype=numpy.float32)
        gy = 1 - numpy.linspace(0, 1, 32, dtype=numpy.float32)
        grid = numpy.stack(numpy.meshgrid(gx, gy, indexing='xy'), axis=-1)
        img = sample2d(img, grid)
        sh = numpy.einsum("hws,hwc->sc", cls._sh_int_64, img) / numpy.pi
        return cls(sh, world_convention)

    def shade(self, normals):
        nh = homogeneous(normals)
        return numpy.einsum("...p,pqd,...q->...d", nh, self.quadratic, nh)


class SampleEnvironments:
    """
    Example environments in the paper
    "An Efficient Representation for Irradiance Environment Maps" (2001).
    """

    @staticmethod
    def grace_cathedral(world_convention):
        return SHEnvironment([
            [ 0.79,  0.44,  0.54],
            [ 0.39,  0.35,  0.6 ],
            [-0.34, -0.18, -0.27],
            [-0.29, -0.06,  0.01],
            [-0.11, -0.05, -0.12],
            [-0.26, -0.22, -0.47],
            [-0.16, -0.09, -0.15],
            [ 0.56,  0.21,  0.14],
            [ 0.21, -0.05, -0.3 ]
        ], world_convention)

    @staticmethod
    def eucalyptus_grove(world_convention):
        return SHEnvironment([
            [ 0.38,  0.43,  0.45],
            [ 0.29,  0.36,  0.41],
            [ 0.04,  0.03,  0.01],
            [-0.1 , -0.1 , -0.09],
            [-0.06, -0.06, -0.04],
            [ 0.01, -0.01, -0.05],
            [-0.09, -0.13, -0.15],
            [-0.06, -0.05, -0.04],
            [ 0.02, -0.  , -0.05]
        ], world_convention)
