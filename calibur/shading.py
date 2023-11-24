import cv2
import numpy
from .graphic_utils import homogeneous
from .generic_utils import get_relative_path


class SHEnvironment(object):
    sh_int_64: numpy.ndarray = numpy.load(get_relative_path("sh_int_64.npz"))["coeff"]

    def __init__(self, sh: numpy.ndarray) -> None:
        """
        The environment lighting method by Ramamoorthi et al.
        "An Efficient Representation for Irradiance Environment Maps" (2001)
        The environment light only depends on surface normal,
        and is represented by Spherical Harmonics (SH) coefficients.
        
        sh: [9, 3] SH coefficients of the environment map.
        """
        self.sh = numpy.array(sh, dtype=numpy.float32)
        c1, c2, c3, c4, c5 = 0.429043, 0.511664, 0.743125, 0.886227, 0.247708
        L00, L1m1, L10, L11, L2m2, L2m1, L20, L21, L22 = self.sh
        self.quadratic = numpy.array([
            [c1 * L22,  c1 * L2m2,  c1 * L21, c2 * L11],
            [c1 * L2m2, -c1 * L22, c1 * L2m1, c2 * L1m1],
            [c1 * L21,  c1 * L2m1,  c3 * L20, c2 * L10],
            [c2 * L11,  c2 * L1m1,  c2 * L10, c4 * L00 - c5 * L20]
        ], dtype=numpy.float32)  # 4, 4, 3
        assert list(self.quadratic.shape) == [4, 4, 3]

    @classmethod
    def from_image(cls, img: numpy.ndarray):
        """
        Box-filtered integration of equirect environment map into SH coefficients.
        img: [H, W, C], preferably W = 2H
        """
        img = img.astype(numpy.float32)
        img = cv2.resize(img, (64, 32), interpolation=cv2.INTER_AREA)
        sh = numpy.einsum("hws,hwc->sc", cls.sh_int_64, img)
        return cls(sh)

    def shade(self, normals):
        """
        Normals: [N, 3]
        """
        nh = homogeneous(normals)
        return numpy.einsum("...p,pqd,...q->...d", nh, self.quadratic, nh)


class SampleEnvironments:
    grace_cathedral = SHEnvironment([
        [ 0.79,  0.44,  0.54],
        [ 0.39,  0.35,  0.6 ],
        [-0.34, -0.18, -0.27],
        [-0.29, -0.06,  0.01],
        [-0.11, -0.05, -0.12],
        [-0.26, -0.22, -0.47],
        [-0.16, -0.09, -0.15],
        [ 0.56,  0.21,  0.14],
        [ 0.21, -0.05, -0.3 ]
    ])
    eucalyptus_grove = SHEnvironment([
        [ 0.38,  0.43,  0.45],
        [ 0.29,  0.36,  0.41],
        [ 0.04,  0.03,  0.01],
        [-0.1 , -0.1 , -0.09],
        [-0.06, -0.06, -0.04],
        [ 0.01, -0.01, -0.05],
        [-0.09, -0.13, -0.15],
        [-0.06, -0.05, -0.04],
        [ 0.02, -0.  , -0.05]
    ])
