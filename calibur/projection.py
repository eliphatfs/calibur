import numpy


def fov_y_to_x(fovy, aspect):
    """
    Compute fov x from y.

    :param aspect: ``W / H``.
    :param fovy: Vertical FOV angle in radians.
    :returns: Horizontal FOV angle in radians.
    """
    return 2 * numpy.arctan(numpy.tan(fovy * 0.5) * aspect)


def fov_to_focal(fov, size):
    """
    Compute fov y and size h to focal y,

    or

    Compute fov x and size w to focal x.
    
    :param fov: Horizontal or vertical FOV angle in radians.
    :param size: Width or height.
    :returns: Focal length in the same unit as ``size``.
    """
    return size / numpy.tan(fov / 2) / 2


def focal_to_fov(focal_length, size):
    """
    Inverse of ``fov_to_focal``.
    Angles are in radians.
    """
    return 2 * numpy.arctan(size / (2 * focal_length))


def projection_gl_persp(w, h, cx, cy, fx, fy, near, far, s=0.0):
    """
    Gets the GL projection P matrix from camera intrinsics.
    The function is unit-insensitive -- the input *c/f/w/h* values can be in any unit
    as long as they are the same (pixel/mm/inch/etc).
    
    ``camera_view_space_pts @ P.T -> gl_clip_space_pts``.

    Applying ``.xyz`` / ``.w`` gives the GL NDC coordinates:

        | ``.x``: left to right => -1 to 1
        | ``.y``: bottom to up => -1 to 1
        | ``.z``: near to far => -1 to 1

    :param w: Width.
    :param h: Height.
    :param cx: Principal point *x* (horizontal).
    :param cy: Principal point *y* (vertical).
    :param fx: Focal length *x* (horizontal).
    :param fy: Focal length *y* (vertical).
    :param near: Camera near clipping plane distance.
    :param far: Camera far clipping plane distance.
    :param s: Camera shear.

    :returns: ``(4, 4)`` GL projection matrix
    """
    f, n = far, near
    return numpy.array([[2*fx/w, -2*s/w,    -2*cx/w+1,              0], 
                        [     0, 2*fy/h,     2*cy/h-1,              0], 
                        [     0,      0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                        [     0,      0,           -1,              0]], dtype=numpy.float32)


def linear_depth_gl(ndc_z, near, far):
    """
    GL NDC Z to linear depth formula.
    """
    return (2.0 * near * far) / (far + near - ndc_z * (far - near))


def intrinsic_cv(cx, cy, fx, fy, s=0.0):
    """
    OpenCV intrinsic matrix construction.
    Input units are usually pixels for CV.

    :param cx: Principal point *x* (horizontal).
    :param cy: Principal point *y* (vertical).
    :param fx: Focal length *x* (horizontal).
    :param fy: Focal length *y* (vertical).
    :param s: Camera shear.

    :returns: ``(4, 4)`` GL projection matrix
    """
    return numpy.array([[fx,  s, cx],
                        [ 0, fy, cy],
                        [ 0,  0,  1]], dtype=numpy.float32)
