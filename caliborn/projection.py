import numpy


def fov_y_to_x(fovy, aspect):
    """
    Compute fov x from y
    Aspect means W / H, angles are in radians
    """
    return 2 * numpy.arctan(numpy.tan(fovy * 0.5) * aspect)


def fov_to_focal(fov, size):
    """
    Compute fov y and size h to focal y
    or
    Compute fov x and size w to focal x
    Angles are in radians
    """
    return size / numpy.tan(fov / 2) / 2


def focal_to_fov(focal_length, size):
    """
    Compute fov y from size h and focal y
    or
    Compute fov x from size w and focal x
    Angles are in radians
    """
    return 2 * numpy.arctan(size / (2 * focal_length))


def projection_gl_persp(w, h, cx, cy, fx, fy, near, far, s=0.0):
    """
    Gets the P matrix from camera intrinsics
    camera_view_space_pts @ P.T -> gl_clip_space_pts
    Applying .xyz / .w gives the GL NDC coordinates
    .x: left to right => -1 to 1
    .y: bottom to up => -1 to 1
    .z: near to far => -1 to 1
    """
    f, n = far, near
    return numpy.array([[2*fx/w, -2*s/w,    -2*cx/w+1,              0], 
                        [     0, 2*fy/h,     2*cy/h-1,              0], 
                        [     0,      0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                        [     0,      0,           -1,              0]])


def linear_depth_gl(ndc_z, near, far):
    """
    GL NDC Z to linear depth formula.
    """
    return (2.0 * near * far) / (far + near - ndc_z * (far - near))


def intrinsic_cv(cx, cy, fx, fy, s=0.0):
    """
    OpenCV intrinsic matrix construction
    """
    return numpy.array([[fx,  s, cx],
                        [ 0, fy, cy],
                        [ 0,  0,  1]])
