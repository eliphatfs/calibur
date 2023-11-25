import os
from typing import Sequence, Union, Type
from typing_extensions import Literal
from functools import wraps
import numpy


cat = numpy.concatenate


def supercat(tensors: Sequence[numpy.ndarray], dim: int = 0):
    """
    Similar to `numpy.concatenate`, but supports broadcasting. For example:

    ``[M, 32], [N, 1, 64] -- supercat 2 --> [N, M, 96]``
    """
    ndim = max(x.ndim for x in tensors)
    tensors = [x.reshape(*[1] * (ndim - x.ndim), *x.shape) for x in tensors]
    shape = [max(x.shape[i] for x in tensors) for i in range(ndim)]
    expand_tensors = []
    for x in tensors:
        shape[dim] = x.shape[dim]
        expand_tensors.append(numpy.broadcast_to(x, shape))
    return cat(expand_tensors, dim)


def type_match(matcher: Union[Type, Sequence[Type]], missing: Literal["ignore", "error"] = "ignore"):
    """
    :meta private:
    """
    def decorator(func):
        @wraps(func)
        def wrapper(elem):
            if isinstance(elem, matcher):
                return func(elem)
            if missing == "ignore":
                return elem
            else:
                raise TypeError("Got unexpected type!", type(elem), "Expected", matcher)
        return wrapper
    return decorator


def container_catamorphism(data, func):
    """
    Transforms leaf elements in ``list``, ``dict``, ``tuple``, ``set`` with ``func``, aka. *tree-map*.
    Nested containers are also supported.
    """
    if isinstance(data, dict):
        return {
            k: container_catamorphism(v, func) for k, v in data.items()
        }
    if isinstance(data, list):
        return [container_catamorphism(x, func) for x in data]
    if isinstance(data, tuple):
        return tuple(container_catamorphism(x, func) for x in data)
    if isinstance(data, set):
        return {container_catamorphism(x, func) for x in data}
    return func(data)


WarningOptionType = Literal['ignore', 'warn', 'raise', 'call', 'print', 'log', None]


class NumPyWarning(object):
    def __init__(
        self,
        all: WarningOptionType = None,
        divide: WarningOptionType = None,
        over: WarningOptionType = None,
        under: WarningOptionType = None,
        invalid: WarningOptionType = None
    ) -> None:
        """
        Context manager for ``numpy`` warnings.
        Possible options are:

        * ``'ignore'``
        * ``'warn'``
        * ``'raise'``
        * ``'call'``
        * ``'print'``
        * ``'log'``
        """
        self.pop = numpy.seterr(all=all, divide=divide, over=over, under=under, invalid=invalid)

    def __enter__(self):
        return self

    def __exit__(self, ty, value, tb):
        numpy.seterr(**self.pop)


def get_relative_path(rel):
    """
    :meta private:
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), rel)


def unbind(arr: numpy.ndarray, axis, keepdims=False):
    """
    Unbinds an NDArray into a list of arrays along an axis.
    """
    if axis < 0:
        axis = arr.ndim + axis
    if keepdims:
        return list(numpy.swapaxes(arr[None], 0, axis + 1))
    else:
        return [numpy.squeeze(x, axis) for x in numpy.swapaxes(arr[None], 0, axis + 1)]
