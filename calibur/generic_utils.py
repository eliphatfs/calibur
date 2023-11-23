from typing import Sequence, TypeVar, Generic, Callable, Union, Type
from typing_extensions import Literal
from functools import wraps
import numpy


# Shorthand
cat = numpy.concatenate


def supercat(tensors: Sequence[numpy.ndarray], dim: int = 0):
    """
    Similar to `numpy.concatenate`, but supports broadcasting. For example:

    [M, 32], [N, 1, 64] -- supercat 2 --> [N, M, 96]
    """
    ndim = max(x.ndim for x in tensors)
    tensors = [x.reshape(*[1] * (ndim - x.ndim), *x.shape) for x in tensors]
    shape = [max(x.size(i) for x in tensors) for i in range(ndim)]
    shape[dim] = -1
    tensors = [numpy.broadcast_to(x, shape) for x in tensors]
    return cat(tensors, dim)


TElem = TypeVar('TElem')
TRes = TypeVar('TRes')


class TC(Generic[TElem]):
    pass


def type_match(matcher: Union[Type, Sequence[Type]], missing: Literal["ignore", "error"] = "ignore"):
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


def container_catamorphism(
    data: TC[TElem], func: Callable[[TElem], TRes]
) -> TC[TRes]:
    """
    Transforms `TElem` in `list`, `dict`, `tuple`, `set` with `func`.
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
        Context manager for numpy warnings
        Possible options: {'ignore', 'warn', 'raise', 'call', 'print', 'log'}
        """
        self.pop = numpy.seterr(all=all, divide=divide, over=over, under=under, invalid=invalid)

    def __enter__(self):
        return self

    def __exit__(self, ty, value, tb):
        numpy.seterr(**self.pop)
