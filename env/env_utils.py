import typing
import functools
from utils.module_utils import (
    get_all_superclasses,
    import_submodules,
    get_all_subclasses_repeated,
)
from env.core import INode

T = typing.TypeVar("T")

def load_all_superclasses(cls: type[INode]) -> set[type[INode]]:
    return get_all_superclasses(cls, INode)

@functools.cache
def load_all_subclasses_sorted() -> list[type[INode]]:
    import_submodules('env')
    subs = get_all_subclasses_repeated(INode)
    result: list[type[INode]] = [INode]
    result_set = set(result)
    for sub in subs:
        if sub not in result_set:
            bases = [b for b in sub.__bases__ if issubclass(b, INode)]
            if all(b in result_set for b in bases):
                result_set.add(sub)
                result.append(sub)
    return result
