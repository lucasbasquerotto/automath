import importlib
import pkgutil
import typing
from env.core import INode

T = typing.TypeVar("T")

def type_sorter_key(t: type) -> str:
    return f'{t.__module__}.{t.__qualname__}'

def import_submodules(package_name: str):
    package = importlib.import_module(package_name)
    for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(module_name)

def get_all_superclasses(cls: type[T], top_cls: type[T] | None) -> set[type[T]]:
    superclasses = set([])
    for superclass in cls.__bases__:
        if top_cls is None or issubclass(superclass, top_cls):
            superclasses.add(superclass)
            superclasses.update(get_all_superclasses(superclass, top_cls))
    return superclasses

def _get_all_subclasses_repeated(cls: type[T]) -> list[type[T]]:
    subclasses = list([cls])
    inner_subs = sorted(cls.__subclasses__(), key=type_sorter_key)
    for subclass in inner_subs:
        subs = _get_all_subclasses_repeated(subclass)
        subclasses += subs
    return subclasses

def load_all_superclasses(cls: type[INode]) -> set[type[INode]]:
    return get_all_superclasses(cls, INode)

def load_all_subclasses_sorted() -> list[type[INode]]:
    import_submodules('env')
    subs = _get_all_subclasses_repeated(INode)
    result: list[type[INode]] = [INode]
    result_set = set(result)
    for sub in subs:
        if sub not in result_set:
            bases = [b for b in sub.__bases__ if issubclass(b, INode)]
            if all(b in result_set for b in bases):
                result_set.add(sub)
                result.append(sub)
    return result
