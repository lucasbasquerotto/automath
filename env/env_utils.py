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

def get_all_subclasses(cls: type[T]) -> set[type[T]]:
    subclasses = set([cls])
    for subclass in cls.__subclasses__():
        subclasses.update(get_all_subclasses(subclass))
    return subclasses

def get_all_subclasses_sorted(cls: type[T]) -> list[type[T]]:
    result_set = get_all_subclasses(cls)
    result = sorted([t for t in result_set], key=type_sorter_key)
    return result

def load_all_superclasses_sorted(cls: type[INode]) -> list[type[INode]]:
    result_set = get_all_superclasses(cls, INode)
    result = sorted([t for t in result_set], key=type_sorter_key)
    return result

def load_all_subclasses_sorted() -> list[type[INode]]:
    import_submodules('env')
    return get_all_subclasses_sorted(INode)
