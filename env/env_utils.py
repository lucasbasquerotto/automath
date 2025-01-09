import importlib
import pkgutil
import typing
from env.core import INode

T = typing.TypeVar("T")

def import_submodules(package_name: str):
    package = importlib.import_module(package_name)
    for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(module_name)

def get_all_subclasses(cls: type[T]) -> set[type[T]]:
    subclasses = set([cls])
    for subclass in cls.__subclasses__():
        subclasses.update(get_all_subclasses(subclass))
    return subclasses

def get_all_subclasses_sorted(cls: type[T]) -> list[type[T]]:
    result_set = get_all_subclasses(cls)
    result = sorted([t for t in result_set], key=lambda t: f'{t.__module__}.{t.__qualname__}')
    return result

def load_all_subclasses_sorted() -> list[type[INode]]:
    import_submodules('env')
    return get_all_subclasses_sorted(INode)
