import time
import typing

T = typing.TypeVar("T")

def run_test(
    name: str,
    fn: typing.Callable[[], T],
) -> T:
    start = time.time()
    result = fn()
    end = time.time()
    print(f'[{name}] Time taken: {end-start:.2f} seconds')
    return result

def run_module_test(
    fn: typing.Callable[[], T],
) -> T:
    name = fn.__module__
    return run_test('>' + name, fn)
