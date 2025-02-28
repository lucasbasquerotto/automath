import time
import typing

T = typing.TypeVar("T")

def run_test(
    name: str,
    fn: typing.Callable[[], T],
    fn_additional_info: typing.Callable[[T], str] | None = None,
) -> T:
    start = time.time()
    result = fn()
    end = time.time()
    suffix = ''
    if fn_additional_info is not None:
        suffix = fn_additional_info(result)
        suffix = (' <' + suffix + '>') if suffix else ''
    print(f'[{name}] Time taken: {end-start:.2f} seconds{suffix}')
    return result

def run_module_test(
    fn: typing.Callable[[], T],
    fn_additional_info: typing.Callable[[T], str] | None = None,
) -> T:
    name = fn.__module__
    return run_test('>' + name, fn, fn_additional_info)
