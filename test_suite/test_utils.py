import time
import typing
from env import full_state

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

def run_main_test(fn: typing.Callable[[], list[full_state.FullState]]):
    def fn_additional_info(final_states: list[full_state.FullState]):
        amount = len(final_states)
        action_amount = sum([fs.history_amount() for fs in final_states])
        return f"Completed tests: {amount} ({action_amount} actions)"
    return run_module_test(fn, fn_additional_info)

def run_info_test(
    name: str,
    fn: typing.Callable[[], list[full_state.FullState]],
):
    def fn_additional_info(final_states: list[full_state.FullState]):
        amount = len(final_states)
        action_amount = sum([fs.history_amount() for fs in final_states])
        return f"Completed tests: {amount} ({action_amount} actions)"
    return run_test(name, fn, fn_additional_info)
