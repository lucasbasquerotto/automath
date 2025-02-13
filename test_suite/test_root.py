import typing
import time
from env import core
from env import state
from env import action
from env import meta_env
from env import full_state
from env.symbol import Symbol
from test_suite import basic_test, control_flow_test, indices_test
from test_suite.action_impl import (
    action_01_state_meta,
    action_02_manage_scratch,
    action_03_define_scratch,
    action_04_update_scratch,
    action_05_manage_args_group,
)
from utils.env_logger import env_logger

def _run_test(
    name: str,
    fn: typing.Callable[[], list[full_state.FullState]],
) -> list[full_state.FullState]:
    start = time.time()
    result = fn()
    end = time.time()
    print(f'[{name}] Time taken: {end-start:.2f} seconds')
    return result

def _run_module_test(
    fn: typing.Callable[[], list[full_state.FullState]],
) -> list[full_state.FullState]:
    name = fn.__module__
    return _run_test('>' + name, fn)

def _final_verification(final_states: list[full_state.FullState]):
    for i_case, full_state_case in enumerate(final_states):
        history_amount = full_state_case.history_amount()
        for i in range(history_amount):
            index = i+1
            current_fs, action_data_opt = full_state_case.at_history(index)
            action_data = action_data_opt.value
            assert current_fs is not None
            assert action_data is not None
            action_opt = action_data.action.apply().cast(core.IOptional[meta_env.IAction])
            act = action_opt.value
            assert act is not None
            assert isinstance(act, action.BaseAction)

            next_fs = act.run_action(current_fs)

            if i < history_amount - 1:
                actual_next_fs, _ = full_state_case.at_history(index+1)
            else:
                actual_next_fs = full_state_case

            expected_next_state = next_fs.current_state.apply()
            actual_next_state = actual_next_fs.current_state.apply()
            assert expected_next_state == actual_next_state, f'{i_case}-{i}'
            assert next_fs == actual_next_fs, f'{i_case}-{i}'
            last_action_data_opt = actual_next_fs.last_action_data
            assert last_action_data_opt == action_data_opt, f'{i_case}-{i}'

            exception = action_data.exception.apply().cast(core.IOptional[core.IExceptionInfo])

            if exception.is_empty().as_bool:
                full_out = act.inner_run(current_fs)
                output = full_out.output.apply().cast(action.IActionOutput)
                next_state = full_out.new_state.apply().cast(state.State)
                actual_output = action_data.output.apply().cast(
                    core.IOptional[action.IActionOutput]
                ).value
                assert next_state == expected_next_state
                assert output == actual_output

    return final_states

def _main_tests() -> list[full_state.FullState]:
    final_states: list[full_state.FullState] = []
    final_states += _run_module_test(basic_test.test)
    final_states += _run_module_test(control_flow_test.test)
    final_states += _run_module_test(indices_test.test)
    final_states += _run_module_test(action_01_state_meta.test)
    final_states += _run_module_test(action_02_manage_scratch.test)
    final_states += _run_module_test(action_03_define_scratch.test)
    final_states += _run_module_test(action_04_update_scratch.test)
    final_states += _run_module_test(action_05_manage_args_group.test)
    return final_states

def all_tests() -> list[full_state.FullState]:
    try:
        final_states = _run_test('main_tests', _main_tests)
        _run_test('final_verification', lambda: _final_verification(final_states))
        return final_states
    except core.InvalidNodeException as e:
        symbol = Symbol.default(e.info.as_node)
        env_logger.debug(str(symbol), exc_info=e)
        raise e

def test() -> list[full_state.FullState]:
    return _run_test('all_tests', all_tests)
