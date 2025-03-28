import typing
from env import (
    core,
    state,
    action,
    meta_env,
full_state,
)
from env.symbol import Symbol
from env.goal_env import GoalEnv
from env.node_types import HaveScratch
from test_suite import (
    test_utils,
    basic_test,
    boolean_test,
    arithmetic_test,
    indices_test,
    control_flow_test,
)
from test_suite.action_impl import (
    action_00_action_meta,
    action_01_state_meta,
    action_02_manage_scratch,
    action_03_define_scratch,
    action_04_update_scratch,
    action_05_manage_args_group,
)
from utils.env_logger import env_logger

def _final_verification(final_states: list[full_state.FullState]):
    for i_case, full_state_case in enumerate(final_states):
        history_amount = full_state_case.history_amount()
        for i in range(history_amount):
            index = i+1
            current_fs, action_data_opt = full_state_case.at_history(index)
            action_data = action_data_opt.value
            assert current_fs is not None
            assert action_data is not None
            raw_action_opt = action_data.raw_action.apply().cast(
                core.IOptional[meta_env.IRawAction])
            action_opt = action_data.action.apply().cast(core.IOptional[meta_env.IAction])
            raw_action = raw_action_opt.value
            act = raw_action if raw_action is not None else action_opt.value
            assert act is not None
            assert isinstance(act, action.BaseAction)

            next_fs = act.run_action(current_fs)

            if i < history_amount - 1:
                actual_next_fs, _ = full_state_case.at_history(index+1)
            else:
                actual_next_fs = full_state_case

            expected_next_state = next_fs.current_state.apply()
            actual_next_state = actual_next_fs.current_state.apply()
            if expected_next_state != actual_next_state:
                print('actual_next_state:', Symbol.default(actual_next_state))
                print('expected_next_state:', Symbol.default(expected_next_state))
            assert expected_next_state == actual_next_state, f'{i_case}-{i}'
            last_action_data_opt = actual_next_fs.last_action_data
            if last_action_data_opt != action_data_opt:
                print('last_action_data_opt:', Symbol.default(last_action_data_opt.as_node))
                print('action_data_opt:', Symbol.default(action_data_opt.as_node))
            assert last_action_data_opt == action_data_opt, f'{i_case}-{i}'
            next_fs = next_fs.minimal_meta()
            actual_next_fs = actual_next_fs.minimal_meta()
            if next_fs != actual_next_fs:
                print('actual_next_fs:', Symbol.default(actual_next_fs))
                print('expected_next_fs:', Symbol.default(next_fs))
            assert next_fs == actual_next_fs, f'{i_case}-{i}'

            exception = action_data.exception.apply().cast(core.IOptional[core.IExceptionInfo])

            if exception.is_empty().as_bool:
                full_out, _ = act.inner_run(current_fs)
                output = full_out.output.apply().cast(action.IActionOutput)
                next_state = full_out.new_state.apply().cast(state.State)
                actual_output = action_data.output.apply().cast(
                    core.IOptional[action.IActionOutput]
                ).value
                assert next_state == expected_next_state
                assert output == actual_output

    return final_states

def run_main_test(fn: typing.Callable[[], list[full_state.FullState]]):
    def fn_additional_info(final_states: list[full_state.FullState]):
        amount = len(final_states)
        action_amount = sum([fs.history_amount() for fs in final_states])
        return f"Completed tests: {amount} ({action_amount} actions)"
    return test_utils.run_module_test(fn, fn_additional_info)

def _main_tests(fast: bool) -> list[full_state.FullState]:
    final_states: list[full_state.FullState] = []

    final_states += run_main_test(basic_test.test)

    final_states += run_main_test(boolean_test.test)
    final_states += run_main_test(arithmetic_test.test)
    final_states += run_main_test(indices_test.test)

    final_states += run_main_test(action_00_action_meta.test)
    final_states += run_main_test(lambda: action_01_state_meta.test(fast))
    final_states += run_main_test(action_02_manage_scratch.test)
    final_states += run_main_test(action_03_define_scratch.test)
    final_states += run_main_test(action_04_update_scratch.test)
    final_states += run_main_test(action_05_manage_args_group.test)

    final_states += run_main_test(control_flow_test.test)

    return final_states

def initialize_cache() -> None:
    GoalEnv(HaveScratch.with_goal(core.Void()))

def all_tests(fast: bool) -> list[full_state.FullState]:
    try:
        test_utils.run_test('initialize_cache', initialize_cache)
        final_states = test_utils.run_test('main_tests', lambda: _main_tests(fast))
        test_utils.run_test('final_verification', lambda: _final_verification(final_states))
        return final_states
    except core.InvalidNodeException as e:
        symbol = Symbol.default(e.info.as_node)
        env_logger.debug(str(symbol), exc_info=e)
        raise e

def test(fast: bool = False) -> list[full_state.FullState]:
    return test_utils.run_test('all_tests', lambda: all_tests(fast))
