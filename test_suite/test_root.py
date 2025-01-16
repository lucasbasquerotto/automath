from env import core
from env import state
from env import action
from env import meta_env
from env import full_state
from test_suite import basic_test
from test_suite.action_impl import action_01_state_meta

def test() -> list[full_state.FullState]:
    final_states: list[full_state.FullState] = []
    final_states += basic_test.test()
    final_states += action_01_state_meta.test()

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
