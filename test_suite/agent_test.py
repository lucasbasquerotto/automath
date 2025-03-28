from env.action import RawAction
from env import core, action, meta_env, state, symbol
from env.full_state import FullState
from env.goal_env import GoalEnv
from env.node_types import HaveScratch
from agent.demo_agent import DemoAgent

def replay_actions_with_demo_agent(full_state: FullState) -> bool:
    """
    Extracts successful actions from a FullState history, creates a DemoAgent with these actions,
    then replays them on a fresh initial state.

    Args:
        full_state: The FullState with a history of actions

    Returns:
        Tuple containing:
        - The final FullState after replaying all actions
        - Boolean indicating whether replay was successful (True) or not (False)
    """
    # Extract successful states and actions from history
    next_states: list[state.State] = []
    successful_actions: list[RawAction] = []

    node_types = full_state.node_types()
    history_amount = full_state.history_amount()
    for i in range(history_amount):
        index = i+1
        current_fs, action_data_opt = full_state.at_history(index)
        action_data = action_data_opt.value
        assert current_fs is not None
        assert action_data is not None

        exception = action_data.exception.apply().cast(core.IOptional[core.IExceptionInfo])

        if exception.is_empty().as_bool:
            if i < history_amount - 1:
                actual_next_fs, _ = full_state.at_history(index+1)
            else:
                actual_next_fs = full_state

            next_state = actual_next_fs.current_state.apply().real(state.State)
            next_states.append(next_state)

            raw_action_opt = action_data.raw_action.apply().cast(
                core.IOptional[meta_env.IRawAction])
            action_opt = action_data.action.apply().cast(core.IOptional[meta_env.IAction])
            raw_action = raw_action_opt.value
            if raw_action is not None:
                assert isinstance(raw_action, action.RawAction)
                successful_actions.append(raw_action)
            else:
                act = action_opt.value
                assert act is not None
                assert isinstance(act, action.IBasicAction)
                raw_action = RawAction.from_basic_action(
                    action=act,
                    full_state=current_fs)
                successful_actions.append(raw_action)

    if not successful_actions:
        return False

    # Create a DemoAgent with the extracted actions
    demo_agent = DemoAgent(successful_actions)

    initial_state, _ = full_state.at_history(1)
    env = GoalEnv(
        goal=HaveScratch.with_goal(core.Void()),
        fn_initial_state=lambda _: initial_state)

    for i in range(len(successful_actions)):
        full_state = env.full_state
        selected = demo_agent.select_action(full_state)
        new_full_state, _, _, _ = env.step(selected)
        new_state = new_full_state.current_state.apply().real(state.State)
        assert new_state == next_states[i], \
            f"Error after the action #{i+1}:\n\n" \
            f"Expected {symbol.Symbol(node=next_states[i], node_types=node_types)}" \
            f", got {symbol.Symbol(node=new_state, node_types=node_types)}"

    return True
