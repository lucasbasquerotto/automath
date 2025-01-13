from env import core
from env import action_impl
from env import state
from env import meta_env
from env import full_state
from env import node_types as node_types_module
from env.goal_env import GoalEnv

def get_current_state(env: GoalEnv):
    return env.full_state.nested_args(
        (full_state.FullState.idx_current, full_state.HistoryNode.idx_state)
    ).apply().cast(state.State)

def action_impl_test():
    params = (core.Param.from_int(1), core.Param.from_int(2), core.Param.from_int(3))
    p1, p2, p3 = params
    goal = node_types_module.HaveScratch.with_goal(
        core.FunctionExpr.with_child(
            core.Or(
                core.And(p1, p2, core.IntBoolean(1)),
                core.And(p2, p3),
            ),
        )
    )

    env = GoalEnv(goal)
    node_types = env.full_state.node_types()

    selected_goal = env.full_state.nested_args(
        (full_state.FullState.idx_meta, meta_env.MetaInfo.idx_goal)
    ).apply()
    assert selected_goal == goal

    current_state = get_current_state(env)
    assert current_state == state.State.create()

    raise NotImplementedError

def test():
    action_impl_test()
