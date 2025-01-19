from env import core, full_state
# from env import core, full_state, node_types
# from env.goal_env import GoalEnv

def test_control_flow() -> list[full_state.FullState]:
    # goal = node_types.HaveScratch.with_goal(core.Void())
    # env = GoalEnv(goal=goal)

    assert core.If(
        core.IntBoolean.create_true(),
        core.Integer(1),
        core.Integer(2),
    ).run() == core.Integer(1)

    assert core.If(
        core.IntBoolean.create(),
        core.Integer(1),
        core.Integer(2),
    ).run() == core.Integer(2)

    assert core.Loop(
        core.FunctionExpr.with_child(
            core.If(
                core.IsEmpty(core.Param.from_int(1)),
                core.LoopGuard.with_args(
                    condition=core.IntBoolean.create_true(),
                    result=core.Integer(9)
                ),
                core.LoopGuard.with_args(
                    condition=core.IntBoolean.create(),
                    result=core.DefaultGroup(core.Integer(0), core.Param.from_int(1))
                ),
            )
        ),
    ).run() == core.Optional(core.DefaultGroup(core.Integer(0), core.Optional(core.Integer(9))))

    return []

def test() -> list[full_state.FullState]:
    final_states: list[full_state.FullState] = []
    final_states += test_control_flow()
    return final_states
