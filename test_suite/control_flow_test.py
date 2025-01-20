from env import core, full_state, state, meta_env, action_impl, node_types
from env.goal_env import GoalEnv

def get_current_state(env: GoalEnv):
    return env.full_state.nested_args(
        (full_state.FullState.idx_current, full_state.HistoryNode.idx_state)
    ).apply().cast(state.State)

def get_last_history_action(env: GoalEnv):
    history = env.full_state.history.apply().cast(full_state.HistoryGroupNode)
    last = history.as_tuple[-1]
    return last.action_data.apply().nested_arg(
        core.Optional.idx_value
    ).apply().cast(full_state.ActionData)

def get_meta_subgroup_type_index(meta_idx: int, node_type: type[core.INode], env: GoalEnv):
    selected_types = env.full_state.meta.apply().nested_args((
        meta_idx,
        meta_env.SubtypeOuterGroup.idx_subtypes,
    )).apply().cast(meta_env.GeneralTypeGroup)
    meta_idx = selected_types.as_tuple.index(node_type.as_type()) + 1
    return meta_idx

def run(
    env: GoalEnv,
    state_meta: state.StateMetaInfo,
    scratches: list[core.INode | None],
    args_groups: list[state.PartialArgsGroup],
    scratch_idx: int,
    new_scratch: core.INode,
):
    # Run Action
    raw_action = action_impl.RunScratch.from_raw(scratch_idx, scratch_idx, 0)
    full_action = action_impl.RunScratch(
        state.StateScratchIndex(scratch_idx),
        state.StateScratchIndex(scratch_idx),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(scratch_idx),
        state.Scratch(new_scratch),
    )
    env.step(raw_action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches = [
        new_scratch if i == scratch_idx-1 else s
        for i, s in enumerate(scratches)
    ]

    expected_state = state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
        args_groups=args_groups,
    )
    if current_state != expected_state:
        print('current_state:', env.symbol(current_state).to_str())
        print('expected_state:', env.symbol(expected_state).to_str())
    assert current_state == expected_state

    expected_history = full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    if last_history_action != expected_history:
        print('last_history_action:', env.symbol(last_history_action).to_str())
        print('expected_history:', env.symbol(expected_history).to_str())
    assert last_history_action == expected_history

    assert env.full_state.goal_achieved() is False

    return scratches

def has_goal(env: GoalEnv, goal: meta_env.IGoal):
    selected_goal = env.full_state.nested_args(
        (full_state.FullState.idx_meta, meta_env.MetaInfo.idx_goal)
    ).apply()
    return selected_goal == goal

def test_control_flow() -> list[full_state.FullState]:
    goal = node_types.HaveScratch.with_goal(core.Void())
    state_meta = state.StateMetaInfo.with_goal_expr(goal)

    if_scratches: list[core.INode | None] = [
        core.If(
            core.IBoolean.true(),
            core.Integer(1),
            core.Integer(2),
        ),
        core.If(
            core.IBoolean.false(),
            core.Integer(1),
            core.Integer(2),
        ),
    ]
    loop_scratches: list[core.INode | None] = [
        core.Loop(
            core.FunctionExpr.with_node(
                core.If(
                    core.IsEmpty(core.Param.from_int(1)),
                    core.LoopGuard.with_args(
                        condition=core.IBoolean.true(),
                        result=core.Integer(9)
                    ),
                    core.LoopGuard.with_args(
                        condition=core.IBoolean.false(),
                        result=core.DefaultGroup(core.Integer(0), core.Param.from_int(1))
                    ),
                )
            ),
        ),
    ]
    fn_scratches: list[core.INode | None] = [
        core.FunctionCall(
            core.FunctionExpr.with_node(
                core.DefaultGroup(
                    core.DefaultGroup(
                        core.FunctionCall(
                            core.TypeNode(core.LessThan),
                            core.Param.from_int(1),
                        ),
                        core.FunctionCall(
                            core.TypeNode(core.LessThan),
                            core.Param.from_int(2),
                        ),
                        core.FunctionCall(
                            core.TypeNode(core.LessThan),
                            core.Param.from_int(3),
                        ),
                    ),
                    core.DefaultGroup(
                        core.FunctionCall(
                            core.TypeNode(core.GreaterThan),
                            core.Param.from_int(1),
                        ),
                        core.FunctionCall(
                            core.TypeNode(core.GreaterThan),
                            core.Param.from_int(2),
                        ),
                        core.FunctionCall(
                            core.TypeNode(core.GreaterThan),
                            core.Param.from_int(3),
                        ),
                    ),
                ),
            ),
            core.DefaultGroup(
                core.IntGroup.from_ints([1, 2]),
                core.IntGroup.from_ints([2, 2]),
                core.IntGroup.from_ints([2, 1]),
            )
        )
    ]
    scratches = if_scratches + loop_scratches + fn_scratches

    args_groups = [state.PartialArgsGroup.create()]
    env = GoalEnv(
        goal=goal,
        fn_initial_state=lambda meta: full_state.FullState.with_args(
            meta=meta,
            current=full_state.HistoryNode.with_args(
                state=state.State.from_raw(
                    meta_info=state_meta,
                    scratches=scratches,
                    args_groups=args_groups,
                ),
                meta_data=meta_env.MetaData.create(),
            )
        ),
    )
    assert has_goal(env=env, goal=goal)

    # If
    index = 0
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=index+1,
        new_scratch=core.Integer(1),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=index+2,
        new_scratch=core.Integer(2),
    )

    # Loop
    index = len(if_scratches)
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=index+1,
        new_scratch=core.Optional(
            core.DefaultGroup(core.Integer(0), core.Optional(core.Integer(9)))
        ),
    )

    # Function Expression
    index = len(if_scratches + loop_scratches)
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=index+1,
        new_scratch=core.DefaultGroup(
            core.DefaultGroup(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
            core.DefaultGroup(
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
    )

    return [env.full_state]

def test() -> list[full_state.FullState]:
    final_states: list[full_state.FullState] = []
    final_states += test_control_flow()
    return final_states
