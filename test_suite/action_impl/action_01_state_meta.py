import typing
from env import core
from env import action_impl
from env import state
from env import meta_env
from env import full_state
from env import node_types as node_types_module
from env.goal_env import GoalEnv
from test_suite import test_utils

def get_current_state(env: GoalEnv):
    return env.full_state.nested_arg(
        (full_state.FullState.idx_current, full_state.HistoryNode.idx_state)
    ).apply().cast(state.State)

def get_remaining_steps(env: GoalEnv) -> int | None:
    value = env.full_state.nested_arg(
        (
            full_state.FullState.idx_current,
            full_state.HistoryNode.idx_meta_data,
            meta_env.MetaData.idx_remaining_steps,
        )
    ).apply().cast(core.IOptional[core.IInt]).value
    return value.as_int if value is not None else None

def get_last_history_action(env: GoalEnv):
    history = env.full_state.history.apply().cast(full_state.HistoryGroupNode)
    last = history.as_tuple[-1]
    return last.action_data.apply().inner_arg(
        core.Optional.idx_value
    ).apply().cast(full_state.BaseActionData)

def get_from_int_type_index(node_type: type[core.INode], env: GoalEnv):
    selected_types = env.full_state.meta.apply().nested_arg((
        meta_env.MetaInfo.idx_from_int_group,
        meta_env.SubtypeOuterGroup.idx_subtypes,
    )).apply().cast(meta_env.GeneralTypeGroup)
    meta_idx = selected_types.as_tuple.index(node_type.as_type()) + 1
    return meta_idx

def goal_test():
    params = (core.Param.from_int(1), core.Param.from_int(2), core.Param.from_int(3))
    p1, p2, p3 = params
    scratch_goal_1 = core.FunctionExpr(
        core.Protocol(
            core.TypeAliasGroup(),
            core.RestTypeGroup(core.INode.as_type()),
            core.INode.as_type(),
        ),
        core.Or(
            core.And(p1, p2, core.IntBoolean(1)),
            core.And(p2, p3),
        ),
    )
    goal_1 = node_types_module.HaveScratch.with_goal(scratch_goal_1)
    scratch_goal_2 = core.Not(core.Eq(core.IBoolean.true(), core.IBoolean.false()))
    goal_2 = node_types_module.HaveScratch.with_goal(scratch_goal_2)
    scratch_goal_3 = core.FunctionCall.define(
        state.FunctionId(1),
        core.DefaultGroup(
            core.DefaultGroup(core.Integer(3), core.Integer(4)),
            core.Eq(p1, core.Integer(27)),
        )
    )
    goal_3 = node_types_module.HaveScratch.with_goal(scratch_goal_3)

    def has_goal(env: GoalEnv, goal: meta_env.IGoal):
        selected_goal = env.full_state.nested_arg(
            (full_state.FullState.idx_meta, meta_env.MetaInfo.idx_goal)
        ).apply()
        return selected_goal == goal

    def fn_before_final_state(
        meta: meta_env.MetaInfo,
        goal: node_types_module.HaveScratch,
        scratches: typing.Sequence[core.INode | None],
    ) -> full_state.FullState:
        state_meta = state.StateMetaInfo.with_goal_expr(goal)
        return full_state.FullState.with_args(
            meta=meta,
            current=full_state.HistoryNode.with_args(
                state=state.State.from_raw(
                    meta_info=state_meta,
                    scratches=scratches,
                ),
                meta_data=meta_env.MetaData.with_args(
                    remaining_steps=(
                        len(scratches)
                        if scratch_goal_1 in scratches
                        else None
                    )
                ),
            )
        )

    def test_goal_specified(
        goal: node_types_module.HaveScratch,
        scratch_goal: core.INode,
        direct: bool,
        error=False,
    ):
        env = GoalEnv(
            goal=goal,
            fn_initial_state=lambda meta: fn_before_final_state(
                meta=meta,
                goal=goal,
                scratches=[
                    scratch_goal
                    if scratch_goal is not None
                    else goal.goal_inner_expr.apply()
                ]
            ),
            max_steps=(
                500
                if scratch_goal == scratch_goal_1
                else None
            )
        )
        assert has_goal(env=env, goal=goal)
        env.full_state.validate()

        current_state = get_current_state(env)
        initial_state = current_state
        prev_remaining_steps = get_remaining_steps(env)

        assert current_state == state.State.from_raw(
            scratches=[scratch_goal],
        )
        assert env.full_state.goal_achieved() is False

        meta_idx = get_from_int_type_index(state.StateScratchIndex, env)
        raw_action = action_impl.VerifyGoal.from_raw(0, meta_idx, 1)
        full_action = action_impl.VerifyGoal(
            core.Optional(),
            full_state.MetaFromIntTypeIndex(meta_idx),
            core.Integer(1),
        )
        output = action_impl.VerifyGoalOutput(
            core.Optional(),
            state.StateScratchIndex(1),
        )
        act = output if direct else raw_action
        env.step(act)
        if prev_remaining_steps is not None:
            remaining_steps = get_remaining_steps(env)
            assert remaining_steps == prev_remaining_steps - 1
            prev_remaining_steps = remaining_steps
        current_state = get_current_state(env)
        last_history_action = get_last_history_action(env)

        if not error:
            state_meta = state.StateMetaInfo.with_goal_achieved(
                state.GoalAchieved.achieved()
            )
            assert current_state.meta_info.apply() == state_meta
            assert current_state == state.State.from_raw(
                meta_info=state_meta,
                scratches=[scratch_goal],
            )
            assert last_history_action == full_state.SuccessActionData.from_args(
                action=core.Optional(output if direct else full_action),
                output=core.Optional(output),
                exception=core.Optional(),
            )
            assert env.full_state.goal_achieved() is True
        else:
            assert current_state == initial_state
            expected_action: core.Optional[core.INode] = core.Optional(
                output if direct else full_action
            )
            actual_action = last_history_action.action.apply().cast(core.IOptional)
            assert expected_action == actual_action
            actual_output_opt = last_history_action.output.apply().cast(core.IOptional)
            assert actual_output_opt == core.Optional(output)
            core.Not(
                last_history_action.exception.apply().cast(core.IOptional).is_empty()
            ).raise_on_false()
            assert env.full_state.goal_achieved() is False

        return env.full_state

    def test_goal_specified_with_group(error=False):
        goal = state.GoalGroup(
            goal_1,
            state.GoalGroup(
                goal_2,
                goal_3,
            ),
        )
        scratch_nest_1 = core.NestedArgIndexGroup.from_ints([1])
        scratch_nest_2 = core.NestedArgIndexGroup.from_ints([2, 1])
        scratch_nest_3 = core.NestedArgIndexGroup.from_ints([2, 2])
        scratches = [
            scratch_goal_1,
            scratch_goal_2,
            scratch_goal_3,
            scratch_nest_1,
            scratch_nest_2,
            scratch_nest_3,
        ]
        env = GoalEnv(
            goal=goal,
            fn_initial_state=lambda meta: fn_before_final_state(
                meta=meta,
                goal=goal,
                scratches=scratches,
            ),
            max_steps=10000
        )
        assert has_goal(env=env, goal=goal)
        env.full_state.validate()

        current_state = get_current_state(env)
        prev_remaining_steps = get_remaining_steps(env)

        state_meta = state.StateMetaInfo.with_goal_achieved(state.GoalAchievedGroup(
            state.GoalAchieved.create(),
            state.GoalAchievedGroup(
                state.GoalAchieved.create(),
                state.GoalAchieved.create(),
            ),
        ))
        assert current_state == state.State.from_raw(
            meta_info=state_meta,
            scratches=scratches,
        )
        assert env.full_state.goal_achieved() is False

        meta_idx = get_from_int_type_index(state.StateScratchIndex, env)

        # Achieve 1st Goal
        raw_action = action_impl.VerifyGoal.from_raw(1+3, meta_idx, 1)
        full_action = action_impl.VerifyGoal(
            core.Optional(state.StateScratchIndex(1+3)),
            full_state.MetaFromIntTypeIndex(meta_idx),
            core.Integer(1),
        )
        output = action_impl.VerifyGoalOutput(
            core.Optional(scratch_nest_1),
            state.StateScratchIndex(1),
        )
        env.step(raw_action)
        if prev_remaining_steps is not None:
            remaining_steps = get_remaining_steps(env)
            assert remaining_steps == prev_remaining_steps - 1
            prev_remaining_steps = remaining_steps
        current_state = get_current_state(env)
        last_history_action = get_last_history_action(env)

        # Verify that 1st Goal was achieved
        state_meta = state.StateMetaInfo.with_goal_achieved(state.GoalAchievedGroup(
            state.GoalAchieved.achieved(),
            state.GoalAchievedGroup(
                state.GoalAchieved.create(),
                state.GoalAchieved.create(),
            ),
        ))
        assert current_state.meta_info.apply() == state_meta
        assert current_state == state.State.from_raw(
            meta_info=state_meta,
            scratches=scratches,
        )
        assert last_history_action == full_state.SuccessActionData.from_args(
            action=core.Optional(full_action),
            output=core.Optional(output),
            exception=core.Optional(),
        )
        assert env.full_state.goal_achieved() is False

        # Achieve 2nd Goal
        output = action_impl.VerifyGoalOutput(
            core.Optional(scratch_nest_2),
            state.StateScratchIndex(2),
        )
        env.step(output)
        if prev_remaining_steps is not None:
            remaining_steps = get_remaining_steps(env)
            assert remaining_steps == prev_remaining_steps - 1
            prev_remaining_steps = remaining_steps
        current_state = get_current_state(env)
        last_history_action = get_last_history_action(env)

        # Verify that 2nd Goal was achieved
        state_meta = state.StateMetaInfo.with_goal_achieved(state.GoalAchievedGroup(
            state.GoalAchieved.achieved(),
            state.GoalAchievedGroup(
                state.GoalAchieved.achieved(),
                state.GoalAchieved.create(),
            ),
        ))
        assert current_state.meta_info.apply() == state_meta
        assert current_state == state.State.from_raw(
            meta_info=state_meta,
            scratches=scratches,
        )
        assert last_history_action == full_state.SuccessActionData.from_args(
            action=core.Optional(output),
            output=core.Optional(output),
            exception=core.Optional(),
        )
        assert env.full_state.goal_achieved() is False

        # Achieve 3rd Goal
        previous_state = current_state
        scratch_idx = 3 if not error else 1
        raw_action = action_impl.VerifyGoal.from_raw(3+3, meta_idx, scratch_idx)
        full_action = action_impl.VerifyGoal(
            core.Optional(state.StateScratchIndex(3+3)),
            full_state.MetaFromIntTypeIndex(meta_idx),
            core.Integer(scratch_idx),
        )
        output = action_impl.VerifyGoalOutput(
            core.Optional(scratch_nest_3),
            state.StateScratchIndex(scratch_idx),
        )
        env.step(raw_action)
        if prev_remaining_steps is not None:
            remaining_steps = get_remaining_steps(env)
            assert remaining_steps == prev_remaining_steps - 1
            prev_remaining_steps = remaining_steps
        current_state = get_current_state(env)
        last_history_action = get_last_history_action(env)

        if not error:
            # Verify that 3rd Goal was achieved
            state_meta = state.StateMetaInfo.with_goal_achieved(state.GoalAchievedGroup(
                state.GoalAchieved.achieved(),
                state.GoalAchievedGroup(
                    state.GoalAchieved.achieved(),
                    state.GoalAchieved.achieved(),
                ),
            ))
            assert current_state.meta_info.apply() == state_meta
            assert current_state == state.State.from_raw(
                meta_info=state_meta,
                scratches=scratches,
            )
            assert last_history_action == full_state.SuccessActionData.from_args(
                action=core.Optional(full_action),
                output=core.Optional(output),
                exception=core.Optional(),
            )
            assert env.full_state.goal_achieved() is True
        else:
            # Verify that 3rd Goal not achieved
            assert current_state == previous_state
            expected_action = core.Optional(full_action)
            actual_action = last_history_action.action.apply().cast(core.IOptional)
            assert expected_action == actual_action
            actual_output_opt = last_history_action.output.apply().cast(core.IOptional)
            assert actual_output_opt == core.Optional(output)
            core.Not(
                last_history_action.exception.apply().cast(core.IOptional).is_empty()
            ).raise_on_false()
            assert env.full_state.goal_achieved() is False

        return env.full_state

    def main() -> list[full_state.FullState]:
        final_states: list[full_state.FullState] = []
        final_states.append(test_goal_specified(goal_1, scratch_goal_1, direct=True))
        final_states.append(test_goal_specified(goal_2, scratch_goal_2, direct=False))
        final_states.append(test_goal_specified(goal_3, scratch_goal_3, direct=True))
        final_states.append(test_goal_specified(goal_1, scratch_goal_2, direct=True, error=True))
        final_states.append(test_goal_specified(goal_2, scratch_goal_3, direct=False, error=True))
        final_states.append(test_goal_specified_with_group())
        final_states.append(test_goal_specified_with_group(error=True))
        return final_states

    return main()

def dynamic_goal_test():
    p1_args = (
        core.NearParentScope.create(),
        core.Integer(1),
    )
    scratch_dynamic_goal = core.FunctionExpr(
        core.Protocol(
            core.TypeAliasGroup(),
            core.RestTypeGroup(core.INode.as_type()),
            core.INode.as_type(),
        ),
        core.And(
            core.Or(
                core.Param(*p1_args),
                core.Param.from_int(2),
                core.IntBoolean.from_int(1),
            ),
            core.Or(
                core.FunctionCall.define(
                    core.TypeNode(core.Param),
                    core.DefaultGroup(*p1_args)
                ),
                core.GreaterThan(
                    core.Param.from_int(3),
                    core.Param.from_int(4)
                ),
            ),
        )
    )
    dynamic_goal_expr = node_types_module.HaveScratch.with_goal(scratch_dynamic_goal)
    goal_1 = node_types_module.HaveDynamicGoal.with_goal(dynamic_goal_expr)
    goal_2 = node_types_module.HaveDynamicGoalAchieved.create()

    def has_goal(env: GoalEnv, goal: meta_env.IGoal):
        selected_goal = env.full_state.nested_arg(
            (full_state.FullState.idx_meta, meta_env.MetaInfo.idx_goal)
        ).apply()
        return selected_goal == goal

    def fn_before_final_state(
        meta: meta_env.MetaInfo,
        goal: node_types_module.HaveScratch,
        scratches: typing.Sequence[core.INode | None],
    ) -> full_state.FullState:
        state_meta = state.StateMetaInfo.with_goal_expr(goal)
        return full_state.FullState.with_args(
            meta=meta,
            current=full_state.HistoryNode.with_args(
                state=state.State.from_raw(
                    meta_info=state_meta,
                    scratches=scratches,
                ),
                meta_data=meta_env.MetaData.with_args(
                    remaining_steps=5
                ),
            )
        )

    goal = state.GoalGroup(
        goal_1,
        goal_2,
    )
    scratch_nest_1 = core.NestedArgIndexGroup.from_ints([1])
    scratch_nest_2 = core.NestedArgIndexGroup.from_ints([2])
    scratches = [
        scratch_dynamic_goal,
        dynamic_goal_expr,
        scratch_nest_1,
        scratch_nest_2,
    ]
    env = GoalEnv(
        goal=goal,
        fn_initial_state=lambda meta: fn_before_final_state(
            meta=meta,
            goal=goal,
            scratches=scratches,
        ),
        max_steps=1000,
    )
    assert has_goal(env=env, goal=goal)
    env.full_state.validate()

    current_state = get_current_state(env)
    prev_remaining_steps = get_remaining_steps(env)

    state_meta = state.StateMetaInfo.with_goal_achieved(state.GoalAchievedGroup(
        state.GoalAchieved.create(),
        state.GoalAchieved.create(),
    ))
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    raw_action = action_impl.CreateDynamicGoal.from_raw(2, 0, 0)
    full_action = action_impl.CreateDynamicGoal(
        state.StateScratchIndex(2),
    )
    output = action_impl.CreateDynamicGoalOutput(
        state.StateDynamicGoalIndex(1),
        dynamic_goal_expr,
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    dynamic_goal_group = state.DynamicGoalGroup(
        state.DynamicGoal.from_goal_expr(dynamic_goal_expr)
    )
    state_meta = state_meta.with_new_args(
        dynamic_goal_group=dynamic_goal_group,
    )
    assert current_state.meta_info.apply() == state_meta
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.SuccessActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    meta_idx = get_from_int_type_index(state.StateDynamicGoalIndex, env)
    raw_action = action_impl.VerifyGoal.from_raw(3, meta_idx, 1)
    full_action = action_impl.VerifyGoal(
        core.Optional(state.StateScratchIndex(3)),
        full_state.MetaFromIntTypeIndex(meta_idx),
        core.Integer(1),
    )
    output = action_impl.VerifyGoalOutput(
        core.Optional(core.NestedArgIndexGroup.from_ints([1])),
        state.StateDynamicGoalIndex(1),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    state_meta = state_meta.with_new_args(
        goal_achieved=state.GoalAchievedGroup(
            state.GoalAchieved.achieved(),
            state.GoalAchieved.create(),
        ),
    )
    assert current_state.meta_info.apply() == state_meta
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.SuccessActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    meta_idx = get_from_int_type_index(state.StateScratchIndex, env)
    raw_action = action_impl.DefineScratchFromInt.from_raw(2, meta_idx, 1)
    full_action = action_impl.DefineScratchFromInt(
        state.StateScratchIndex(2),
        full_state.MetaFromIntTypeIndex(meta_idx),
        core.Integer(1),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(2),
        state.Scratch(state.StateScratchIndex(1)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[1] = state.StateScratchIndex(1)
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.SuccessActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    raw_action = action_impl.VerifyDynamicGoal.from_raw(1, 0, 2)
    full_action = action_impl.VerifyDynamicGoal(
        state.StateDynamicGoalIndex(1),
        core.Optional.create(),
        state.StateScratchIndex(2),
    )
    output = action_impl.VerifyDynamicGoalOutput(
        state.StateDynamicGoalIndex(1),
        core.Optional.create(),
        state.StateScratchIndex(1),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    dynamic_goal_group = state.DynamicGoalGroup(
        state.DynamicGoal.from_goal_expr(
            dynamic_goal_expr
        ).apply_goal_achieved(state.Optional())
    )
    state_meta = state_meta.with_new_args(
        dynamic_goal_group=dynamic_goal_group,
    )
    assert current_state.meta_info.apply() == state_meta
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.SuccessActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    meta_idx = get_from_int_type_index(state.StateDynamicGoalIndex, env)
    raw_action = action_impl.VerifyGoal.from_raw(4, meta_idx, 1)
    full_action = action_impl.VerifyGoal(
        core.Optional(state.StateScratchIndex(4)),
        full_state.MetaFromIntTypeIndex(meta_idx),
        core.Integer(1),
    )
    output = action_impl.VerifyGoalOutput(
        core.Optional(core.NestedArgIndexGroup.from_ints([2])),
        state.StateDynamicGoalIndex(1),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    state_meta = state_meta.with_new_args(
        goal_achieved=state.GoalAchievedGroup(
            state.GoalAchieved.achieved(),
            state.GoalAchieved.achieved(),
        ),
    )
    assert current_state.meta_info.apply() == state_meta
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.SuccessActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is True

    return [env.full_state]

def test() -> list[full_state.FullState]:
    final_states: list[full_state.FullState] = []
    final_states += test_utils.run_test('>>goal_test', goal_test)
    final_states += test_utils.run_test('>>dynamic_goal_test', dynamic_goal_test)
    return final_states
