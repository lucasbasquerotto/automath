from env import core, state, full_state, action_impl, meta_env, node_types, action
from env.goal_env import GoalEnv

def get_current_state(env: GoalEnv):
    return env.full_state.nested_args(
        (full_state.FullState.idx_current, full_state.HistoryNode.idx_state)
    ).apply().cast(state.State)

def get_remaining_steps(env: GoalEnv) -> int | None:
    value = env.full_state.nested_args(
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
    return last.action_data.apply().nested_arg(
        core.Optional.idx_value
    ).apply().cast(full_state.ActionData)

def get_default_type_index(node_type: type[core.INode], env: GoalEnv):
    selected_types = env.full_state.meta.apply().nested_args((
        meta_env.MetaInfo.idx_default_group,
        meta_env.SubtypeOuterGroup.idx_subtypes,
    )).apply().cast(meta_env.GeneralTypeGroup)
    meta_idx = selected_types.as_tuple.index(node_type.as_type()) + 1
    return meta_idx

def get_from_int_type_index(node_type: type[core.INode], env: GoalEnv):
    selected_types = env.full_state.meta.apply().nested_args((
        meta_env.MetaInfo.idx_from_int_group,
        meta_env.SubtypeOuterGroup.idx_subtypes,
    )).apply().cast(meta_env.GeneralTypeGroup)
    meta_idx = selected_types.as_tuple.index(node_type.as_type()) + 1
    return meta_idx

def get_from_single_child_type_index(node_type: type[core.INode], env: GoalEnv):
    selected_types = env.full_state.meta.apply().nested_args((
        meta_env.MetaInfo.idx_single_child_group,
        meta_env.SubtypeOuterGroup.idx_subtypes,
    )).apply().cast(meta_env.GeneralTypeGroup)
    meta_idx = selected_types.as_tuple.index(node_type.as_type()) + 1
    return meta_idx

def get_from_full_state_int_index_type_index(node_type: type[core.INode], env: GoalEnv):
    selected_types = env.full_state.meta.apply().nested_args((
        meta_env.MetaInfo.idx_full_state_int_index_group,
        meta_env.SubtypeOuterGroup.idx_subtypes,
    )).apply().cast(meta_env.GeneralTypeGroup)
    meta_idx = selected_types.as_tuple.index(node_type.as_type()) + 1
    return meta_idx

def test_define_scratch() -> list[full_state.FullState]:
    def has_goal(env: GoalEnv, goal: meta_env.IGoal):
        selected_goal = env.full_state.nested_args(
            (full_state.FullState.idx_meta, meta_env.MetaInfo.idx_goal)
        ).apply()
        return selected_goal == goal

    goal = node_types.HaveScratch(core.Void())
    env = GoalEnv(goal=node_types.HaveScratch(core.Void()))
    assert has_goal(env=env, goal=goal)

    current_state = get_current_state(env)
    prev_remaining_steps = get_remaining_steps(env)

    state_meta = state.StateMetaInfo.with_goal_achieved(state.GoalAchieved.create())
    scratches: list[core.INode | None] = []
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    raw_action: action.BaseAction = action_impl.CreateScratch.from_raw(0, 0, 0)
    full_action: action.BaseAction = action_impl.CreateScratch.create()
    output: action.GeneralAction = action_impl.CreateScratchOutput(
        state.StateScratchIndex(1),
        core.Optional.create(),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches.append(None)
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    meta_idx = get_from_int_type_index(core.Var, env)
    raw_action = action_impl.DefineScratchFromInt.from_raw(1, meta_idx, 5)
    full_action = action_impl.DefineScratchFromInt(
        state.StateScratchIndex(1),
        full_state.MetaFromIntTypeIndex(meta_idx),
        core.Integer(5),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        core.Optional(core.Var.from_int(5)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = core.Var.from_int(5)
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    meta_idx = get_from_single_child_type_index(core.Optional, env)
    raw_action = action_impl.DefineScratchFromSingleArg.from_raw(1, meta_idx, 1)
    full_action = action_impl.DefineScratchFromSingleArg(
        state.StateScratchIndex(1),
        full_state.MetaSingleChildTypeIndex(meta_idx),
        state.StateScratchIndex(1),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        core.Optional(core.Optional(core.Var.from_int(5))),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = core.Optional(core.Var.from_int(5))
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    meta_idx = get_from_full_state_int_index_type_index(full_state.MetaAllTypesTypeIndex, env)
    idx_node = env.full_state.node_types().index(core.Param) + 1
    raw_action = action_impl.DefineScratchFromIntIndex.from_raw(1, meta_idx, idx_node)
    full_action = action_impl.DefineScratchFromIntIndex(
        state.StateScratchIndex(1),
        full_state.MetaFullStateIntIndexTypeIndex(meta_idx),
        core.Integer(idx_node),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        core.Optional(core.TypeNode(core.Param)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = core.TypeNode(core.Param)
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    meta_idx = get_from_full_state_int_index_type_index(full_state.MetaAllTypesTypeIndex, env)
    idx_node = env.full_state.node_types().index(core.Param) + 1
    raw_action = action_impl.DefineScratchFromFunctionWithIntArg.from_raw(1, 1, 3)
    full_action = action_impl.DefineScratchFromFunctionWithIntArg(
        state.StateScratchIndex(1),
        state.StateScratchIndex(1),
        core.Integer(3),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        core.Optional(core.Param.from_int(3)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = core.Param.from_int(3)
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    raw_action = action_impl.DefineScratchFromFunctionWithIntArg.from_raw(1, 1, 7)
    full_action = action_impl.DefineScratchFromFunctionWithIntArg(
        state.StateScratchIndex(1),
        state.StateScratchIndex(1),
        core.Integer(7),
    )
    fn_call: core.INode = core.FunctionCall(
        core.Param.from_int(3),
        core.IntGroup.from_ints([7]),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        core.Optional(fn_call),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = fn_call
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )

    # Run Action
    meta_idx = get_from_full_state_int_index_type_index(full_state.MetaAllTypesTypeIndex, env)
    idx_node = env.full_state.node_types().index(core.NestedArgIndexGroup) + 1
    raw_action = action_impl.DefineScratchFromIntIndex.from_raw(1, meta_idx, idx_node)
    full_action = action_impl.DefineScratchFromIntIndex(
        state.StateScratchIndex(1),
        full_state.MetaFullStateIntIndexTypeIndex(meta_idx),
        core.Integer(idx_node),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        core.Optional(core.TypeNode(core.NestedArgIndexGroup)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = core.TypeNode(core.NestedArgIndexGroup)
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    raw_action = action_impl.DefineScratchFromFunctionWithIntArg.from_raw(1, 1, 7)
    full_action = action_impl.DefineScratchFromFunctionWithIntArg(
        state.StateScratchIndex(1),
        state.StateScratchIndex(1),
        core.Integer(7),
    )
    fn_call = core.NestedArgIndexGroup(core.Integer(7))
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        core.Optional(fn_call),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = fn_call
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    raw_action = action_impl.CreateScratch.from_raw(0, 0, 0)
    full_action = action_impl.CreateScratch.create()
    output = action_impl.CreateScratchOutput(
        state.StateScratchIndex(2),
        core.Optional.create(),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches.append(None)
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    meta_idx = get_from_full_state_int_index_type_index(full_state.MetaAllTypesTypeIndex, env)
    idx_node = env.full_state.node_types().index(core.Optional) + 1
    raw_action = action_impl.DefineScratchFromIntIndex.from_raw(2, meta_idx, idx_node)
    full_action = action_impl.DefineScratchFromIntIndex(
        state.StateScratchIndex(2),
        full_state.MetaFullStateIntIndexTypeIndex(meta_idx),
        core.Integer(idx_node),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(2),
        core.Optional(core.TypeNode(core.Optional)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[1] = core.TypeNode(core.Optional)
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    raw_action = action_impl.DefineScratchFromFunctionWithSingleArg.from_raw(2, 2, 1)
    full_action = action_impl.DefineScratchFromFunctionWithSingleArg(
        state.StateScratchIndex(2),
        state.StateScratchIndex(2),
        state.StateScratchIndex(1),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(2),
        core.Optional(core.Optional(fn_call)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[1] = core.Optional(fn_call)
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    meta_idx = get_from_full_state_int_index_type_index(full_state.MetaAllTypesTypeIndex, env)
    idx_node = env.full_state.node_types().index(core.DefaultGroup) + 1
    raw_action = action_impl.DefineScratchFromIntIndex.from_raw(1, meta_idx, idx_node)
    full_action = action_impl.DefineScratchFromIntIndex(
        state.StateScratchIndex(1),
        full_state.MetaFullStateIntIndexTypeIndex(meta_idx),
        core.Integer(idx_node),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        core.Optional(core.TypeNode(core.DefaultGroup)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = core.TypeNode(core.DefaultGroup)
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    raw_action = action_impl.DefineScratchFromFunctionWithSingleArg.from_raw(2, 1, 2)
    full_action = action_impl.DefineScratchFromFunctionWithSingleArg(
        state.StateScratchIndex(2),
        state.StateScratchIndex(1),
        state.StateScratchIndex(2),
    )
    fn_call = core.DefaultGroup(core.Optional(fn_call))
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(2),
        core.Optional(fn_call),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[1] = fn_call
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    meta_idx = get_from_int_type_index(state.FunctionId, env)
    raw_action = action_impl.DefineScratchFromInt.from_raw(1, meta_idx, 10)
    full_action = action_impl.DefineScratchFromInt(
        state.StateScratchIndex(1),
        full_state.MetaFromIntTypeIndex(meta_idx),
        core.Integer(10),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        core.Optional(state.FunctionId.from_int(10)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = state.FunctionId.from_int(10)
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    raw_action = action_impl.DefineScratchFromFunctionWithSingleArg.from_raw(1, 1, 2)
    full_action = action_impl.DefineScratchFromFunctionWithSingleArg(
        state.StateScratchIndex(1),
        state.StateScratchIndex(1),
        state.StateScratchIndex(2),
    )
    fn_call = core.FunctionCall(
        state.FunctionId.from_int(10),
        core.DefaultGroup(fn_call),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        core.Optional(fn_call),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = fn_call
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    meta_idx = get_from_full_state_int_index_type_index(full_state.MetaAllTypesTypeIndex, env)
    idx_node = env.full_state.node_types().index(core.ExtendedTypeGroup) + 1
    raw_action = action_impl.DefineScratchFromIntIndex.from_raw(1, meta_idx, idx_node)
    full_action = action_impl.DefineScratchFromIntIndex(
        state.StateScratchIndex(1),
        full_state.MetaFullStateIntIndexTypeIndex(meta_idx),
        core.Integer(idx_node),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        core.Optional(core.TypeNode(core.ExtendedTypeGroup)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = core.TypeNode(core.ExtendedTypeGroup)
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    raw_action = action_impl.DefineScratchFromFunctionWithArgs.from_raw(1, 1, 0)
    full_action = action_impl.DefineScratchFromFunctionWithArgs(
        state.StateScratchIndex(1),
        state.StateScratchIndex(1),
        core.Optional(),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        core.Optional(core.ExtendedTypeGroup.create()),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = core.ExtendedTypeGroup.create()
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    meta_idx = get_from_full_state_int_index_type_index(full_state.MetaAllTypesTypeIndex, env)
    idx_node = env.full_state.node_types().index(core.And) + 1
    raw_action = action_impl.DefineScratchFromIntIndex.from_raw(1, meta_idx, idx_node)
    full_action = action_impl.DefineScratchFromIntIndex(
        state.StateScratchIndex(1),
        full_state.MetaFullStateIntIndexTypeIndex(meta_idx),
        core.Integer(idx_node),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        core.Optional(core.TypeNode(core.And)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = core.TypeNode(core.And)
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    raw_action = action_impl.DefineScratchFromFunctionWithArgs.from_raw(1, 1, 0)
    full_action = action_impl.DefineScratchFromFunctionWithArgs(
        state.StateScratchIndex(1),
        state.StateScratchIndex(1),
        core.Optional(),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        core.Optional(core.And()),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = core.And()
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    meta_idx = get_from_int_type_index(state.FunctionId, env)
    raw_action = action_impl.DefineScratchFromInt.from_raw(1, meta_idx, 2)
    full_action = action_impl.DefineScratchFromInt(
        state.StateScratchIndex(1),
        full_state.MetaFromIntTypeIndex(meta_idx),
        core.Integer(2),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        core.Optional(state.FunctionId.from_int(2)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = state.FunctionId.from_int(2)
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    raw_action = action_impl.DefineScratchFromFunctionWithArgs.from_raw(2, 1, 0)
    full_action = action_impl.DefineScratchFromFunctionWithArgs(
        state.StateScratchIndex(2),
        state.StateScratchIndex(1),
        core.Optional(),
    )
    fn_call = core.FunctionCall(
        state.FunctionId.from_int(2),
        core.DefaultGroup(),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(2),
        core.Optional(fn_call),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[1] = fn_call
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    raw_action = action_impl.CreateArgsGroup.from_raw(2, 0, 0)
    full_action = action_impl.CreateArgsGroup(
        core.Integer(2),
        core.Optional(),
        core.Optional(),
    )
    output = action_impl.CreateArgsGroupOutput(
        state.StateArgsGroupIndex(1),
        state.PartialArgsGroup.from_int(2),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    args_group = state.PartialArgsGroup.from_int(2)
    args_groups = tuple([args_group])
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
        args_groups=args_groups,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    raw_action = action_impl.DefineArgsGroup.from_raw(1, 1, 1)
    full_action = action_impl.DefineArgsGroup(
        state.StateArgsGroupIndex(1),
        core.NodeArgIndex(1),
        state.StateScratchIndex(1),
    )
    output = action_impl.DefineArgsGroupArgOutput(
        state.StateArgsGroupIndex(1),
        core.NodeArgIndex(1),
        state.OptionalContext(state.FunctionId.from_int(2)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    args_group = state.PartialArgsGroup(
        core.ExtendedTypeGroup.from_int(2),
        core.LaxOpaqueScope(
            core.ScopeId(1),
            core.OptionalValueGroup(
                core.Optional(state.FunctionId.from_int(2)),
                core.Optional(),
            ),
        )
    )
    args_groups = tuple([args_group])
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
        args_groups=args_groups,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    raw_action = action_impl.DefineArgsGroup.from_raw(1, 2, 1)
    full_action = action_impl.DefineArgsGroup(
        state.StateArgsGroupIndex(1),
        core.NodeArgIndex(2),
        state.StateScratchIndex(1),
    )
    output = action_impl.DefineArgsGroupArgOutput(
        state.StateArgsGroupIndex(1),
        core.NodeArgIndex(2),
        state.OptionalContext(state.FunctionId.from_int(2)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    args_group = state.PartialArgsGroup(
        core.ExtendedTypeGroup.from_int(2),
        core.LaxOpaqueScope(
            core.ScopeId(1),
            core.OptionalValueGroup(
                core.Optional(state.FunctionId.from_int(2)),
                core.Optional(state.FunctionId.from_int(2)),
            ),
        )
    )
    args_groups = tuple([args_group])
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
        args_groups=args_groups,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    raw_action = action_impl.DefineScratchFromFunctionWithArgs.from_raw(2, 1, 1)
    full_action = action_impl.DefineScratchFromFunctionWithArgs(
        state.StateScratchIndex(2),
        state.StateScratchIndex(1),
        core.Optional(state.StateArgsGroupIndex(1)),
    )
    fn_call = core.FunctionCall(
        state.FunctionId.from_int(2),
        core.DefaultGroup(
            state.FunctionId.from_int(2),
            state.FunctionId.from_int(2),
        ),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(2),
        core.Optional(fn_call),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[1] = fn_call
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
        args_groups=args_groups,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    meta_idx = get_from_full_state_int_index_type_index(full_state.MetaAllTypesTypeIndex, env)
    idx_node = env.full_state.node_types().index(core.Eq) + 1
    raw_action = action_impl.DefineScratchFromIntIndex.from_raw(1, meta_idx, idx_node)
    full_action = action_impl.DefineScratchFromIntIndex(
        state.StateScratchIndex(1),
        full_state.MetaFullStateIntIndexTypeIndex(meta_idx),
        core.Integer(idx_node),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        core.Optional(core.TypeNode(core.Eq)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = core.TypeNode(core.Eq)
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
        args_groups=args_groups,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    raw_action = action_impl.DefineScratchFromFunctionWithArgs.from_raw(2, 1, 1)
    full_action = action_impl.DefineScratchFromFunctionWithArgs(
        state.StateScratchIndex(2),
        state.StateScratchIndex(1),
        core.Optional(state.StateArgsGroupIndex(1)),
    )
    fn_call = core.Eq(
        state.FunctionId.from_int(2),
        state.FunctionId.from_int(2),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(2),
        core.Optional(fn_call),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[1] = fn_call
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
        args_groups=args_groups,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    raw_action = action_impl.DeleteArgsGroupOutput.from_raw(1, 0, 0)
    full_action = action_impl.DeleteArgsGroupOutput(
        state.StateArgsGroupIndex(1),
    )
    output = full_action
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    # DefineScratchFromScratchNode

    # Run Action
    raw_action = action_impl.ClearScratch.from_raw(1, 0, 0)
    full_action = action_impl.ClearScratch(
        state.StateScratchIndex(1),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        core.Optional(),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = None
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    meta_idx = get_default_type_index(core.Void, env)
    raw_action = action_impl.DefineScratchFromDefault.from_raw(1, meta_idx, 0)
    full_action = action_impl.DefineScratchFromDefault(
        state.StateScratchIndex(1),
        full_state.MetaDefaultTypeIndex(meta_idx),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        core.Optional(core.Void()),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = core.Void()
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
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
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    state_meta = state_meta.with_new_args(
        goal_achieved=state.GoalAchieved.achieved(),
    )
    assert current_state.meta_info.apply() == state_meta
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is True


    return [env.full_state]

def test() -> list[full_state.FullState]:
    final_states: list[full_state.FullState] = []
    final_states += test_define_scratch()
    return final_states
