# pylint: disable=too-many-lines
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

def get_from_int_type_index(node_type: type[core.INode], env: GoalEnv):
    selected_types = env.full_state.meta.apply().nested_args((
        meta_env.MetaInfo.idx_from_int_group,
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

def test_manage_args_group() -> list[full_state.FullState]:
    def has_goal(env: GoalEnv, goal: meta_env.IGoal):
        selected_goal = env.full_state.nested_args(
            (full_state.FullState.idx_meta, meta_env.MetaInfo.idx_goal)
        ).apply()
        return selected_goal == goal

    goal = node_types.HaveScratch.with_goal(core.And(
        core.LessThan(core.Var.from_int(2), core.Var.from_int(1)),
        core.GreaterThan(core.Var.from_int(3), core.Var.from_int(1)),
        core.Or(
            core.Param.from_int(1),
            core.GreaterThan(core.Var.from_int(2), core.Var.from_int(1)),
            core.LessThan(core.Var.from_int(3), core.Var.from_int(1)),
        )
    ))
    env = GoalEnv(goal=goal)
    assert has_goal(env=env, goal=goal)
    env.full_state.validate()

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
        state.Scratch.create(),
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
    raw_action = action_impl.CreateArgsGroup.from_raw(2, 0, 0)
    full_action = action_impl.CreateArgsGroup(
        core.Integer(2),
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
    args_groups = [args_group]
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
    meta_idx = get_from_int_type_index(core.Var, env)
    raw_action = action_impl.DefineScratchFromInt.from_raw(1, meta_idx, 1)
    full_action = action_impl.DefineScratchFromInt(
        state.StateScratchIndex(1),
        full_state.MetaFromIntTypeIndex(meta_idx),
        core.Integer(1),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        state.Scratch(core.Var.from_int(1)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = core.Var.from_int(1)
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
        state.Scratch(core.Var.from_int(1)),
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
        core.Optional(),
        core.Optional(core.Var.from_int(1)),
    )
    args_groups = [args_group]
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
    raw_action = action_impl.CreateArgsGroup.from_raw(2, 1, 0)
    full_action = action_impl.CreateArgsGroup(
        core.Integer(2),
        core.Optional(state.StateArgsGroupIndex(1)),
    )
    output = action_impl.CreateArgsGroupOutput(
        state.StateArgsGroupIndex(2),
        args_group,
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    args_groups.append(args_group)
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
    meta_idx = get_from_int_type_index(core.Var, env)
    raw_action = action_impl.DefineScratchFromInt.from_raw(1, meta_idx, 2)
    full_action = action_impl.DefineScratchFromInt(
        state.StateScratchIndex(1),
        full_state.MetaFromIntTypeIndex(meta_idx),
        core.Integer(2),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        state.Scratch(core.Var.from_int(2)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = core.Var.from_int(2)
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
        state.Scratch(core.Var.from_int(2)),
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
        core.Optional(core.Var.from_int(2)),
        core.Optional(core.Var.from_int(1)),
    )
    args_groups[0] = args_group
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
    meta_idx = get_from_int_type_index(core.Var, env)
    raw_action = action_impl.DefineScratchFromInt.from_raw(1, meta_idx, 3)
    full_action = action_impl.DefineScratchFromInt(
        state.StateScratchIndex(1),
        full_state.MetaFromIntTypeIndex(meta_idx),
        core.Integer(3),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        state.Scratch(core.Var.from_int(3)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = core.Var.from_int(3)
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
    raw_action = action_impl.DefineArgsGroup.from_raw(2, 1, 1)
    full_action = action_impl.DefineArgsGroup(
        state.StateArgsGroupIndex(2),
        core.NodeArgIndex(1),
        state.StateScratchIndex(1),
    )
    output = action_impl.DefineArgsGroupArgOutput(
        state.StateArgsGroupIndex(2),
        core.NodeArgIndex(1),
        state.Scratch(core.Var.from_int(3)),
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
        core.Optional(core.Var.from_int(3)),
        core.Optional(core.Var.from_int(1)),
    )
    args_groups[1] = args_group
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
    raw_action = action_impl.CreateArgsGroup.from_raw(3, 1, 0)
    full_action = action_impl.CreateArgsGroup(
        core.Integer(3),
        core.Optional(state.StateArgsGroupIndex(1)),
    )
    args_group = state.PartialArgsGroup(
        core.Optional(core.Var.from_int(2)),
        core.Optional(core.Var.from_int(1)),
        core.Optional(),
    )
    output = action_impl.CreateArgsGroupOutput(
        state.StateArgsGroupIndex(3),
        args_group,
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    args_groups.append(args_group)
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
    meta_idx = get_from_int_type_index(core.Param, env)
    raw_action = action_impl.DefineScratchFromInt.from_raw(1, meta_idx, 1)
    full_action = action_impl.DefineScratchFromInt(
        state.StateScratchIndex(1),
        full_state.MetaFromIntTypeIndex(meta_idx),
        core.Integer(1),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        state.Scratch(core.Param.from_int(1)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = core.Param.from_int(1)
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
    raw_action = action_impl.DefineArgsGroup.from_raw(3, 1, 1)
    full_action = action_impl.DefineArgsGroup(
        state.StateArgsGroupIndex(3),
        core.NodeArgIndex(1),
        state.StateScratchIndex(1),
    )
    output = action_impl.DefineArgsGroupArgOutput(
        state.StateArgsGroupIndex(3),
        core.NodeArgIndex(1),
        state.Scratch(core.Param.from_int(1)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    or_params: list[core.Optional] = [
        core.Optional(core.Param.from_int(1)),
        core.Optional(core.Var.from_int(1)),
        core.Optional(),
    ]
    args_group = state.PartialArgsGroup(*or_params)
    args_groups[2] = args_group
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
    idx_node = env.full_state.node_types().index(core.LessThan) + 1
    raw_action = action_impl.DefineScratchFromIntIndex.from_raw(1, meta_idx, idx_node)
    full_action = action_impl.DefineScratchFromIntIndex(
        state.StateScratchIndex(1),
        full_state.MetaFullStateIntIndexTypeIndex(meta_idx),
        core.Integer(idx_node),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        state.Scratch(core.TypeNode(core.LessThan)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[0] = core.TypeNode(core.LessThan)
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
    raw_action = action_impl.CreateScratch.from_raw(0, 0, 0)
    full_action = action_impl.CreateScratch.create()
    output = action_impl.CreateScratchOutput(
        state.StateScratchIndex(2),
        state.Scratch.create(),
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
    idx_node = env.full_state.node_types().index(core.GreaterThan) + 1
    raw_action = action_impl.DefineScratchFromIntIndex.from_raw(2, meta_idx, idx_node)
    full_action = action_impl.DefineScratchFromIntIndex(
        state.StateScratchIndex(2),
        full_state.MetaFullStateIntIndexTypeIndex(meta_idx),
        core.Integer(idx_node),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(2),
        state.Scratch(core.TypeNode(core.GreaterThan)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[1] = core.TypeNode(core.GreaterThan)
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
    raw_action = action_impl.CreateScratch.from_raw(0, 0, 0)
    full_action = action_impl.CreateScratch.create()
    output = action_impl.CreateScratchOutput(
        state.StateScratchIndex(3),
        state.Scratch.create(),
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
        args_groups=args_groups,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    raw_action = action_impl.DefineScratchFromFunctionWithArgs.from_raw(3, 2, 1)
    full_action = action_impl.DefineScratchFromFunctionWithArgs(
        state.StateScratchIndex(3),
        state.StateScratchIndex(2),
        core.Optional(state.StateArgsGroupIndex(1)),
    )
    fn_call: core.INode = core.GreaterThan(core.Var.from_int(2), core.Var.from_int(1))
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(3),
        state.Scratch(fn_call),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[2] = fn_call
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
    raw_action = action_impl.DefineArgsGroup.from_raw(3, 2, 3)
    full_action = action_impl.DefineArgsGroup(
        state.StateArgsGroupIndex(3),
        core.NodeArgIndex(2),
        state.StateScratchIndex(3),
    )
    output = action_impl.DefineArgsGroupArgOutput(
        state.StateArgsGroupIndex(3),
        core.NodeArgIndex(2),
        state.Scratch(fn_call),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    or_params[1] = core.Optional(fn_call)
    args_group = state.PartialArgsGroup(*or_params)
    args_groups[2] = args_group
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
    raw_action = action_impl.DefineScratchFromFunctionWithArgs.from_raw(3, 1, 2)
    full_action = action_impl.DefineScratchFromFunctionWithArgs(
        state.StateScratchIndex(3),
        state.StateScratchIndex(1),
        core.Optional(state.StateArgsGroupIndex(2)),
    )
    fn_call = core.LessThan(core.Var.from_int(3), core.Var.from_int(1))
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(3),
        state.Scratch(fn_call),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[2] = fn_call
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
    raw_action = action_impl.DefineArgsGroup.from_raw(3, 3, 3)
    full_action = action_impl.DefineArgsGroup(
        state.StateArgsGroupIndex(3),
        core.NodeArgIndex(3),
        state.StateScratchIndex(3),
    )
    output = action_impl.DefineArgsGroupArgOutput(
        state.StateArgsGroupIndex(3),
        core.NodeArgIndex(3),
        state.Scratch(fn_call),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    or_params[2] = core.Optional(fn_call)
    args_group = state.PartialArgsGroup(*or_params)
    args_groups[2] = args_group
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
    idx_node = env.full_state.node_types().index(core.Or) + 1
    raw_action = action_impl.DefineScratchFromIntIndex.from_raw(3, meta_idx, idx_node)
    full_action = action_impl.DefineScratchFromIntIndex(
        state.StateScratchIndex(3),
        full_state.MetaFullStateIntIndexTypeIndex(meta_idx),
        core.Integer(idx_node),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(3),
        state.Scratch(core.TypeNode(core.Or)),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[2] = core.TypeNode(core.Or)
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
    raw_action = action_impl.DefineScratchFromFunctionWithArgs.from_raw(3, 3, 3)
    full_action = action_impl.DefineScratchFromFunctionWithArgs(
        state.StateScratchIndex(3),
        state.StateScratchIndex(3),
        core.Optional(state.StateArgsGroupIndex(3)),
    )
    or_call = core.Or(*[p.value_or_raise for p in or_params])
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(3),
        state.Scratch(or_call),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[2] = or_call
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
    raw_action = action_impl.DefineArgsGroup.from_raw(3, 3, 3)
    full_action = action_impl.DefineArgsGroup(
        state.StateArgsGroupIndex(3),
        core.NodeArgIndex(3),
        state.StateScratchIndex(3),
    )
    output = action_impl.DefineArgsGroupArgOutput(
        state.StateArgsGroupIndex(3),
        core.NodeArgIndex(3),
        state.Scratch(or_call),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    and_params = [p for p in or_params]
    and_params[2] = core.Optional(or_call)
    args_group = state.PartialArgsGroup(*and_params)
    args_groups[2] = args_group
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
    raw_action = action_impl.DefineScratchFromFunctionWithArgs.from_raw(3, 1, 1)
    full_action = action_impl.DefineScratchFromFunctionWithArgs(
        state.StateScratchIndex(3),
        state.StateScratchIndex(1),
        core.Optional(state.StateArgsGroupIndex(1)),
    )
    p1_call = core.LessThan(core.Var.from_int(2), core.Var.from_int(1))
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(3),
        state.Scratch(p1_call),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[2] = p1_call
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
    raw_action = action_impl.DefineArgsGroup.from_raw(3, 1, 3)
    full_action = action_impl.DefineArgsGroup(
        state.StateArgsGroupIndex(3),
        core.NodeArgIndex(1),
        state.StateScratchIndex(3),
    )
    output = action_impl.DefineArgsGroupArgOutput(
        state.StateArgsGroupIndex(3),
        core.NodeArgIndex(1),
        state.Scratch(p1_call),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    and_params[0] = core.Optional(p1_call)
    args_group = state.PartialArgsGroup(*and_params)
    args_groups[2] = args_group
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
    raw_action = action_impl.DefineScratchFromFunctionWithArgs.from_raw(3, 2, 2)
    full_action = action_impl.DefineScratchFromFunctionWithArgs(
        state.StateScratchIndex(3),
        state.StateScratchIndex(2),
        core.Optional(state.StateArgsGroupIndex(2)),
    )
    p2_call = core.GreaterThan(core.Var.from_int(3), core.Var.from_int(1))
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(3),
        state.Scratch(p2_call),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[2] = p2_call
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
    raw_action = action_impl.DefineArgsGroup.from_raw(3, 2, 3)
    full_action = action_impl.DefineArgsGroup(
        state.StateArgsGroupIndex(3),
        core.NodeArgIndex(2),
        state.StateScratchIndex(3),
    )
    output = action_impl.DefineArgsGroupArgOutput(
        state.StateArgsGroupIndex(3),
        core.NodeArgIndex(2),
        state.Scratch(p2_call),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    and_params[1] = core.Optional(p2_call)
    args_group = state.PartialArgsGroup(*and_params)
    args_groups[2] = args_group
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
    idx_node = env.full_state.node_types().index(core.And) + 1
    raw_action = action_impl.DefineScratchFromIntIndex.from_raw(1, meta_idx, idx_node)
    full_action = action_impl.DefineScratchFromIntIndex(
        state.StateScratchIndex(1),
        full_state.MetaFullStateIntIndexTypeIndex(meta_idx),
        core.Integer(idx_node),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(1),
        state.Scratch(core.TypeNode(core.And)),
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
        args_groups=args_groups,
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    raw_action = action_impl.DefineScratchFromFunctionWithArgs.from_raw(2, 1, 3)
    full_action = action_impl.DefineScratchFromFunctionWithArgs(
        state.StateScratchIndex(2),
        state.StateScratchIndex(1),
        core.Optional(state.StateArgsGroupIndex(3)),
    )
    and_call = core.And(*[p.value_or_raise for p in and_params])
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(2),
        state.Scratch(and_call),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[1] = and_call
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
    raw_action = action_impl.CreateArgsGroup.from_raw(1, 3, 0)
    full_action = action_impl.CreateArgsGroup(
        core.Integer(1),
        core.Optional(state.StateArgsGroupIndex(3)),
    )
    args_group = state.PartialArgsGroup(
        core.Optional(
            core.LessThan(core.Var.from_int(2), core.Var.from_int(1))
        ),
    )
    output = action_impl.CreateArgsGroupOutput(
        state.StateArgsGroupIndex(4),
        args_group,
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    args_groups.append(args_group)
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
    raw_action = action_impl.CreateArgsGroup.from_raw(0, 3, 0)
    full_action = action_impl.CreateArgsGroup(
        core.Integer(0),
        core.Optional(state.StateArgsGroupIndex(3)),
    )
    args_group = state.PartialArgsGroup()
    output = action_impl.CreateArgsGroupOutput(
        state.StateArgsGroupIndex(5),
        args_group,
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    args_groups.append(args_group)
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
    raw_action = action_impl.DeleteArgsGroupOutput.from_raw(5, 0, 0)
    full_action = action_impl.DeleteArgsGroupOutput(
        state.StateArgsGroupIndex(5),
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
    args_groups = args_groups[:-1]
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

    for _ in range(4):
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
        args_groups = args_groups[1:]
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
    meta_idx = get_from_int_type_index(state.StateScratchIndex, env)
    raw_action = action_impl.VerifyGoal.from_raw(0, meta_idx, 2)
    full_action = action_impl.VerifyGoal(
        core.Optional(),
        full_state.MetaFromIntTypeIndex(meta_idx),
        core.Integer(2),
    )
    output = action_impl.VerifyGoalOutput(
        core.Optional(),
        state.StateScratchIndex(2),
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

    assert get_remaining_steps(env) is None

    return [env.full_state]

def test() -> list[full_state.FullState]:
    final_states: list[full_state.FullState] = []
    final_states += test_manage_args_group()
    return final_states
