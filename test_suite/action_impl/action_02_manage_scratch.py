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

def test_manage_scratch() -> list[full_state.FullState]:
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
    )
    assert last_history_action == full_state.ActionData.from_args(
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    meta_idx = get_default_type_index(core.Void, env)
    raw_action = action_impl.DefineScratchFromDefault.from_raw(2, meta_idx, 0)
    full_action = action_impl.DefineScratchFromDefault(
        state.StateScratchIndex(2),
        full_state.MetaDefaultTypeIndex(meta_idx),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(2),
        state.Scratch(core.Void()),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches[1] = core.Void()
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
    raw_action = action_impl.DeleteScratchOutput.from_raw(1, 0, 0)
    full_action = action_impl.DeleteScratchOutput(state.StateScratchIndex(1))
    output = action_impl.DeleteScratchOutput(state.StateScratchIndex(1))
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches.pop(0)
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
    raw_action = action_impl.CreateScratch.from_raw(1, 0, 0)
    full_action = action_impl.CreateScratch(
        core.Optional(state.StateScratchIndex(1))
    )
    output = action_impl.CreateScratchOutput(
        state.StateScratchIndex(3),
        state.Scratch(core.Void()),
    )
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches.append(core.Void())
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
    raw_action = action_impl.DeleteScratchOutput.from_raw(3, 0, 0)
    full_action = action_impl.DeleteScratchOutput(state.StateScratchIndex(3))
    output = action_impl.DeleteScratchOutput(state.StateScratchIndex(3))
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches.pop(-1)
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
    raw_action = action_impl.DeleteScratchOutput.from_raw(2, 0, 0)
    full_action = action_impl.DeleteScratchOutput(state.StateScratchIndex(2))
    output = action_impl.DeleteScratchOutput(state.StateScratchIndex(2))
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches.pop(-1)
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
    final_states += test_manage_scratch()
    return final_states
