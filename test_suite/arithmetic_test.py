# pylint: disable=too-many-lines
from env import core, state, full_state, action, action_impl, meta_env, node_types
from env.goal_env import GoalEnv

def get_current_state(env: GoalEnv):
    return env.full_state.nested_arg(
        (full_state.FullState.idx_current, full_state.HistoryNode.idx_state)
    ).apply().cast(state.State)

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

def get_basic_action_index(node_type: type[action.IBasicAction], env: GoalEnv):
    selected_types = env.full_state.meta.apply().real(
        meta_env.MetaInfo
    ).allowed_basic_actions.apply().cast(meta_env.GeneralTypeGroup)
    meta_idx = selected_types.as_tuple.index(node_type.as_type()) + 1
    return meta_idx

def run_single_eq(left_expr: core.INode, right_expr: core.INode, result: bool):
    goal = node_types.HaveResultScratch.with_goal(core.Eq(left_expr, right_expr))
    env = GoalEnv(
        goal=goal,
        max_steps=3,
        allowed_actions=node_types.ESSENTIAL_ACTIONS,
    )
    assert has_goal(env=env, goal=goal)

    current_state = get_current_state(env)

    state_meta = state.StateMetaInfo.with_goal_achieved(state.GoalAchieved.create())
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    action_idx = get_basic_action_index(action_impl.CreateScratch, env)
    raw_action: action.RawAction = action.RawAction(
        full_state.MetaAllowedBasicActionsTypeIndex(action_idx),
        core.Integer(0),
        core.Integer(0),
        core.Integer(0),
    )
    full_action: action.BaseAction = action_impl.CreateScratch(core.Optional())
    output: meta_env.IActionOutput[full_state.FullState] = action_impl.CreateScratchOutput(
        state.StateScratchIndex(1),
        state.Scratch(),
    )
    env.step(raw_action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    expected_history = full_state.SuccessActionData.from_args(
        raw_action=core.Optional(raw_action),
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    if last_history_action != expected_history:
        print('last_history_action:', env.symbol(last_history_action))
        print('expected_history:', env.symbol(expected_history))
    assert last_history_action == expected_history

    expected_state = state.State.from_raw(
        meta_info=state_meta,
        scratches=[None],
    )
    if current_state != expected_state:
        print('current_state:', env.symbol(current_state))
        print('expected_state:', env.symbol(expected_state))
    assert current_state == expected_state

    assert env.full_state.goal_achieved() is False

    # Run Action
    action_idx = get_basic_action_index(action_impl.DefineScratchFromInt, env)
    scratch_idx = 1
    meta_idx = get_from_int_type_index(state.IntBoolean, env)
    value = 1 if result else 0
    raw_action = action.RawAction(
        full_state.MetaAllowedBasicActionsTypeIndex(action_idx),
        core.Integer(scratch_idx),
        core.Integer(meta_idx),
        core.Integer(value),
    )
    full_action = action_impl.DefineScratchFromInt(
        state.StateScratchIndex(scratch_idx),
        full_state.MetaFromIntTypeIndex(meta_idx),
        core.Integer(value),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(scratch_idx),
        state.Scratch(state.IntBoolean(value)),
    )
    env.step(raw_action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    expected_history = full_state.SuccessActionData.from_args(
        raw_action=core.Optional(raw_action),
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    if last_history_action != expected_history:
        print('last_history_action:', env.symbol(last_history_action))
        print('expected_history:', env.symbol(expected_history))
    assert last_history_action == expected_history

    expected_state = state.State.from_raw(
        meta_info=state_meta,
        scratches=[state.IntBoolean(value)],
    )
    if current_state != expected_state:
        print('current_state:', env.symbol(current_state))
        print('expected_state:', env.symbol(expected_state))
    assert current_state == expected_state

    assert env.full_state.goal_achieved() is False

    # Run Action
    action_idx = get_basic_action_index(action_impl.VerifyGoal, env)
    meta_idx = get_from_int_type_index(state.StateScratchIndex, env)
    scratch_idx = 1
    raw_action = action.RawAction(
        full_state.MetaAllowedBasicActionsTypeIndex(action_idx),
        core.Integer(0),
        core.Integer(meta_idx),
        core.Integer(scratch_idx),
    )
    full_action = action_impl.VerifyGoal(
        core.Optional(),
        full_state.MetaFromIntTypeIndex(meta_idx),
        core.Integer(scratch_idx),
    )
    output = action_impl.VerifyGoalOutput(
        core.Optional(),
        state.StateScratchIndex(scratch_idx),
    )
    env.step(raw_action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    expected_history = full_state.SuccessActionData.from_args(
        raw_action=core.Optional(raw_action),
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    if last_history_action != expected_history:
        print('last_history_action:', env.symbol(last_history_action))
        print('expected_history:', env.symbol(expected_history))
    assert last_history_action == expected_history

    state_meta = state_meta.with_new_args(
        goal_achieved=state.GoalAchieved.achieved(),
    )
    expected_state = state.State.from_raw(
        meta_info=state_meta,
        scratches=[state.IntBoolean(value)],
    )
    if current_state != expected_state:
        print('current_state:', env.symbol(current_state))
        print('expected_state:', env.symbol(expected_state))
    assert current_state == expected_state

    return env.full_state

def run_single(raw_expr: core.INode, correct_expr: core.INode):
    goal = node_types.HaveResultScratch.with_goal(raw_expr.as_node)
    state_meta = state.StateMetaInfo.with_goal_achieved(state.GoalAchieved.create())
    env = GoalEnv(
        goal=goal,
        fn_initial_state=lambda meta: full_state.FullState.with_args(
            meta=meta,
            current=full_state.HistoryNode.with_args(
                state=state.State.from_raw(
                    meta_info=state_meta,
                    scratches=[correct_expr],
                ),
                meta_data=meta_env.MetaData.create(),
            )
        ),
        max_steps=1,
        allowed_actions=node_types.ESSENTIAL_ACTIONS,
    )
    assert has_goal(env=env, goal=goal)

    current_state = get_current_state(env)

    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=[correct_expr],
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    action_idx = get_basic_action_index(action_impl.VerifyGoal, env)
    meta_idx = get_from_int_type_index(state.StateScratchIndex, env)
    scratch_idx = 1
    raw_action = action.RawAction(
        full_state.MetaAllowedBasicActionsTypeIndex(action_idx),
        core.Integer(0),
        core.Integer(meta_idx),
        core.Integer(scratch_idx),
    )
    full_action = action_impl.VerifyGoal(
        core.Optional(),
        full_state.MetaFromIntTypeIndex(meta_idx),
        core.Integer(scratch_idx),
    )
    output = action_impl.VerifyGoalOutput(
        core.Optional(),
        state.StateScratchIndex(scratch_idx),
    )
    env.step(raw_action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    expected_history = full_state.SuccessActionData.from_args(
        raw_action=core.Optional(raw_action),
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    if last_history_action != expected_history:
        print('last_history_action:', env.symbol(last_history_action))
        print('expected_history:', env.symbol(expected_history))
    assert last_history_action == expected_history

    state_meta = state_meta.with_new_args(
        goal_achieved=state.GoalAchieved.achieved(),
    )
    expected_state = state.State.from_raw(
        meta_info=state_meta,
        scratches=[correct_expr],
    )
    if current_state != expected_state:
        print('current_state:', env.symbol(current_state))
        print('expected_state:', env.symbol(expected_state))
    assert current_state == expected_state

    return env.full_state

def run(raw_expr: core.INode, correct_expr: core.INode, wrong_exprs=list[core.INode]):
    final_states: list[full_state.FullState] = []

    for wrong_expr in wrong_exprs:
        final_states.append(run_single_eq(
            left_expr=raw_expr,
            right_expr=wrong_expr,
            result=False,
        ))

    for wrong_expr in wrong_exprs:
        final_states.append(run_single_eq(
            left_expr=wrong_expr,
            right_expr=raw_expr,
            result=False,
        ))

    final_states.append(run_single(
        raw_expr=raw_expr,
        correct_expr=correct_expr,
    ))

    final_states.append(run_single_eq(
        left_expr=raw_expr,
        right_expr=correct_expr,
        result=True,
    ))

    final_states.append(run_single_eq(
        left_expr=correct_expr,
        right_expr=raw_expr,
        result=True,
    ))

    return final_states

def has_goal(env: GoalEnv, goal: meta_env.IGoal):
    selected_goal = env.full_state.nested_arg(
        (full_state.FullState.idx_meta, meta_env.MetaInfo.idx_goal)
    ).apply()
    return selected_goal == goal

def test_arithmetic() -> list[full_state.FullState]:
    final_states: list[full_state.FullState] = []

    final_states += run(
        raw_expr=core.INumber.zero(),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[
            core.INumber.one(),
            core.BinaryInt(core.IBoolean.true(), core.IBoolean.true()),
            core.Integer(0),
            core.DefaultGroup(core.INumber.zero()),
        ],
    )
    final_states += run(
        raw_expr=core.INumber.one(),
        correct_expr=core.INumber.one(),
        wrong_exprs=[
            core.INumber.zero(),
            core.BinaryInt(core.IBoolean.true(), core.IBoolean.false(), core.IBoolean.true()),
            core.Integer(1),
            core.DefaultGroup(core.INumber.one()),
        ],
    )

    final_states += run(
        raw_expr=core.SignedInt(
            core.NegativeSign(core.IBoolean.false()),
            core.INumber.zero(),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[
            core.INumber.one(),
        ],
    )
    final_states += run(
        raw_expr=core.SignedInt(
            core.NegativeSign(core.IBoolean.false()),
            core.INumber.one(),
        ),
        correct_expr=core.INumber.one(),
        wrong_exprs=[
            core.INumber.minus_one(),
        ],
    )
    final_states += run(
        raw_expr=core.SignedInt.minus_one(),
        correct_expr=core.SignedInt.minus_one(),
        wrong_exprs=[
            core.INumber.one(),
        ],
    )
    final_states += run(
        raw_expr=core.SignedInt(
            core.NegativeSign(core.IBoolean.false()),
            core.BinaryInt(
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.false(),
            core.IBoolean.true(),
        ),
        wrong_exprs=[
            core.INumber.minus_one(),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
        ],
    )
    final_states += run(
        raw_expr=core.SignedInt(
            core.NegativeSign.create(),
            core.BinaryInt(
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.SignedInt(
            core.NegativeSign.create(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        wrong_exprs=[
            core.INumber.minus_one(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ],
    )
    final_states += run(
        raw_expr=core.SignedInt(
            core.NegativeSign.create(),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.false(),
        ),
        wrong_exprs=[
            core.INumber.one(),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ],
    )
    final_states += run(
        raw_expr=core.SignedInt(
            core.NegativeSign(core.IBoolean.false()),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.SignedInt(
            core.NegativeSign.create(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
        ),
        wrong_exprs=[
            core.INumber.one(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
        ],
    )
    final_states += run(
        raw_expr=core.SignedInt(
            core.NegativeSign.create(),
            core.SignedInt(
                core.NegativeSign(core.IBoolean.false()),
                core.BinaryInt(
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.SignedInt(
            core.NegativeSign.create(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
        ),
        wrong_exprs=[
            core.INumber.one(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
        ],
    )

    final_states += run(
        raw_expr=core.IntToBinary(core.Integer(0)),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[
            core.INumber.one(),
            core.BinaryInt(core.IBoolean.true(), core.IBoolean.false()),
            core.Integer(0),
            core.Optional(core.INumber.zero()),
        ],
    )
    final_states += run(
        raw_expr=core.IntToBinary(core.Integer(1)),
        correct_expr=core.INumber.one(),
        wrong_exprs=[
            core.INumber.zero(),
            core.BinaryInt(core.IBoolean.true(), core.IBoolean.false(), core.IBoolean.false()),
            core.Integer(1),
            core.Optional(core.INumber.one()),
        ],
    )
    final_states += run(
        raw_expr=core.IntToBinary(core.Integer(2)),
        correct_expr=core.BinaryInt(core.IBoolean.true(), core.IBoolean.false()),
        wrong_exprs=[
            core.INumber.one(),
            core.BinaryInt(core.IBoolean.true(), core.IBoolean.true()),
            core.Integer(2),
            core.Optional(core.BinaryInt(core.IBoolean.true(), core.IBoolean.false())),
        ],
    )
    final_states += run(
        raw_expr=core.IntToBinary(core.Integer(1000)),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.true(),
            core.IBoolean.true(),
            core.IBoolean.true(),
            core.IBoolean.true(),
            core.IBoolean.false(),
            core.IBoolean.true(),
            core.IBoolean.false(),
            core.IBoolean.false(),
            core.IBoolean.false(),
        ),
        wrong_exprs=[
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
            core.Integer(1000),
            core.Optional(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                ),
            ),
        ],
    )

    final_states += run(
        raw_expr=core.BinaryToInt(core.INumber.zero()),
        correct_expr=core.Integer(0),
        wrong_exprs=[
            core.INumber.zero(),
            core.BinaryInt(core.IBoolean.true(), core.IBoolean.false()),
            core.Void(),
        ],
    )
    final_states += run(
        raw_expr=core.BinaryToInt(core.INumber.one()),
        correct_expr=core.Integer(1),
        wrong_exprs=[
            core.INumber.one(),
            core.BinaryInt(core.IBoolean.true(), core.IBoolean.false(), core.IBoolean.false()),
        ],
    )
    final_states += run(
        raw_expr=core.BinaryToInt(core.BinaryInt(core.IBoolean.true(), core.IBoolean.false())),
        correct_expr=core.Integer(2),
        wrong_exprs=[
            core.BinaryInt(core.IBoolean.true(), core.IBoolean.false()),
            core.Integer(1),
        ],
    )
    final_states += run(
        raw_expr=core.BinaryToInt(core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.true(),
            core.IBoolean.true(),
            core.IBoolean.true(),
            core.IBoolean.true(),
            core.IBoolean.false(),
            core.IBoolean.true(),
            core.IBoolean.false(),
            core.IBoolean.false(),
            core.IBoolean.false(),
        )),
        correct_expr=core.Integer(1000),
        wrong_exprs=[
            core.Integer(999),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
        ],
    )

    final_states += run(
        raw_expr=core.Add(
            core.INumber.zero(),
            core.INumber.zero(),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Add(
            core.INumber.one(),
            core.INumber.minus_one(),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Add(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Add(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Add(
            core.INumber.one(),
            core.INumber.zero(),
        ),
        correct_expr=core.INumber.one(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Add(
            core.INumber.zero(),
            core.INumber.one(),
        ),
        correct_expr=core.INumber.one(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Add(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.INumber.zero(),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.false(),
            core.IBoolean.true(),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Add(
            core.INumber.zero(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.false(),
            core.IBoolean.true(),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Add(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
            core.INumber.zero(),
        ),
        correct_expr=core.SignedInt(
            core.NegativeSign.create(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Add(
            core.INumber.zero(),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
        ),
        correct_expr=core.SignedInt(
            core.NegativeSign.create(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Add(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.false(),
            core.IBoolean.false(),
            core.IBoolean.false(),
        ),
        wrong_exprs=[
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.true(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ],
    )
    final_states += run(
        raw_expr=core.Add(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
            core.SignedInt(
                core.NegativeSign(core.IBoolean.false()),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.false(),
            core.IBoolean.true(),
            core.IBoolean.false(),
        ),
        wrong_exprs=[
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
            core.Integer(10),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.true(),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ],
    )
    final_states += run(
        raw_expr=core.Add(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.SignedInt(
            core.NegativeSign.create(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
        ),
        wrong_exprs=[
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
        ],
    )

    final_states += run(
        raw_expr=core.Subtract(
            core.INumber.zero(),
            core.INumber.zero(),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Subtract(
            core.INumber.one(),
            core.INumber.one(),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Subtract(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Subtract(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Subtract(
            core.INumber.one(),
            core.INumber.zero(),
        ),
        correct_expr=core.INumber.one(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Subtract(
            core.INumber.zero(),
            core.INumber.one(),
        ),
        correct_expr=core.INumber.minus_one(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Subtract(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
            core.INumber.zero(),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.true(),
            core.IBoolean.false(),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Subtract(
            core.INumber.zero(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
        ),
        correct_expr=core.SignedInt(
            core.NegativeSign.create(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Subtract(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
            core.INumber.zero(),
        ),
        correct_expr=core.SignedInt(
            core.NegativeSign.create(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Subtract(
            core.INumber.zero(),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.true(),
            core.IBoolean.false(),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Subtract(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.BinaryInt(
                core.IBoolean.false(),
                core.IBoolean.true(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.false(),
        ),
        wrong_exprs=[
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
        ],
    )
    final_states += run(
        raw_expr=core.Subtract(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
            core.SignedInt(
                core.NegativeSign(core.IBoolean.false()),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.SignedInt(
            core.NegativeSign.create(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
        ),
        wrong_exprs=[
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
        ],
    )
    final_states += run(
        raw_expr=core.Subtract(
            core.BinaryInt(
                core.IBoolean.false(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.false(),
            core.IBoolean.true(),
            core.IBoolean.false(),
        ),
        wrong_exprs=[
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ],
    )

    final_states += run(
        raw_expr=core.Multiply(
            core.INumber.zero(),
            core.INumber.zero(),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Multiply(
            core.INumber.zero(),
            core.INumber.one(),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Multiply(
            core.INumber.one(),
            core.INumber.zero(),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Multiply(
            core.INumber.one(),
            core.INumber.one(),
        ),
        correct_expr=core.INumber.one(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Multiply(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.INumber.zero(),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Multiply(
            core.INumber.minus_one(),
            core.INumber.one(),
        ),
        correct_expr=core.INumber.minus_one(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Multiply(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.INumber.one(),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.false(),
            core.IBoolean.true(),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Multiply(
            core.INumber.one(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.false(),
            core.IBoolean.true(),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Multiply(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
            core.INumber.one(),
        ),
        correct_expr=core.SignedInt(
            core.NegativeSign.create(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Multiply(
            core.INumber.one(),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
        ),
        correct_expr=core.SignedInt(
            core.NegativeSign.create(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Multiply(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.INumber.minus_one(),
        ),
        correct_expr=core.SignedInt(
            core.NegativeSign.create(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Multiply(
            core.INumber.minus_one(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.SignedInt(
            core.NegativeSign.create(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Multiply(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
            core.INumber.minus_one(),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.false(),
            core.IBoolean.true(),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Multiply(
            core.INumber.minus_one(),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.false(),
            core.IBoolean.true(),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Multiply(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.true(),
            core.IBoolean.true(),
            core.IBoolean.true(),
        ),
        wrong_exprs=[
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.true(),
            ),
        ],
    )
    final_states += run(
        raw_expr=core.Multiply(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
            core.SignedInt(
                core.NegativeSign(core.IBoolean.false()),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.true(),
            core.IBoolean.false(),
            core.IBoolean.false(),
            core.IBoolean.false(),
        ),
        wrong_exprs=[
            core.SignedInt(
                core.NegativeSign(core.IBoolean.true()),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                ),
            ),
        ],
    )
    final_states += run(
        raw_expr=core.Multiply(
            core.BinaryInt(
                core.IBoolean.false(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.SignedInt(
            core.NegativeSign.create(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
        ),
        wrong_exprs=[
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
        ],
    )
    final_states += run(
        raw_expr=core.Multiply(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                ),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.true(),
            core.IBoolean.false(),
            core.IBoolean.false(),
            core.IBoolean.false(),
        ),
        wrong_exprs=[
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                ),
            ),
        ],
    )

    final_states += run(
        raw_expr=core.DivideInt(
            core.INumber.zero(),
            core.INumber.one(),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.DivideInt(
            core.INumber.zero(),
            core.INumber.minus_one(),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.DivideInt(
            core.INumber.zero(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.DivideInt(
            core.INumber.zero(),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.DivideInt(
            core.INumber.one(),
            core.INumber.one(),
        ),
        correct_expr=core.INumber.one(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.DivideInt(
            core.INumber.one(),
            core.INumber.minus_one(),
        ),
        correct_expr=core.INumber.minus_one(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.DivideInt(
            core.INumber.one(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.DivideInt(
            core.INumber.one(),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.DivideInt(
            core.INumber.minus_one(),
            core.INumber.minus_one(),
        ),
        correct_expr=core.INumber.one(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.DivideInt(
            core.INumber.minus_one(),
            core.INumber.one(),
        ),
        correct_expr=core.INumber.minus_one(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.DivideInt(
            core.INumber.minus_one(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.DivideInt(
            core.INumber.minus_one(),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.DivideInt(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.INumber.one(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.DivideInt(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
        ),
        correct_expr=core.INumber.minus_one(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.DivideInt(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.INumber.minus_one(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.DivideInt(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
        ),
        correct_expr=core.INumber.one(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.DivideInt(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.INumber.one(),
        wrong_exprs=[
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
        ],
    )
    final_states += run(
        raw_expr=core.DivideInt(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
            core.SignedInt(
                core.NegativeSign(core.IBoolean.false()),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[
            core.INumber.one(),
            core.INumber.minus_one(),
        ],
    )
    final_states += run(
        raw_expr=core.DivideInt(
            core.BinaryInt(
                core.IBoolean.false(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[
            core.INumber.one(),
            core.INumber.minus_one(),
        ],
    )
    final_states += run(
        raw_expr=core.DivideInt(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.true(),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.DivideInt(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.SignedInt(
            core.NegativeSign.create(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
            ),
        ),
        wrong_exprs=[
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
            ),
        ],
    )
    final_states += run(
        raw_expr=core.DivideInt(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                ),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.true(),
        ),
        wrong_exprs=[
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
        ],
    )

    final_states += run(
        raw_expr=core.Modulo(
            core.INumber.zero(),
            core.INumber.one(),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Modulo(
            core.INumber.one(),
            core.INumber.one(),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Modulo(
            core.INumber.minus_one(),
            core.INumber.minus_one(),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Modulo(
            core.INumber.one(),
            core.INumber.minus_one(),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Modulo(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Modulo(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Modulo(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.false(),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Modulo(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
            core.SignedInt(
                core.NegativeSign(core.IBoolean.false()),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.false(),
            core.IBoolean.false(),
        ),
        wrong_exprs=[
            core.INumber.zero(),
            core.INumber.one(),
        ],
    )
    final_states += run(
        raw_expr=core.Modulo(
            core.BinaryInt(
                core.IBoolean.false(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.false(),
            core.IBoolean.false(),
        ),
        wrong_exprs=[
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                ),
            ),
        ],
    )
    final_states += run(
        raw_expr=core.Modulo(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                ),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.BinaryInt(
            core.IBoolean.true(),
            core.IBoolean.false(),
            core.IBoolean.false(),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Modulo(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Modulo(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Modulo(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                ),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.INumber.zero(),
        wrong_exprs=[],
    )

    final_states += run(
        raw_expr=core.GreaterThan(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.IBoolean.true(),
        wrong_exprs=[
            core.IBoolean.false(),
        ],
    )
    final_states += run(
        raw_expr=core.GreaterThan(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.IBoolean.false(),
        wrong_exprs=[
            core.IBoolean.true(),
        ],
    )
    final_states += run(
        raw_expr=core.GreaterThan(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
        ),
        correct_expr=core.IBoolean.false(),
        wrong_exprs=[
            core.IBoolean.true(),
        ],
    )

    final_states += run(
        raw_expr=core.LessThan(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.IBoolean.true(),
        wrong_exprs=[
            core.IBoolean.false(),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
            ),
        ],
    )
    final_states += run(
        raw_expr=core.LessThan(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
        ),
        correct_expr=core.IBoolean.false(),
        wrong_exprs=[core.IBoolean.true()],
    )
    final_states += run(
        raw_expr=core.LessThan(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.IBoolean.true(),
        wrong_exprs=[core.IBoolean.false()],
    )
    final_states += run(
        raw_expr=core.LessThan(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
        ),
        correct_expr=core.IBoolean.false(),
        wrong_exprs=[core.IBoolean.true()],
    )

    final_states += run(
        raw_expr=core.GreaterOrEqual(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.IBoolean.true(),
        wrong_exprs=[
            core.IBoolean.false(),
        ],
    )
    final_states += run(
        raw_expr=core.GreaterOrEqual(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
        ),
        correct_expr=core.IBoolean.true(),
        wrong_exprs=[
            core.IBoolean.false(),
        ],
    )
    final_states += run(
        raw_expr=core.GreaterOrEqual(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
        ),
        correct_expr=core.IBoolean.false(),
        wrong_exprs=[
            core.IBoolean.true(),
        ],
    )
    final_states += run(
        raw_expr=core.GreaterOrEqual(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                ),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
        ),
        correct_expr=core.IBoolean.true(),
        wrong_exprs=[
            core.IBoolean.false(),
        ],
    )

    final_states += run(
        raw_expr=core.LessOrEqual(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.IBoolean.true(),
        wrong_exprs=[
            core.IBoolean.false(),
        ],
    )
    final_states += run(
        raw_expr=core.LessOrEqual(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.IBoolean.true(),
        wrong_exprs=[
            core.IBoolean.false(),
        ],
    )
    final_states += run(
        raw_expr=core.LessOrEqual(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.IBoolean.true(),
        wrong_exprs=[
            core.IBoolean.false(),
        ],
    )
    final_states += run(
        raw_expr=core.LessOrEqual(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
        ),
        correct_expr=core.IBoolean.false(),
        wrong_exprs=[
            core.IBoolean.true(),
        ],
    )

    final_states += run(
        raw_expr=core.Float(
            core.BinaryInt.zero(),
            core.BinaryInt.zero(),
        ),
        correct_expr=core.Float(
            core.BinaryInt.zero(),
            core.BinaryInt.zero(),
        ),
        wrong_exprs=[
            core.INumber.zero(),
            core.Integer(0),
        ],
    )
    final_states += run(
        raw_expr=core.Float(
            core.BinaryInt.zero(),
            core.BinaryInt.one(),
        ),
        correct_expr=core.Float(
            core.BinaryInt.zero(),
            core.BinaryInt.zero(),
        ),
        wrong_exprs=[
            core.INumber.zero(),
            core.INumber.one(),
        ],
    )
    final_states += run(
        raw_expr=core.Float(
            core.BinaryInt.one(),
            core.BinaryInt.one(),
        ),
        correct_expr=core.Float(
            core.BinaryInt.one(),
            core.BinaryInt.one(),
        ),
        wrong_exprs=[
            core.INumber.one(),
            core.INumber.zero(),
        ],
    )
    final_states += run(
        raw_expr=core.Float(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.BinaryInt(
                core.IBoolean.false(),
                core.IBoolean.true(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.Float(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
            ),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Float(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
            core.BinaryInt(
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
        ),
        correct_expr=core.Float(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.Float(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.Float(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        wrong_exprs=[],
    )

    final_states += run(
        raw_expr=core.AsFloat(
            core.INumber.zero(),
        ),
        correct_expr=core.Float(
            core.BinaryInt.zero(),
            core.BinaryInt.zero(),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.AsFloat(
            core.INumber.one(),
        ),
        correct_expr=core.Float(
            core.BinaryInt.one(),
            core.BinaryInt.one(),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.AsFloat(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        correct_expr=core.Float(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.true(),
            ),
        ),
        wrong_exprs=[],
    )
    final_states += run(
        raw_expr=core.AsFloat(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
        ),
        correct_expr=core.Float(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
            ),
        ),
        wrong_exprs=[],
    )

    final_states += run(
        raw_expr=core.Add(
            core.Float(
                core.INumber.zero(),
                core.INumber.one(),
            ),
            core.Float(
                core.INumber.zero(),
                core.INumber.one(),
            ),
        ),
        correct_expr=core.Float(
            core.INumber.zero(),
            core.INumber.zero(),
        ),
        wrong_exprs=[],
    )
    # [(5/2^3) * 2^3] + [(2/2^2) * 2^3] = 5 + 4 = 9
    final_states += run(
        raw_expr=core.Add(
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
        ),
        correct_expr=core.Float(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
        ),
        wrong_exprs=[],
    )
    # [(5/2^3) * 2^3] + [-(2/2^2) * 2^3] = 5 + -4 = 1
    final_states += run(
        raw_expr=core.Add(
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
            core.Float(
                core.SignedInt(
                    core.NegativeSign.create(),
                    core.BinaryInt(
                        core.IBoolean.true(),
                        core.IBoolean.false(),
                    ),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
        ),
        correct_expr=core.Float(
            core.INumber.one(),
            core.INumber.one(),
        ),
        wrong_exprs=[],
    )
    # [-(5/2^3) * 2^3] + [(2/2^2) * 2^3] = -5 + 4 = -1
    final_states += run(
        raw_expr=core.Add(
            core.Float(
                core.SignedInt(
                    core.NegativeSign.create(),
                    core.BinaryInt(
                        core.IBoolean.true(),
                        core.IBoolean.false(),
                        core.IBoolean.true(),
                    ),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
        ),
        correct_expr=core.Float(
            core.INumber.minus_one(),
            core.INumber.one(),
        ),
        wrong_exprs=[],
    )
    # [-(5/2^3) * 2^3] + [-(2/2^2) * 2^3] = -5 + -4 = -9
    final_states += run(
        raw_expr=core.Add(
            core.Float(
                core.SignedInt(
                    core.NegativeSign.create(),
                    core.BinaryInt(
                        core.IBoolean.true(),
                        core.IBoolean.false(),
                        core.IBoolean.true(),
                    ),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
            core.Float(
                core.SignedInt(
                    core.NegativeSign.create(),
                    core.BinaryInt(
                        core.IBoolean.true(),
                        core.IBoolean.false(),
                    ),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
        ),
        correct_expr=core.Float(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
        ),
        wrong_exprs=[],
    )
    # [(5/2^3) * 2^3] + [(2/2^2) * 2^6] = 5 + 32 = 37
    final_states += run(
        raw_expr=core.Add(
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.Float(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
        ),
        wrong_exprs=[],
    )
    # [(5/2^3) * 2^2] + [(2/2^2) * 2^-2] = 5/2 + 1/8 = 21/8 = [(21*2^2/2^7)*2^2]
    final_states += run(
        raw_expr=core.Add(
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
                core.SignedInt(
                    core.NegativeSign.create(),
                    core.BinaryInt(
                        core.IBoolean.true(),
                        core.IBoolean.false(),
                    ),
                ),
            ),
        ),
        correct_expr=core.Float(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
        ),
        wrong_exprs=[],
    )

    final_states += run(
        raw_expr=core.Subtract(
            core.Float(
                core.INumber.zero(),
                core.INumber.one(),
            ),
            core.Float(
                core.INumber.zero(),
                core.INumber.one(),
            ),
        ),
        correct_expr=core.Float(
            core.INumber.zero(),
            core.INumber.zero(),
        ),
        wrong_exprs=[],
    )
    # [(5/2^3) * 2^3] - [(2/2^2) * 2^3] = 5 - 4 = 1
    final_states += run(
        raw_expr=core.Subtract(
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
        ),
        correct_expr=core.Float(
            core.INumber.one(),
            core.INumber.one(),
        ),
        wrong_exprs=[],
    )
    # [(5/2^3) * 2^3] - [-(2/2^2) * 2^3] = 5 - -4 = 9
    final_states += run(
        raw_expr=core.Subtract(
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
            core.Float(
                core.SignedInt(
                    core.NegativeSign.create(),
                    core.BinaryInt(
                        core.IBoolean.true(),
                        core.IBoolean.false(),
                    ),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
        ),
        correct_expr=core.Float(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
        ),
        wrong_exprs=[],
    )
    # [-(5/2^3) * 2^3] - [(2/2^2) * 2^3] = -5 - 4 = -9
    final_states += run(
        raw_expr=core.Subtract(
            core.Float(
                core.SignedInt(
                    core.NegativeSign.create(),
                    core.BinaryInt(
                        core.IBoolean.true(),
                        core.IBoolean.false(),
                        core.IBoolean.true(),
                    ),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
        ),
        correct_expr=core.Float(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
        ),
        wrong_exprs=[],
    )
    # [-(5/2^3) * 2^3] - [-(2/2^2) * 2^3] = -5 - -4 = -1
    final_states += run(
        raw_expr=core.Subtract(
            core.Float(
                core.SignedInt(
                    core.NegativeSign.create(),
                    core.BinaryInt(
                        core.IBoolean.true(),
                        core.IBoolean.false(),
                        core.IBoolean.true(),
                    ),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
            core.Float(
                core.SignedInt(
                    core.NegativeSign.create(),
                    core.BinaryInt(
                        core.IBoolean.true(),
                        core.IBoolean.false(),
                    ),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
        ),
        correct_expr=core.Float(
            core.INumber.minus_one(),
            core.INumber.one(),
        ),
        wrong_exprs=[],
    )
    # [(5/2^3) * 2^3] - [(2/2^2) * 2^6] = 5 - 32 = -27
    final_states += run(
        raw_expr=core.Subtract(
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.Float(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
        ),
        wrong_exprs=[],
    )
    # [(5/2^3) * 2^2] - [(2/2^2) * 2^-2] = 5/2 - 1/8 = 19/8 = [(19*2^2/2^7)*2^2]
    final_states += run(
        raw_expr=core.Subtract(
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
                core.SignedInt(
                    core.NegativeSign.create(),
                    core.BinaryInt(
                        core.IBoolean.true(),
                        core.IBoolean.false(),
                    ),
                ),
            ),
        ),
        correct_expr=core.Float(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
        ),
        wrong_exprs=[],
    )

    final_states += run(
        raw_expr=core.Multiply(
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
            core.AsFloat(
                core.INumber.zero(),
            ),
        ),
        correct_expr=core.Float(
            core.BinaryInt.zero(),
            core.BinaryInt.zero(),
        ),
        wrong_exprs=[],
    )
    # [(5/2^3) * 2^3] * [(2/2^2) * 2^6] = 5 * 32 = 160 = [(10/2^4)*2^8]
    final_states += run(
        raw_expr=core.Multiply(
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                ),
            ),
        ),
        correct_expr=core.Float(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
                core.IBoolean.false(),
            ),
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.false(),
            ),
        ),
        wrong_exprs=[],
    )
    # [(7/2^3) * 2] * [(13/2^4) * 2^-1] = 7/2^2 * 13/2^5 = 91/2^7 = [(91/2^7)*2^0]
    final_states += run(
        raw_expr=core.Multiply(
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
                core.INumber.one(),
            ),
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
                core.INumber.minus_one(),
            ),
        ),
        correct_expr=core.Float(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
                core.IBoolean.true(),
            ),
            core.INumber.zero(),
        ),
        wrong_exprs=[],
    )
    # [-(7/2^3) * 2] * [(13/2^4) * 2^-2] = 7/2^2 * 13/2^6 = -91/2^8 = [-(91/2^7)*2^-1]
    final_states += run(
        raw_expr=core.Multiply(
            core.Float(
                core.SignedInt(
                    core.NegativeSign.create(),
                    core.BinaryInt(
                        core.IBoolean.true(),
                        core.IBoolean.true(),
                        core.IBoolean.true(),
                    ),
                ),
                core.INumber.one(),
            ),
            core.Float(
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
                core.SignedInt(
                    core.NegativeSign.create(),
                    core.BinaryInt(
                        core.IBoolean.true(),
                        core.IBoolean.false(),
                    ),
                ),
            ),
        ),
        correct_expr=core.Float(
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                    core.IBoolean.true(),
                ),
            ),
            core.INumber.minus_one(),
        ),
        wrong_exprs=[],
    )
    # [-(7/2^3) * 2^-3] * [-(15/2^4) * 2^-2] = -7/2^6 * -15/2^6 = 105/2^12 = [(105/2^7)*2^-5]
    final_states += run(
        raw_expr=core.Multiply(
            core.Float(
                core.SignedInt(
                    core.NegativeSign.create(),
                    core.BinaryInt(
                        core.IBoolean.true(),
                        core.IBoolean.true(),
                        core.IBoolean.true(),
                    ),
                ),
                core.SignedInt(
                    core.NegativeSign.create(),
                    core.BinaryInt(
                        core.IBoolean.true(),
                        core.IBoolean.true(),
                    ),
                ),
            ),
            core.Float(
                core.SignedInt(
                    core.NegativeSign.create(),
                    core.BinaryInt(
                        core.IBoolean.true(),
                        core.IBoolean.true(),
                        core.IBoolean.true(),
                        core.IBoolean.true(),
                    ),
                ),
                core.SignedInt(
                    core.NegativeSign.create(),
                    core.BinaryInt(
                        core.IBoolean.true(),
                        core.IBoolean.false(),
                    ),
                ),
            ),
        ),
        correct_expr=core.Float(
            core.BinaryInt(
                core.IBoolean.true(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.true(),
                core.IBoolean.false(),
                core.IBoolean.false(),
                core.IBoolean.true(),
            ),
            core.SignedInt(
                core.NegativeSign.create(),
                core.BinaryInt(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
        ),
        wrong_exprs=[],
    )

    #TODO: Add tests for Float (Division, Comparisons)

    return final_states

def test() -> list[full_state.FullState]:
    final_states: list[full_state.FullState] = []
    final_states += test_arithmetic()
    return final_states
