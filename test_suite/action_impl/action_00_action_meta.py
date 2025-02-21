from env.core import (
    DefaultGroup,
    Param,
    And,
    Or,
    Integer,
    IntBoolean,
    BooleanExceptionInfo,
    IsEmpty,
    NodeArgIndex,
    NodeArgReverseIndex,
    InstanceType,
    IOptional,
    Add,
    IInt,
)
from env.full_state import (
    FullState,
    MetaAllTypesTypeIndex,
    HistoryGroupNode,
    HistoryNode,
    BaseActionData,
    ActionErrorActionData,
    SuccessActionData,
    MetaFromIntTypeIndex,
)
from env.meta_env import MetaInfo, SubtypeOuterGroup, GeneralTypeGroup, MetaData
from env.goal_env import GoalEnv
from env.node_types import HaveScratch, INode
from env.state import (
    State,
    Scratch,
    StateMetaInfo,
    GoalAchieved,
    PartialArgsGroup,
    StateArgsGroupIndex,
    StateScratchIndex,
)
from env.action_impl import (
    DynamicAction,
    DynamicActionOutput,
    GroupAction,
    GroupActionOutput,
    VerifyGoalOutput,
    DefineScratchFromInt,
    DefineScratchOutput,
    DefineScratchFromIntIndex,
    CreateArgsGroupOutput,
    CreateScratchOutput,
    VerifyGoal,
    CreateScratch,
    CreateArgsGroup,
    DefineArgsGroup,
    DefineArgsGroupArgOutput,
    DefineScratchFromFunctionWithArgs,
    DeleteArgsGroupOutput,
    DeleteScratchOutput,
    DefineScratchFromSingleArg,
    RestoreHistoryStateOutput,
)
from env.action import IAction, BaseAction, IActionOutput
from env.core import Optional

def get_current_state(env: GoalEnv):
    return env.full_state.nested_arg(
        (FullState.idx_current, HistoryNode.idx_state)
    ).apply().cast(State)

def get_last_history_action(env: GoalEnv):
    history = env.full_state.history.apply().cast(HistoryGroupNode)
    last = history.as_tuple[-1]
    return last.action_data.apply().inner_arg(Optional.idx_value).apply()

def get_from_int_type_index(node_type: type[INode], meta: MetaInfo):
    selected_types = meta.nested_arg((
        MetaInfo.idx_from_int_group,
        SubtypeOuterGroup.idx_subtypes,
    )).apply().cast(GeneralTypeGroup)
    meta_idx = selected_types.as_tuple.index(node_type.as_type()) + 1
    return meta_idx

def get_info_type_index(
    node_type: type[INode],
    meta: MetaInfo,
    node_types: tuple[type[INode], ...],
):
    index_type_idx = node_types.index(MetaAllTypesTypeIndex) + 1
    type_node = node_types[index_type_idx-1].as_type()
    selected_types = meta.nested_arg((
        MetaInfo.idx_full_state_int_index_group,
        SubtypeOuterGroup.idx_subtypes,
    )).apply().cast(GeneralTypeGroup)
    meta_idx = selected_types.as_tuple.index(type_node) + 1
    node_idx = node_types.index(node_type) + 1
    return meta_idx, node_idx

def get_empty_exception(action: IAction):
    return ActionErrorActionData.from_args(
        raw_action=Optional(),
        action=Optional(action),
        output=Optional(),
        exception=Optional(BooleanExceptionInfo(IsEmpty(Optional()))),
    )

def get_single_child_type_index(node_type: type[INode], meta: MetaInfo):
    selected_types = meta.nested_arg((
        MetaInfo.idx_single_child_group,
        SubtypeOuterGroup.idx_subtypes,
    )).apply().cast(GeneralTypeGroup)
    meta_idx = selected_types.as_tuple.index(node_type.as_type()) + 1
    return meta_idx

def get_remaining_steps(env: GoalEnv) -> int | None:
    value = env.full_state.nested_arg(
        (
            FullState.idx_current,
            HistoryNode.idx_meta_data,
            MetaData.idx_remaining_steps,
        )
    ).apply().cast(IOptional[IInt]).value
    return value.as_int if value is not None else None

def dynamic_action_test():
    params = (Param.from_int(1), Param.from_int(2), Param.from_int(3))
    p1, p2, p3 = params
    goal = HaveScratch.with_goal(
        InstanceType(
            Or(
                And(p1, p2, IntBoolean(1)),
                And(p2, p3),
            ),
        )
    )

    node_types = GoalEnv.default_node_types()

    def get_action_info_list(meta: MetaInfo):
        actions: list[BaseAction] = []
        outputs: list[IActionOutput] = []

        meta_idx, type_idx = get_info_type_index(And, meta, node_types)
        actions.append(DefineScratchFromIntIndex.from_raw(1, meta_idx, type_idx))
        outputs.append(DefineScratchOutput(
            StateScratchIndex(1),
            Scratch(And.as_type()),
        ))

        actions.append(CreateScratch.create())
        outputs.append(CreateScratchOutput(
            StateScratchIndex(2),
            Scratch(),
        ))

        meta_idx, type_idx = get_info_type_index(Or, meta, node_types)
        actions.append(DefineScratchFromIntIndex.from_raw(2, meta_idx, type_idx))
        outputs.append(DefineScratchOutput(
            StateScratchIndex(2),
            Scratch(Or.as_type()),
        ))

        actions.append(CreateScratch.create())
        outputs.append(CreateScratchOutput(
            StateScratchIndex(3),
            Scratch(),
        ))

        actions.append(CreateArgsGroup.from_raw(3, 0, 0))
        outputs.append(CreateArgsGroupOutput(
            StateArgsGroupIndex(1),
            PartialArgsGroup.from_int(3),
        ))

        meta_idx = get_from_int_type_index(Param, meta)
        actions.append(DefineScratchFromInt.from_raw(3, meta_idx, 1))
        outputs.append(DefineScratchOutput(
            StateScratchIndex(3),
            Scratch(Param.from_int(1)),
        ))

        actions.append(DefineArgsGroup.from_raw(1, 1, 3))
        outputs.append(DefineArgsGroupArgOutput(
            StateArgsGroupIndex(1),
            NodeArgIndex(1),
            Scratch(Param.from_int(1)),
        ))

        meta_idx = get_from_int_type_index(Param, meta)
        actions.append(DefineScratchFromInt.from_raw(3, meta_idx, 2))
        outputs.append(DefineScratchOutput(
            StateScratchIndex(3),
            Scratch(Param.from_int(2)),
        ))

        actions.append(DefineArgsGroup.from_raw(1, 2, 3))
        outputs.append(DefineArgsGroupArgOutput(
            StateArgsGroupIndex(1),
            NodeArgIndex(2),
            Scratch(Param.from_int(2)),
        ))

        meta_idx = get_from_int_type_index(IntBoolean, meta)
        actions.append(DefineScratchFromInt.from_raw(3, meta_idx, 1))
        outputs.append(DefineScratchOutput(
            StateScratchIndex(3),
            Scratch(IntBoolean.from_int(1)),
        ))

        actions.append(DefineArgsGroup.from_raw(1, 3, 3))
        outputs.append(DefineArgsGroupArgOutput(
            StateArgsGroupIndex(1),
            NodeArgIndex(3),
            Scratch(IntBoolean.from_int(1)),
        ))

        actions.append(CreateArgsGroup.from_raw(2, 0, 0))
        outputs.append(CreateArgsGroupOutput(
            StateArgsGroupIndex(2),
            PartialArgsGroup.from_int(2),
        ))

        meta_idx = get_from_int_type_index(Param, meta)
        actions.append(DefineScratchFromInt.from_raw(3, meta_idx, 2))
        outputs.append(DefineScratchOutput(
            StateScratchIndex(3),
            Scratch(Param.from_int(2)),
        ))

        actions.append(DefineArgsGroup.from_raw(2, 1, 3))
        outputs.append(DefineArgsGroupArgOutput(
            StateArgsGroupIndex(2),
            NodeArgIndex(1),
            Scratch(Param.from_int(2)),
        ))

        meta_idx = get_from_int_type_index(Param, meta)
        actions.append(DefineScratchFromInt.from_raw(3, meta_idx, 3))
        outputs.append(DefineScratchOutput(
            StateScratchIndex(3),
            Scratch(Param.from_int(3)),
        ))

        actions.append(DefineArgsGroup.from_raw(2, 2, 3))
        outputs.append(DefineArgsGroupArgOutput(
            StateArgsGroupIndex(2),
            NodeArgIndex(2),
            Scratch(Param.from_int(3)),
        ))

        actions.append(DefineScratchFromFunctionWithArgs.from_raw(3, 1, 1))
        outputs.append(DefineScratchOutput(
            StateScratchIndex(3),
            Scratch(And(
                Param.from_int(1),
                Param.from_int(2),
                IntBoolean.from_int(1),
            )),
        ))

        actions.append(DefineScratchFromFunctionWithArgs.from_raw(1, 1, 2))
        outputs.append(DefineScratchOutput(
            StateScratchIndex(1),
            Scratch(And(
                Param.from_int(2),
                Param.from_int(3),
            )),
        ))

        actions.append(DefineArgsGroup.from_raw(2, 1, 3))
        outputs.append(DefineArgsGroupArgOutput(
            StateArgsGroupIndex(2),
            NodeArgIndex(1),
            Scratch(And(
                Param.from_int(1),
                Param.from_int(2),
                IntBoolean.from_int(1),
            )),
        ))

        actions.append(DefineArgsGroup.from_raw(2, 2, 1))
        outputs.append(DefineArgsGroupArgOutput(
            StateArgsGroupIndex(2),
            NodeArgIndex(2),
            Scratch(And(
                Param.from_int(2),
                Param.from_int(3),
            )),
        ))

        actions.append(DefineScratchFromFunctionWithArgs.from_raw(1, 2, 2))
        outputs.append(DefineScratchOutput(
            StateScratchIndex(1),
            Scratch(Or(
                And(
                    Param.from_int(1),
                    Param.from_int(2),
                    IntBoolean.from_int(1),
                ),
                And(
                    Param.from_int(2),
                    Param.from_int(3),
                ),
            )),
        ))

        actions.append(DeleteArgsGroupOutput.from_raw(1, 0, 0))
        outputs.append(DeleteArgsGroupOutput(StateArgsGroupIndex(1)))

        actions.append(DeleteArgsGroupOutput.from_raw(1, 0, 0))
        outputs.append(DeleteArgsGroupOutput(StateArgsGroupIndex(1)))

        actions.append(DeleteScratchOutput.from_raw(3, 0, 0))
        outputs.append(DeleteScratchOutput(StateScratchIndex(3)))

        actions.append(DeleteScratchOutput.from_raw(2, 0, 0))
        outputs.append(DeleteScratchOutput(StateScratchIndex(2)))

        meta_idx = get_single_child_type_index(InstanceType, meta)
        actions.append(DefineScratchFromSingleArg.from_raw(1, meta_idx, 1))
        outputs.append(DefineScratchOutput(
            StateScratchIndex(1),
            Scratch(InstanceType(Or(
                And(
                    Param.from_int(1),
                    Param.from_int(2),
                    IntBoolean.from_int(1),
                ),
                And(
                    Param.from_int(2),
                    Param.from_int(3),
                ),
            ))),
        ))

        actions.append(CreateScratch.create())
        outputs.append(CreateScratchOutput(
            StateScratchIndex(2),
            Scratch(),
        ))

        meta_idx = get_from_int_type_index(Param, meta)
        actions.append(DefineScratchFromInt.from_raw(2, meta_idx, 4))
        outputs.append(DefineScratchOutput(
            StateScratchIndex(2),
            Scratch(Param.from_int(4)),
        ))

        actions.append(DeleteScratchOutput.from_raw(2, 0, 0))
        outputs.append(DeleteScratchOutput(StateScratchIndex(2)))

        meta_idx = get_from_int_type_index(StateScratchIndex, meta)
        actions.append(VerifyGoal.from_raw(0, meta_idx, 1))
        outputs.append(VerifyGoalOutput(
            Optional(),
            StateScratchIndex(1),
        ))

        assert len(actions) == len(outputs)

        return list(zip(actions, outputs))

    env = GoalEnv(
        goal=goal,
        fn_initial_state=lambda meta: FullState.with_args(
            meta=meta,
            current=HistoryNode.with_args(
                state=State.from_raw(
                    meta_info=StateMetaInfo.with_goal_expr(goal),
                    scratches= [GroupAction(*[a for a, _ in get_action_info_list(meta)])],
                ),
                meta_data=MetaData.with_args(
                    remaining_steps=1,
                ),
            )
        ),
    )
    env.full_state.validate()

    selected_goal = env.full_state.nested_arg((FullState.idx_meta, MetaInfo.idx_goal)).apply()
    assert selected_goal == goal

    meta = env.full_state.meta.apply().cast(MetaInfo)
    action_info_list = get_action_info_list(meta)
    group_action = GroupAction(*[a for a, _ in action_info_list])
    scratches = tuple([group_action])

    current_state = get_current_state(env)
    assert current_state == State.from_raw(scratches=scratches)

    action = DynamicAction.from_raw(1, 0, 0)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    expected_history = SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DynamicAction(StateScratchIndex(1))),
        output=Optional(DynamicActionOutput(
            group_action,
            GroupActionOutput(*[
                DefaultGroup(action, output)
                for action, output in action_info_list
            ]),
        )),
        exception=Optional(),
    )
    scratch_goal = InstanceType(Or(
        And(
            Param.from_int(1),
            Param.from_int(2),
            IntBoolean.from_int(1),
        ),
        And(
            Param.from_int(2),
            Param.from_int(3),
        ),
    ))
    expected_state = State.from_raw(
        meta_info=StateMetaInfo.with_goal_achieved(GoalAchieved.achieved()),
        scratches=[scratch_goal],
    )

    if last_history_action != expected_history:
        print('last_history_action:', env.symbol(last_history_action))
        print('expected_history:', env.symbol(expected_history))

    if current_state != expected_state:
        print('current_state:', env.symbol(current_state))
        print('expected_state:', env.symbol(expected_state))

    actual_action = last_history_action.real(BaseActionData).action.apply()
    expected_action = Optional(DynamicAction(StateScratchIndex(1)))

    assert actual_action == expected_action
    assert current_state == expected_state
    assert last_history_action == expected_history

    assert env.full_state.goal_achieved() is True

    remaining_steps = get_remaining_steps(env)
    assert remaining_steps == 0

    return [env.full_state]

def restore_history_single_test(history_amount: int, delete_goal_after: int, history_index: int):
    assert 0 <= delete_goal_after < history_amount

    scratch_goal = Add(
        Integer(history_amount*2),
        Integer(1+history_amount*3),
        Integer(3+delete_goal_after),
    )
    goal = HaveScratch.with_goal(scratch_goal)

    initial_scratches = [
        (
            Integer(i+1)
            if i != delete_goal_after
            else scratch_goal
        )
        for i in range(history_amount)
    ]
    single_restore = history_amount - history_index <= delete_goal_after
    original_remaining_steps = 2 if single_restore else 3

    env = GoalEnv(
        goal=goal,
        max_steps=original_remaining_steps + history_amount,
        fn_initial_state=lambda meta: FullState.with_args(
            meta=meta,
            current=HistoryNode.with_args(
                state=State.from_raw(
                    meta_info=StateMetaInfo.with_goal_expr(goal),
                    scratches= [],
                ),
                meta_data=MetaData.with_args(
                    remaining_steps=original_remaining_steps,
                ),
            ),
            history=HistoryGroupNode(*[
                HistoryNode.with_args(
                    state=State.from_raw(
                        meta_info=StateMetaInfo.with_goal_expr(goal),
                        scratches=initial_scratches[i:],
                    ),
                    meta_data=MetaData.with_args(
                        remaining_steps=original_remaining_steps + history_amount - i,
                    ),
                    action_data=Optional(
                        SuccessActionData.from_args(
                            raw_action=Optional(),
                            action=Optional(DeleteScratchOutput(
                                StateScratchIndex(1),
                            )),
                            output=Optional(DeleteScratchOutput(
                                StateScratchIndex(1),
                            )),
                            exception=Optional(),
                        ),
                    ),
                )
                for i in range(history_amount)
            ]),
        ),
    )
    env.full_state.validate()

    selected_goal = env.full_state.nested_arg((FullState.idx_meta, MetaInfo.idx_goal)).apply()
    assert selected_goal == goal

    meta = env.full_state.meta.apply().cast(MetaInfo)
    state_meta = StateMetaInfo.with_goal_expr(goal)
    prev_remaining_steps = get_remaining_steps(env)

    # Run Action
    raw_action: BaseAction = RestoreHistoryStateOutput.from_raw(history_index, 0, 0)
    full_action: BaseAction = RestoreHistoryStateOutput(NodeArgReverseIndex(history_index))
    output: BaseAction = full_action
    env.step(raw_action)
    if prev_remaining_steps is not None:
        remaining_steps = get_remaining_steps(env)
        assert remaining_steps == prev_remaining_steps - 1
        prev_remaining_steps = remaining_steps
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    truncate_scratch_at = history_amount-history_index
    scratches = initial_scratches[truncate_scratch_at:]

    expected_history = SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(full_action),
        output=Optional(output),
        exception=Optional(),
    )
    if last_history_action != expected_history:
        print('last_history_action:', env.symbol(last_history_action))
        print('expected_history:', env.symbol(expected_history))
    assert last_history_action == expected_history

    expected_state = State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    if current_state != expected_state:
        print('current_state:', env.symbol(current_state))
        print('expected_state:', env.symbol(expected_state))
    assert current_state == expected_state

    assert env.full_state.goal_achieved() is False

    if single_restore:
        assert delete_goal_after - truncate_scratch_at >= 0
        assert scratches[delete_goal_after - truncate_scratch_at] == scratch_goal
    else:
        # Add 1 because the previous action increased the history items
        new_history_index = history_amount+1

        # Run Action
        raw_action = RestoreHistoryStateOutput.from_raw(new_history_index, 0, 0)
        full_action = RestoreHistoryStateOutput(NodeArgReverseIndex(new_history_index))
        output = full_action
        env.step(raw_action)
        if prev_remaining_steps is not None:
            remaining_steps = get_remaining_steps(env)
            assert remaining_steps == prev_remaining_steps - 1
            prev_remaining_steps = remaining_steps
        current_state = get_current_state(env)
        last_history_action = get_last_history_action(env)

        # Verify
        scratches = initial_scratches

        expected_history = SuccessActionData.from_args(
            raw_action=Optional(),
            action=Optional(full_action),
            output=Optional(output),
            exception=Optional(),
        )
        if last_history_action != expected_history:
            print('last_history_action:', env.symbol(last_history_action))
            print('expected_history:', env.symbol(expected_history))
        assert last_history_action == expected_history

        expected_state = State.from_raw(
            meta_info=state_meta,
            scratches=scratches,
        )
        if current_state != expected_state:
            print('current_state:', env.symbol(current_state))
            print('expected_state:', env.symbol(expected_state))
        assert current_state == expected_state

        assert env.full_state.goal_achieved() is False

    # Run Action
    meta_idx = get_from_int_type_index(StateScratchIndex, meta)
    scratch_index = 1 + (
        (delete_goal_after - truncate_scratch_at)
        if single_restore
        else delete_goal_after
    )
    raw_action = VerifyGoal.from_raw(0, meta_idx, scratch_index)
    full_action = VerifyGoal(
        Optional(),
        MetaFromIntTypeIndex(meta_idx),
        Integer(scratch_index),
    )
    output = VerifyGoalOutput(
        Optional(),
        StateScratchIndex(scratch_index),
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
        goal_achieved=GoalAchieved.achieved(),
    )

    expected_history = SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(full_action),
        output=Optional(output),
        exception=Optional(),
    )
    if last_history_action != expected_history:
        print('last_history_action:', env.symbol(last_history_action))
        print('expected_history:', env.symbol(expected_history))
    assert last_history_action == expected_history

    expected_state = State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
    )
    if current_state != expected_state:
        print('current_state:', env.symbol(current_state))
        print('expected_state:', env.symbol(expected_state))
    assert current_state == expected_state

    assert env.full_state.goal_achieved() is True

    assert remaining_steps == 0, remaining_steps

    return [env.full_state]

def restore_history_test() -> list[FullState]:
    final_states: list[FullState] = []
    final_states += restore_history_single_test(1, 0, 1)
    final_states += restore_history_single_test(3, 2, 1)
    final_states += restore_history_single_test(3, 0, 3)
    final_states += restore_history_single_test(3, 0, 1)
    final_states += restore_history_single_test(7, 4, 2)
    final_states += restore_history_single_test(9, 7, 8)
    final_states += restore_history_single_test(9, 7, 1)
    return final_states

def test() -> list[FullState]:
    final_states: list[FullState] = []
    final_states += dynamic_action_test()
    final_states += restore_history_test()
    return final_states
