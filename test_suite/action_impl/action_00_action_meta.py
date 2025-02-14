from env.core import (
    DefaultGroup,
    Param,
    And,
    Or,
    IntBoolean,
    BooleanExceptionInfo,
    IsEmpty,
    NodeArgIndex,
    InstanceType,
    IOptional,
    IInt,
)
from env.full_state import (
    FullState,
    MetaAllTypesTypeIndex,
    HistoryGroupNode,
    HistoryNode,
    ActionData,
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
    return ActionData.from_args(
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

def action_meta_test():
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

    expected_history = ActionData.from_args(
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

    actual_action = last_history_action.real(ActionData).action.apply()
    expected_action = Optional(DynamicAction(StateScratchIndex(1)))

    assert actual_action == expected_action
    assert current_state == expected_state
    assert last_history_action == expected_history

    assert env.full_state.goal_achieved() is True

    remaining_steps = get_remaining_steps(env)
    assert remaining_steps == 0

    return [env.full_state]


def test() -> list[FullState]:
    return action_meta_test()
