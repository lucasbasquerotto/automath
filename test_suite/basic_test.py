from env.core import (
    Param,
    And,
    Or,
    IntBoolean,
    Integer,
    BooleanExceptionInfo,
    IsEmpty,
    NodeArgIndex,
    InstanceType,
)
from env.full_state import (
    FullState,
    MetaFromIntTypeIndex,
    MetaFullStateIntIndexTypeIndex,
    MetaAllTypesTypeIndex,
    HistoryGroupNode,
    HistoryNode,
    ActionErrorActionData,
    SuccessActionData,
)
from env.meta_env import MetaInfo, SubtypeOuterGroup, GeneralTypeGroup
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
from env.action import IAction
from env.core import Optional

def get_current_state(env: GoalEnv):
    return env.full_state.nested_arg(
        (FullState.idx_current, HistoryNode.idx_state)
    ).apply().cast(State)

def get_last_history_action(env: GoalEnv):
    history = env.full_state.history.apply().cast(HistoryGroupNode)
    last = history.as_tuple[-1]
    return last.action_data.apply().inner_arg(Optional.idx_value).apply()

def get_from_int_type_index(node_type: type[INode], env: GoalEnv):
    selected_types = env.full_state.meta.apply().nested_arg((
        MetaInfo.idx_from_int_group,
        SubtypeOuterGroup.idx_subtypes,
    )).apply().cast(GeneralTypeGroup)
    meta_idx = selected_types.as_tuple.index(node_type.as_type()) + 1
    return meta_idx

def get_info_type_index(node_type: type[INode], env: GoalEnv, node_types: tuple[type[INode], ...]):
    index_type_idx = node_types.index(MetaAllTypesTypeIndex) + 1
    type_node = node_types[index_type_idx-1].as_type()
    selected_types = env.full_state.meta.apply().nested_arg((
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

def get_single_child_type_index(node_type: type[INode], env: GoalEnv):
    selected_types = env.full_state.meta.apply().nested_arg((
        MetaInfo.idx_single_child_group,
        SubtypeOuterGroup.idx_subtypes,
    )).apply().cast(GeneralTypeGroup)
    meta_idx = selected_types.as_tuple.index(node_type.as_type()) + 1
    return meta_idx


def basic_test():
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

    env = GoalEnv(goal)
    env.full_state.validate()
    node_types = env.full_state.node_types()

    selected_goal = env.full_state.nested_arg((FullState.idx_meta, MetaInfo.idx_goal)).apply()
    assert selected_goal == goal

    current_state = get_current_state(env)
    assert current_state == State.create()

    # Test case 1
    action = CreateScratch.create()
    env.step(action)
    current_state = get_current_state(env)
    assert current_state == State.from_raw(
        scratches=tuple([None]),
        args_groups=tuple(),
    )
    assert env.full_state.goal_achieved() is False
    last_history_action = get_last_history_action(env)
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(CreateScratch(Optional())),
        output=Optional(CreateScratchOutput(
            StateScratchIndex(1),
            Scratch(),
        )),
        exception=Optional(),
    )

    # Test case 2
    # should return an exception (index won't return element)
    action = DefineScratchFromInt.from_raw(1, len(node_types)+1, 3)
    env.step(action)
    current_state = get_current_state(env)
    assert current_state == State.from_raw(
        scratches=tuple([None]),
        args_groups=tuple(),
    )
    assert env.full_state.goal_achieved() is False
    last_history_action = get_last_history_action(env)
    print(env.symbol(last_history_action))
    assert last_history_action == get_empty_exception(action)

    meta_idx, type_idx = get_info_type_index(And, env, node_types)
    action = DefineScratchFromIntIndex.from_raw(1, meta_idx, type_idx)
    env.step(action)
    current_state = get_current_state(env)
    assert current_state == State.from_raw(
        scratches=tuple([And.as_type()]),
        args_groups=tuple(),
    )
    assert env.full_state.goal_achieved() is False
    last_history_action = get_last_history_action(env)
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DefineScratchFromIntIndex(
            StateScratchIndex(1),
            MetaFullStateIntIndexTypeIndex(meta_idx),
            Integer(type_idx)
        )),
        output=Optional(DefineScratchOutput(
            StateScratchIndex(1),
            Scratch(And.as_type()),
        )),
        exception=Optional(),
    )

    # Test case 3
    action = CreateScratch.create()
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    assert current_state == State.from_raw(
        scratches=tuple([And.as_type(), None]),
        args_groups=tuple(),
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(CreateScratch(Optional())),
        output=Optional(CreateScratchOutput(
            StateScratchIndex(2),
            Scratch(),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 4
    meta_idx, type_idx = get_info_type_index(Or, env, node_types)
    action = DefineScratchFromIntIndex.from_raw(2, meta_idx, type_idx)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    assert current_state == State.from_raw(
        scratches=tuple([And.as_type(), Or.as_type()]),
        args_groups=tuple(),
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DefineScratchFromIntIndex(
            StateScratchIndex(2),
            MetaFullStateIntIndexTypeIndex(meta_idx),
            Integer(type_idx)
        )),
        output=Optional(DefineScratchOutput(
            StateScratchIndex(2),
            Scratch(Or.as_type()),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 5
    action = CreateScratch.create()
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    assert current_state == State.from_raw(
        scratches=tuple([And.as_type(), Or.as_type(), None]),
        args_groups=tuple(),
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(CreateScratch(Optional())),
        output=Optional(CreateScratchOutput(
            StateScratchIndex(3),
            Scratch(),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 6
    action = CreateArgsGroup.from_raw(3, 0, 0)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    args_group = PartialArgsGroup.from_int(3)
    assert current_state == State.from_raw(
        scratches=tuple([And.as_type(), Or.as_type(), None]),
        args_groups=tuple([args_group]),
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(CreateArgsGroup(
            Integer(3),
            Optional(),
        )),
        output=Optional(CreateArgsGroupOutput(
            StateArgsGroupIndex(1),
            args_group,
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 7
    meta_idx = get_from_int_type_index(Param, env)
    action = DefineScratchFromInt.from_raw(3, meta_idx, 1)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    assert current_state == State.from_raw(
        scratches=tuple([And.as_type(), Or.as_type(), Param.from_int(1)]),
        args_groups=tuple([args_group]),
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DefineScratchFromInt(
            StateScratchIndex(3),
            MetaFromIntTypeIndex(meta_idx),
            Integer(1),
        )),
        output=Optional(DefineScratchOutput(
            StateScratchIndex(3),
            Scratch(Param.from_int(1)),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 8
    action = DefineArgsGroup.from_raw(1, 1, 3)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    args_group = PartialArgsGroup(
        Optional(Param.from_int(1)),
        Optional(),
        Optional(),
    )
    assert current_state == State.from_raw(
        scratches=tuple([And.as_type(), Or.as_type(), Param.from_int(1)]),
        args_groups=tuple([args_group]),
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DefineArgsGroup(
            StateArgsGroupIndex(1),
            NodeArgIndex(1),
            StateScratchIndex(3),
        )),
        output=Optional(DefineArgsGroupArgOutput(
            StateArgsGroupIndex(1),
            NodeArgIndex(1),
            Scratch(Param.from_int(1)),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 9
    meta_idx = get_from_int_type_index(Param, env)
    action = DefineScratchFromInt.from_raw(3, meta_idx, 2)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    assert current_state == State.from_raw(
        scratches=tuple([And.as_type(), Or.as_type(), Param.from_int(2)]),
        args_groups=tuple([args_group]),
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DefineScratchFromInt(
            StateScratchIndex(3),
            MetaFromIntTypeIndex(meta_idx),
            Integer(2),
        )),
        output=Optional(DefineScratchOutput(
            StateScratchIndex(3),
            Scratch(Param.from_int(2)),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 10
    action = DefineArgsGroup.from_raw(1, 2, 3)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    args_group = PartialArgsGroup(
        Optional(Param.from_int(1)),
        Optional(Param.from_int(2)),
        Optional(),
    )
    assert current_state == State.from_raw(
        scratches=tuple([And.as_type(), Or.as_type(), Param.from_int(2)]),
        args_groups=tuple([args_group]),
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DefineArgsGroup(
            StateArgsGroupIndex(1),
            NodeArgIndex(2),
            StateScratchIndex(3),
        )),
        output=Optional(DefineArgsGroupArgOutput(
            StateArgsGroupIndex(1),
            NodeArgIndex(2),
            Scratch(Param.from_int(2)),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 11
    meta_idx = get_from_int_type_index(IntBoolean, env)
    action = DefineScratchFromInt.from_raw(3, meta_idx, 1)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    assert current_state == State.from_raw(
        scratches=tuple([And.as_type(), Or.as_type(), IntBoolean.from_int(1)]),
        args_groups=tuple([args_group]),
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DefineScratchFromInt(
            StateScratchIndex(3),
            MetaFromIntTypeIndex(meta_idx),
            Integer(1),
        )),
        output=Optional(DefineScratchOutput(
            StateScratchIndex(3),
            Scratch(IntBoolean.from_int(1)),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 12
    action = DefineArgsGroup.from_raw(1, 3, 3)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    args_group = PartialArgsGroup(
        Optional(Param.from_int(1)),
        Optional(Param.from_int(2)),
        Optional(IntBoolean.from_int(1)),
    )
    scratches = tuple([And.as_type(), Or.as_type(), IntBoolean.from_int(1)])
    args_groups = tuple([args_group])
    assert current_state == State.from_raw(
        scratches=scratches,
        args_groups=args_groups,
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DefineArgsGroup(
            StateArgsGroupIndex(1),
            NodeArgIndex(3),
            StateScratchIndex(3),
        )),
        output=Optional(DefineArgsGroupArgOutput(
            StateArgsGroupIndex(1),
            NodeArgIndex(3),
            Scratch(IntBoolean.from_int(1)),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 13
    action = CreateArgsGroup.from_raw(2, 0, 0)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    arg_group_2 = PartialArgsGroup(
        Optional(),
        Optional(),
    )
    args_groups = tuple([args_group, arg_group_2])
    assert current_state == State.from_raw(
        scratches=scratches,
        args_groups=args_groups,
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(CreateArgsGroup(
            Integer(2),
            Optional(),
        )),
        output=Optional(CreateArgsGroupOutput(
            StateArgsGroupIndex(2),
            arg_group_2,
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 14
    meta_idx = get_from_int_type_index(Param, env)
    action = DefineScratchFromInt.from_raw(3, meta_idx, 2)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    scratches = tuple([And.as_type(), Or.as_type(), Param.from_int(2)])
    assert current_state == State.from_raw(
        scratches=scratches,
        args_groups=args_groups,
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DefineScratchFromInt(
            StateScratchIndex(3),
            MetaFromIntTypeIndex(meta_idx),
            Integer(2),
        )),
        output=Optional(DefineScratchOutput(
            StateScratchIndex(3),
            Scratch(Param.from_int(2)),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 15
    action = DefineArgsGroup.from_raw(2, 1, 3)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    arg_group_2 = PartialArgsGroup(
        Optional(Param.from_int(2)),
        Optional(),
    )
    args_groups = tuple([args_group, arg_group_2])
    assert current_state == State.from_raw(
        scratches=scratches,
        args_groups=args_groups,
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DefineArgsGroup(
            StateArgsGroupIndex(2),
            NodeArgIndex(1),
            StateScratchIndex(3),
        )),
        output=Optional(DefineArgsGroupArgOutput(
            StateArgsGroupIndex(2),
            NodeArgIndex(1),
            Scratch(Param.from_int(2)),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 16
    meta_idx = get_from_int_type_index(Param, env)
    action = DefineScratchFromInt.from_raw(3, meta_idx, 3)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    scratches = tuple([And.as_type(), Or.as_type(), Param.from_int(3)])
    assert current_state == State.from_raw(
        scratches=scratches,
        args_groups=args_groups,
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DefineScratchFromInt(
            StateScratchIndex(3),
            MetaFromIntTypeIndex(meta_idx),
            Integer(3),
        )),
        output=Optional(DefineScratchOutput(
            StateScratchIndex(3),
            Scratch(Param.from_int(3)),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 17
    action = DefineArgsGroup.from_raw(2, 2, 3)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    arg_group_2 = PartialArgsGroup(
        Optional(Param.from_int(2)),
        Optional(Param.from_int(3)),
    )
    args_groups = tuple([args_group, arg_group_2])
    assert current_state == State.from_raw(
        scratches=scratches,
        args_groups=args_groups,
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DefineArgsGroup(
            StateArgsGroupIndex(2),
            NodeArgIndex(2),
            StateScratchIndex(3),
        )),
        output=Optional(DefineArgsGroupArgOutput(
            StateArgsGroupIndex(2),
            NodeArgIndex(2),
            Scratch(Param.from_int(3)),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 18
    action = DefineScratchFromFunctionWithArgs.from_raw(3, 1, 1)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    call_1 = And(
        Param.from_int(1),
        Param.from_int(2),
        IntBoolean.from_int(1),
    )
    scratches = tuple([And.as_type(), Or.as_type(), call_1])
    assert current_state == State.from_raw(
        scratches=scratches,
        args_groups=args_groups,
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DefineScratchFromFunctionWithArgs(
            StateScratchIndex(3),
            StateScratchIndex(1),
            Optional(StateArgsGroupIndex(1)),
        )),
        output=Optional(DefineScratchOutput(
            StateScratchIndex(3),
            Scratch(call_1),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 19
    action = DefineScratchFromFunctionWithArgs.from_raw(1, 1, 2)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    call_2 = And(
        Param.from_int(2),
        Param.from_int(3),
    )
    scratches = tuple([call_2, Or.as_type(), call_1])
    assert current_state == State.from_raw(
        scratches=scratches,
        args_groups=args_groups,
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DefineScratchFromFunctionWithArgs(
            StateScratchIndex(1),
            StateScratchIndex(1),
            Optional(StateArgsGroupIndex(2)),
        )),
        output=Optional(DefineScratchOutput(
            StateScratchIndex(1),
            Scratch(call_2),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 20
    action = DefineArgsGroup.from_raw(2, 1, 3)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    arg_group_2 = PartialArgsGroup(
        Optional(call_1),
        Optional(Param.from_int(3)),
    )
    args_groups = tuple([args_group, arg_group_2])
    assert current_state == State.from_raw(
        scratches=scratches,
        args_groups=args_groups
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DefineArgsGroup(
            StateArgsGroupIndex(2),
            NodeArgIndex(1),
            StateScratchIndex(3),
        )),
        output=Optional(DefineArgsGroupArgOutput(
            StateArgsGroupIndex(2),
            NodeArgIndex(1),
            Scratch(call_1),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 21
    action = DefineArgsGroup.from_raw(2, 2, 1)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    arg_group_2 = PartialArgsGroup(
        Optional(call_1),
        Optional(call_2),
    )
    args_groups = tuple([args_group, arg_group_2])
    assert current_state == State.from_raw(
        scratches=scratches,
        args_groups=args_groups,
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DefineArgsGroup(
            StateArgsGroupIndex(2),
            NodeArgIndex(2),
            StateScratchIndex(1),
        )),
        output=Optional(DefineArgsGroupArgOutput(
            StateArgsGroupIndex(2),
            NodeArgIndex(2),
            Scratch(call_2),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 22
    action = DefineScratchFromFunctionWithArgs.from_raw(1, 2, 2)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    call_3 = Or(
        call_1,
        call_2,
    )
    scratches = tuple([call_3, Or.as_type(), call_1])

    assert current_state == State.from_raw(
        scratches=scratches,
        args_groups=args_groups,
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DefineScratchFromFunctionWithArgs(
            StateScratchIndex(1),
            StateScratchIndex(2),
            Optional(StateArgsGroupIndex(2)),
        )),
        output=Optional(DefineScratchOutput(
            StateScratchIndex(1),
            Scratch(call_3),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 23
    action = DeleteArgsGroupOutput.from_raw(1, 0, 0)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    args_groups = tuple([arg_group_2])
    assert current_state == State.from_raw(
        scratches=scratches,
        args_groups=args_groups,
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DeleteArgsGroupOutput(
            StateArgsGroupIndex(1),
        )),
        output=Optional(DeleteArgsGroupOutput(
            StateArgsGroupIndex(1),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 24
    action = DeleteArgsGroupOutput.from_raw(1, 0, 0)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    args_groups = tuple()
    assert current_state == State.from_raw(
        scratches=scratches,
        args_groups=args_groups
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DeleteArgsGroupOutput(
            StateArgsGroupIndex(1),
        )),
        output=Optional(DeleteArgsGroupOutput(
            StateArgsGroupIndex(1),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 25
    action = DeleteScratchOutput.from_raw(3, 0, 0)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    scratches = tuple([call_3, Or.as_type()])
    assert current_state == State.from_raw(
        scratches=scratches,
        args_groups=args_groups,
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DeleteScratchOutput(
            StateScratchIndex(3),
        )),
        output=Optional(DeleteScratchOutput(
            StateScratchIndex(3),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 26
    action = DeleteScratchOutput.from_raw(2, 0, 0)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    scratches = tuple([call_3])
    assert current_state == State.from_raw(
        scratches=scratches,
        args_groups=args_groups,
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(DeleteScratchOutput(
            StateScratchIndex(2),
        )),
        output=Optional(DeleteScratchOutput(
            StateScratchIndex(2),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 27
    meta_idx = get_single_child_type_index(InstanceType, env)
    action = DefineScratchFromSingleArg.from_raw(1, meta_idx, 1)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    content = current_state.nested_arg((
        State.idx_scratch_group,
        1,
        Scratch.idx_value,
    )).apply()
    assert content == goal.goal_inner_expr.apply()
    verification = goal.evaluate(current_state, StateScratchIndex(1))
    verification.raise_on_false()

    # Test case 28
    action = CreateScratch.create()
    env.step(action)
    meta_idx = get_from_int_type_index(Param, env)
    action = DefineScratchFromInt.from_raw(2, meta_idx, 4)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    verification = goal.evaluate(current_state, StateScratchIndex(2))
    assert verification.as_bool is False

    # Test case 29
    action = DeleteScratchOutput.from_raw(2, 0, 0)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    meta_idx = get_from_int_type_index(StateScratchIndex, env)
    action = VerifyGoal.from_raw(0, meta_idx, 1)
    env.step(action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)
    scratch_goal = InstanceType(call_3)
    assert current_state == State.from_raw(
        meta_info=StateMetaInfo.with_goal_achieved(GoalAchieved.achieved()),
        scratches=[scratch_goal],
        args_groups=args_groups,
    )
    assert last_history_action == SuccessActionData.from_args(
        raw_action=Optional(),
        action=Optional(VerifyGoal(
            Optional(),
            MetaFromIntTypeIndex(meta_idx),
            Integer(1),
        )),
        output=Optional(VerifyGoalOutput(
            Optional(),
            StateScratchIndex(1),

        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is True

    return [env.full_state]


def test() -> list[FullState]:
    return basic_test()
