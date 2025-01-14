from env.core import (
    FunctionExpr,
    Param,
    And,
    Or,
    IntBoolean,
    Integer,
    ExtendedTypeGroup,
    RestTypeGroup,
    UnknownType,
    LaxOpaqueScope,
    ScopeId,
    OptionalValueGroup,
    BooleanExceptionInfo,
    IsEmpty,
    NodeArgIndex,
)
from env.full_state import (
    FullState,
    MetaFromIntTypeIndex,
    MetaFullStateIntIndexTypeIndex,
    MetaAllTypesTypeIndex,
    HistoryGroupNode,
    HistoryNode,
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
    OptionalContext,
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
from env.action import IAction, ActionData
from env.core import Optional

def get_current_state(env: GoalEnv):
    return env.full_state.nested_args(
        (FullState.idx_current, HistoryNode.idx_state)
    ).apply().cast(State)

def get_from_int_type_index(node_type: type[INode], env: GoalEnv):
    selected_types = env.full_state.meta.apply().nested_args((
        MetaInfo.idx_from_int_group,
        SubtypeOuterGroup.idx_subtypes,
    )).apply().cast(GeneralTypeGroup)
    meta_idx = selected_types.as_tuple.index(node_type.as_type()) + 1
    return meta_idx

def get_info_type_index(node_type: type[INode], env: GoalEnv, node_types: tuple[type[INode], ...]):
    index_type_idx = node_types.index(MetaAllTypesTypeIndex) + 1
    type_node = node_types[index_type_idx-1].as_type()
    selected_types = env.full_state.meta.apply().nested_args((
        MetaInfo.idx_full_state_int_index_group,
        SubtypeOuterGroup.idx_subtypes,
    )).apply().cast(GeneralTypeGroup)
    meta_idx = selected_types.as_tuple.index(type_node) + 1
    node_idx = node_types.index(node_type) + 1
    return meta_idx, node_idx

def get_last_history_meta(env: GoalEnv):
    history = env.full_state.history.apply().cast(HistoryGroupNode)
    last = history.as_tuple[-1]
    return last.meta_data.apply().nested_arg(Optional.idx_value).apply()

def get_empty_exception(action: IAction):
    return ActionData.from_args(
        action=Optional(action),
        output=Optional(),
        exception=Optional(BooleanExceptionInfo(IsEmpty(Optional()))),
    )

def get_single_child_type_index(node_type: type[INode], env: GoalEnv):
    selected_types = env.full_state.meta.apply().nested_args((
        MetaInfo.idx_single_child_group,
        SubtypeOuterGroup.idx_subtypes,
    )).apply().cast(GeneralTypeGroup)
    meta_idx = selected_types.as_tuple.index(node_type.as_type()) + 1
    return meta_idx


def basic_test():
    params = (Param.from_int(1), Param.from_int(2), Param.from_int(3))
    p1, p2, p3 = params
    goal = HaveScratch.with_goal(
        FunctionExpr.with_child(
            Or(
                And(p1, p2, IntBoolean(1)),
                And(p2, p3),
            ),
        )
    )

    env = GoalEnv(goal)
    node_types = env.full_state.node_types()

    selected_goal = env.full_state.nested_args((FullState.idx_meta, MetaInfo.idx_goal)).apply()
    assert selected_goal == goal

    current_state = get_current_state(env)
    assert current_state == State.create()

    # Test case 1
    action = CreateScratch.create()
    env.step(action)
    current_state = get_current_state(env)
    assert current_state == State.from_raw(
        scratchs=tuple([None]),
        args_groups=tuple(),
    )
    assert env.full_state.goal_achieved() is False
    last_history_meta = get_last_history_meta(env)
    assert last_history_meta == ActionData.from_args(
        action=Optional(CreateScratch(Optional())),
        output=Optional(CreateScratchOutput(
            StateScratchIndex(1),
            Optional(),
        )),
        exception=Optional(),
    )

    # Test case 2
    # should return an exception (index won't return element)
    action = DefineScratchFromInt.from_raw(1, len(node_types)+1, 3)
    env.step(action)
    current_state = get_current_state(env)
    assert current_state == State.from_raw(
        scratchs=tuple([None]),
        args_groups=tuple(),
    )
    assert env.full_state.goal_achieved() is False
    last_history_meta = get_last_history_meta(env)
    assert last_history_meta == get_empty_exception(action)

    meta_idx, type_idx = get_info_type_index(And, env, node_types)
    action = DefineScratchFromIntIndex.from_raw(1, meta_idx, type_idx)
    env.step(action)
    current_state = get_current_state(env)
    assert current_state == State.from_raw(
        scratchs=tuple([And.as_type()]),
        args_groups=tuple(),
    )
    assert env.full_state.goal_achieved() is False
    last_history_meta = get_last_history_meta(env)
    assert last_history_meta == ActionData.from_args(
        action=Optional(DefineScratchFromIntIndex(
            StateScratchIndex(1),
            MetaFullStateIntIndexTypeIndex(meta_idx),
            Integer(type_idx)
        )),
        output=Optional(DefineScratchOutput(
            StateScratchIndex(1),
            Optional(And.as_type()),
        )),
        exception=Optional(),
    )

    # Test case 3
    action = CreateScratch.create()
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    assert current_state == State.from_raw(
        scratchs=tuple([And.as_type(), None]),
        args_groups=tuple(),
    )
    assert last_history_meta == ActionData.from_args(
        action=Optional(CreateScratch(Optional())),
        output=Optional(CreateScratchOutput(
            StateScratchIndex(2),
            Optional(),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 4
    meta_idx, type_idx = get_info_type_index(Or, env, node_types)
    action = DefineScratchFromIntIndex.from_raw(2, meta_idx, type_idx)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    assert current_state == State.from_raw(
        scratchs=tuple([And.as_type(), Or.as_type()]),
        args_groups=tuple(),
    )
    assert last_history_meta == ActionData.from_args(
        action=Optional(DefineScratchFromIntIndex(
            StateScratchIndex(2),
            MetaFullStateIntIndexTypeIndex(meta_idx),
            Integer(type_idx)
        )),
        output=Optional(DefineScratchOutput(
            StateScratchIndex(2),
            Optional(Or.as_type()),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 5
    action = CreateScratch.create()
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    assert current_state == State.from_raw(
        scratchs=tuple([And.as_type(), Or.as_type(), None]),
        args_groups=tuple(),
    )
    assert last_history_meta == ActionData.from_args(
        action=Optional(CreateScratch(Optional())),
        output=Optional(CreateScratchOutput(
            StateScratchIndex(3),
            Optional(),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 6
    action = CreateArgsGroup.from_raw(3, 0, 0)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    arg_group = PartialArgsGroup(
        ExtendedTypeGroup(RestTypeGroup(UnknownType())),
        LaxOpaqueScope(
            ScopeId(1),
            OptionalValueGroup(
                Optional(),
                Optional(),
                Optional(),
            ),
        )
    )
    assert current_state == State.from_raw(
        scratchs=tuple([And.as_type(), Or.as_type(), None]),
        args_groups=tuple([arg_group]),
    )
    assert last_history_meta == ActionData.from_args(
        action=Optional(CreateArgsGroup(
            Integer(3),
            Optional(),
            Optional(),
        )),
        output=Optional(CreateArgsGroupOutput(
            StateArgsGroupIndex(1),
            arg_group,
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 7
    meta_idx = get_from_int_type_index(Param, env)
    action = DefineScratchFromInt.from_raw(3, meta_idx, 1)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    assert current_state == State.from_raw(
        scratchs=tuple([And.as_type(), Or.as_type(), Param.from_int(1)]),
        args_groups=tuple([arg_group]),
    )
    assert last_history_meta == ActionData.from_args(
        action=Optional(DefineScratchFromInt(
            StateScratchIndex(3),
            MetaFromIntTypeIndex(meta_idx),
            Integer(1),
        )),
        output=Optional(DefineScratchOutput(
            StateScratchIndex(3),
            Optional(Param.from_int(1)),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 8
    action = DefineArgsGroup.from_raw(1, 1, 3)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    arg_group = PartialArgsGroup(
        ExtendedTypeGroup(RestTypeGroup(UnknownType())),
        LaxOpaqueScope(
            ScopeId(1),
            OptionalValueGroup(
                Optional(Param.from_int(1)),
                Optional(),
                Optional(),
            ),
        )
    )
    assert current_state == State.from_raw(
        scratchs=tuple([And.as_type(), Or.as_type(), Param.from_int(1)]),
        args_groups=tuple([arg_group]),
    )
    assert last_history_meta == ActionData.from_args(
        action=Optional(DefineArgsGroup(
            StateArgsGroupIndex(1),
            NodeArgIndex(1),
            StateScratchIndex(3),
        )),
        output=Optional(DefineArgsGroupArgOutput(
            StateArgsGroupIndex(1),
            NodeArgIndex(1),
            OptionalContext(Param.from_int(1)),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 9
    meta_idx = get_from_int_type_index(Param, env)
    action = DefineScratchFromInt.from_raw(3, meta_idx, 2)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    assert current_state == State.from_raw(
        scratchs=tuple([And.as_type(), Or.as_type(), Param.from_int(2)]),
        args_groups=tuple([arg_group]),
    )
    assert last_history_meta == ActionData.from_args(
        action=Optional(DefineScratchFromInt(
            StateScratchIndex(3),
            MetaFromIntTypeIndex(meta_idx),
            Integer(2),
        )),
        output=Optional(DefineScratchOutput(
            StateScratchIndex(3),
            Optional(Param.from_int(2)),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 10
    action = DefineArgsGroup.from_raw(1, 2, 3)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    arg_group = PartialArgsGroup(
        ExtendedTypeGroup(RestTypeGroup(UnknownType())),
        LaxOpaqueScope(
            ScopeId(1),
            OptionalValueGroup(
                Optional(Param.from_int(1)),
                Optional(Param.from_int(2)),
                Optional(),
            ),
        )
    )
    assert current_state == State.from_raw(
        scratchs=tuple([And.as_type(), Or.as_type(), Param.from_int(2)]),
        args_groups=tuple([arg_group]),
    )
    assert last_history_meta == ActionData.from_args(
        action=Optional(DefineArgsGroup(
            StateArgsGroupIndex(1),
            NodeArgIndex(2),
            StateScratchIndex(3),
        )),
        output=Optional(DefineArgsGroupArgOutput(
            StateArgsGroupIndex(1),
            NodeArgIndex(2),
            OptionalContext(Param.from_int(2)),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 11
    meta_idx = get_from_int_type_index(IntBoolean, env)
    action = DefineScratchFromInt.from_raw(3, meta_idx, 1)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    assert current_state == State.from_raw(
        scratchs=tuple([And.as_type(), Or.as_type(), IntBoolean.from_int(1)]),
        args_groups=tuple([arg_group]),
    )
    assert last_history_meta == ActionData.from_args(
        action=Optional(DefineScratchFromInt(
            StateScratchIndex(3),
            MetaFromIntTypeIndex(meta_idx),
            Integer(1),
        )),
        output=Optional(DefineScratchOutput(
            StateScratchIndex(3),
            Optional(IntBoolean.from_int(1)),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 12
    action = DefineArgsGroup.from_raw(1, 3, 3)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    arg_group = PartialArgsGroup(
        ExtendedTypeGroup(RestTypeGroup(UnknownType())),
        LaxOpaqueScope(
            ScopeId(1),
            OptionalValueGroup(
                Optional(Param.from_int(1)),
                Optional(Param.from_int(2)),
                Optional(IntBoolean.from_int(1)),
            ),
        )
    )
    scratchs = tuple([And.as_type(), Or.as_type(), IntBoolean.from_int(1)])
    args_groups = tuple([arg_group])
    assert current_state == State.from_raw(
        scratchs=scratchs,
        args_groups=args_groups,
    )
    assert last_history_meta == ActionData.from_args(
        action=Optional(DefineArgsGroup(
            StateArgsGroupIndex(1),
            NodeArgIndex(3),
            StateScratchIndex(3),
        )),
        output=Optional(DefineArgsGroupArgOutput(
            StateArgsGroupIndex(1),
            NodeArgIndex(3),
            OptionalContext(IntBoolean.from_int(1)),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 13
    action = CreateArgsGroup.from_raw(2, 0, 0)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    arg_group_2 = PartialArgsGroup(
        ExtendedTypeGroup(RestTypeGroup(UnknownType())),
        LaxOpaqueScope(
            ScopeId(1),
            OptionalValueGroup(
                Optional(),
                Optional(),
            ),
        )
    )
    args_groups = tuple([arg_group, arg_group_2])
    assert current_state == State.from_raw(
        scratchs=scratchs,
        args_groups=args_groups,
    )
    assert last_history_meta == ActionData.from_args(
        action=Optional(CreateArgsGroup(
            Integer(2),
            Optional(),
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
    last_history_meta = get_last_history_meta(env)
    scratchs = tuple([And.as_type(), Or.as_type(), Param.from_int(2)])
    assert current_state == State.from_raw(
        scratchs=scratchs,
        args_groups=args_groups,
    )
    assert last_history_meta == ActionData.from_args(
        action=Optional(DefineScratchFromInt(
            StateScratchIndex(3),
            MetaFromIntTypeIndex(meta_idx),
            Integer(2),
        )),
        output=Optional(DefineScratchOutput(
            StateScratchIndex(3),
            Optional(Param.from_int(2)),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 15
    action = DefineArgsGroup.from_raw(2, 1, 3)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    arg_group_2 = PartialArgsGroup(
        ExtendedTypeGroup(RestTypeGroup(UnknownType())),
        LaxOpaqueScope(
            ScopeId(1),
            OptionalValueGroup(
                Optional(Param.from_int(2)),
                Optional(),
            ),
        )
    )
    args_groups = tuple([arg_group, arg_group_2])
    assert current_state == State.from_raw(
        scratchs=scratchs,
        args_groups=args_groups,
    )
    assert last_history_meta == ActionData.from_args(
        action=Optional(DefineArgsGroup(
            StateArgsGroupIndex(2),
            NodeArgIndex(1),
            StateScratchIndex(3),
        )),
        output=Optional(DefineArgsGroupArgOutput(
            StateArgsGroupIndex(2),
            NodeArgIndex(1),
            OptionalContext(Param.from_int(2)),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 16
    meta_idx = get_from_int_type_index(Param, env)
    action = DefineScratchFromInt.from_raw(3, meta_idx, 3)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    scratchs = tuple([And.as_type(), Or.as_type(), Param.from_int(3)])
    assert current_state == State.from_raw(
        scratchs=scratchs,
        args_groups=args_groups,
    )
    assert last_history_meta == ActionData.from_args(
        action=Optional(DefineScratchFromInt(
            StateScratchIndex(3),
            MetaFromIntTypeIndex(meta_idx),
            Integer(3),
        )),
        output=Optional(DefineScratchOutput(
            StateScratchIndex(3),
            Optional(Param.from_int(3)),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 17
    action = DefineArgsGroup.from_raw(2, 2, 3)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    arg_group_2 = PartialArgsGroup(
        ExtendedTypeGroup(RestTypeGroup(UnknownType())),
        LaxOpaqueScope(
            ScopeId(1),
            OptionalValueGroup(
                Optional(Param.from_int(2)),
                Optional(Param.from_int(3)),
            ),
        )
    )
    args_groups = tuple([arg_group, arg_group_2])
    assert current_state == State.from_raw(
        scratchs=scratchs,
        args_groups=args_groups,
    )
    assert last_history_meta == ActionData.from_args(
        action=Optional(DefineArgsGroup(
            StateArgsGroupIndex(2),
            NodeArgIndex(2),
            StateScratchIndex(3),
        )),
        output=Optional(DefineArgsGroupArgOutput(
            StateArgsGroupIndex(2),
            NodeArgIndex(2),
            OptionalContext(Param.from_int(3)),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 18
    action = DefineScratchFromFunctionWithArgs.from_raw(3, 1, 1)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    call_1 = And(
        Param.from_int(1),
        Param.from_int(2),
        IntBoolean.from_int(1),
    )
    scratchs = tuple([And.as_type(), Or.as_type(), call_1])
    assert current_state == State.from_raw(
        scratchs=scratchs,
        args_groups=args_groups,
    )
    assert last_history_meta == ActionData.from_args(
        action=Optional(DefineScratchFromFunctionWithArgs(
            StateScratchIndex(3),
            StateScratchIndex(1),
            Optional(StateArgsGroupIndex(1)),
        )),
        output=Optional(DefineScratchOutput(
            StateScratchIndex(3),
            Optional(call_1),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 19
    action = DefineScratchFromFunctionWithArgs.from_raw(1, 1, 2)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    call_2 = And(
        Param.from_int(2),
        Param.from_int(3),
    )
    scratchs = tuple([call_2, Or.as_type(), call_1])
    assert current_state == State.from_raw(
        scratchs=scratchs,
        args_groups=args_groups,
    )
    assert last_history_meta == ActionData.from_args(
        action=Optional(DefineScratchFromFunctionWithArgs(
            StateScratchIndex(1),
            StateScratchIndex(1),
            Optional(StateArgsGroupIndex(2)),
        )),
        output=Optional(DefineScratchOutput(
            StateScratchIndex(1),
            Optional(call_2),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 20
    action = DefineArgsGroup.from_raw(2, 1, 3)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    arg_group_2 = PartialArgsGroup(
        ExtendedTypeGroup(RestTypeGroup(UnknownType())),
        LaxOpaqueScope(
            ScopeId(1),
            OptionalValueGroup(
                Optional(call_1),
                Optional(Param.from_int(3)),
            ),
        )
    )
    args_groups = tuple([arg_group, arg_group_2])
    assert current_state == State.from_raw(
        scratchs=scratchs,
        args_groups=args_groups
    )
    assert last_history_meta == ActionData.from_args(
        action=Optional(DefineArgsGroup(
            StateArgsGroupIndex(2),
            NodeArgIndex(1),
            StateScratchIndex(3),
        )),
        output=Optional(DefineArgsGroupArgOutput(
            StateArgsGroupIndex(2),
            NodeArgIndex(1),
            OptionalContext(call_1),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 21
    action = DefineArgsGroup.from_raw(2, 2, 1)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    arg_group_2 = PartialArgsGroup(
        ExtendedTypeGroup(RestTypeGroup(UnknownType())),
        LaxOpaqueScope(
            ScopeId(1),
            OptionalValueGroup(
                Optional(call_1),
                Optional(call_2),
            ),
        )
    )
    args_groups = tuple([arg_group, arg_group_2])
    assert current_state == State.from_raw(
        scratchs=scratchs,
        args_groups=args_groups,
    )
    assert last_history_meta == ActionData.from_args(
        action=Optional(DefineArgsGroup(
            StateArgsGroupIndex(2),
            NodeArgIndex(2),
            StateScratchIndex(1),
        )),
        output=Optional(DefineArgsGroupArgOutput(
            StateArgsGroupIndex(2),
            NodeArgIndex(2),
            OptionalContext(call_2),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 22
    action = DefineScratchFromFunctionWithArgs.from_raw(1, 2, 2)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    call_3 = Or(
        call_1,
        call_2,
    )
    scratchs = tuple([call_3, Or.as_type(), call_1])

    assert current_state == State.from_raw(
        scratchs=scratchs,
        args_groups=args_groups,
    )
    assert last_history_meta == ActionData.from_args(
        action=Optional(DefineScratchFromFunctionWithArgs(
            StateScratchIndex(1),
            StateScratchIndex(2),
            Optional(StateArgsGroupIndex(2)),
        )),
        output=Optional(DefineScratchOutput(
            StateScratchIndex(1),
            Optional(call_3),
        )),
        exception=Optional(),
    )
    assert env.full_state.goal_achieved() is False

    # Test case 23
    action = DeleteArgsGroupOutput.from_raw(1, 0, 0)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    args_groups = tuple([arg_group_2])
    assert current_state == State.from_raw(
        scratchs=scratchs,
        args_groups=args_groups,
    )
    assert last_history_meta == ActionData.from_args(
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
    last_history_meta = get_last_history_meta(env)
    args_groups = tuple()
    assert current_state == State.from_raw(
        scratchs=scratchs,
        args_groups=args_groups
    )
    assert last_history_meta == ActionData.from_args(
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
    last_history_meta = get_last_history_meta(env)
    scratchs = tuple([call_3, Or.as_type()])
    assert current_state == State.from_raw(
        scratchs=scratchs,
        args_groups=args_groups,
    )
    assert last_history_meta == ActionData.from_args(
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
    last_history_meta = get_last_history_meta(env)
    scratchs = tuple([call_3])
    assert current_state == State.from_raw(
        scratchs=scratchs,
        args_groups=args_groups,
    )
    assert last_history_meta == ActionData.from_args(
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
    meta_idx = get_single_child_type_index(FunctionExpr, env)
    action = DefineScratchFromSingleArg.from_raw(1, meta_idx, 1)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    content = current_state.nested_args((
        State.idx_scratch_group,
        1,
        Scratch.idx_child,
        Optional.idx_value,
    )).apply()
    assert content == goal.definition_expr.apply()
    verification = goal.evaluate(current_state, StateScratchIndex(1))
    verification.raise_on_false()

    # Test case 28
    action = CreateScratch.create()
    env.step(action)
    meta_idx = get_from_int_type_index(Param, env)
    action = DefineScratchFromInt.from_raw(2, meta_idx, 4)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    verification = goal.evaluate(current_state, StateScratchIndex(2))
    assert verification.as_bool is False

    # Test case 29
    action = DeleteScratchOutput.from_raw(2, 0, 0)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    meta_idx = get_from_int_type_index(StateScratchIndex, env)
    action = VerifyGoal.from_raw(0, meta_idx, 1)
    env.step(action)
    current_state = get_current_state(env)
    last_history_meta = get_last_history_meta(env)
    scratch_goal = FunctionExpr(ExtendedTypeGroup.rest(), call_3)
    assert current_state == State.from_raw(
        meta_info=StateMetaInfo.create_with_goal(GoalAchieved.achieved()),
        scratchs=[scratch_goal],
        args_groups=args_groups,
    )
    assert last_history_meta == ActionData.from_args(
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

def test():
    basic_test()
