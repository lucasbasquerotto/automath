import typing
from abc import ABC
from env.core import (
    INode,
    InheritableNode,
    IDefault,
    BaseGroup,
    IFromInt,
    IInt,
    INodeIndex,
    NodeMainBaseIndex,
    NodeArgBaseIndex,
    IFromSingleChild,
    IGroup,
    IFunction,
    IBoolean,
    TypeNode,
    IOptional,
    Optional,
    ExtendedTypeGroup,
    CountableTypeGroup,
    TmpNestedArg,
    TmpNestedArgs,
    IInstantiable)
from env.state import (
    State,
    Scratch,
    PartialArgsGroup,
    GoalAchieved,
    GoalAchievedGroup,
    StateDefinition)
from env.meta_env import (
    MetaInfo,
    IMetaData,
    IFullState,
    IFullStateIndex,
    IFullStateIntIndex,
    IAction,
    IBasicAction,
    SubtypeOuterGroup,
    DetailedType,
    GeneralTypeGroup,
    IGoal,
    Goal,
    GoalGroup,
    IActionOutput)

T = typing.TypeVar('T', bound=INode)

###########################################################
################# FULL STATE DEFINITIONS ##################
###########################################################

class HistoryNode(InheritableNode, IDefault, IInstantiable):

    idx_state = 1
    idx_meta_data = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            State,
            IOptional[IMetaData],
        ]))

    @classmethod
    def create(cls):
        return cls(State.create(), Optional.create())

    @classmethod
    def _create_goal_achieved_with_goal(cls, goal: IGoal):
        if isinstance(goal, Goal):
            return GoalAchieved.create()
        if isinstance(goal, GoalGroup):
            return GoalAchievedGroup(*[
                cls._create_goal_achieved_with_goal(sub_goal)
                for sub_goal in goal.as_tuple
            ])
        raise NotImplementedError(type(goal))

    @classmethod
    def create_with_goal(cls, goal: IGoal):
        goal_achieved = cls._create_goal_achieved_with_goal(goal)
        return cls(State.create_with_goal(goal_achieved), Optional.create())

    @property
    def state(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_state)

    @property
    def meta_data(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_meta_data)

class HistoryGroupNode(BaseGroup[HistoryNode], IInstantiable):

    @classmethod
    def item_type(cls) -> type[HistoryNode]:
        return HistoryNode

###########################################################
####################### FULL STATE ########################
###########################################################

class FullState(InheritableNode, IFullState, IFromSingleChild[MetaInfo], IInstantiable):

    idx_meta = 1
    idx_current = 2
    idx_history = 3

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            MetaInfo,
            HistoryNode,
            HistoryGroupNode,
        ]))

    @classmethod
    def with_child(cls, child: MetaInfo) -> typing.Self:
        goal = child.goal.apply().cast(IGoal)
        return cls.new(child, HistoryNode.create_with_goal(goal), HistoryGroupNode())

    @property
    def meta(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_meta)

    @property
    def current(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_current)

    @property
    def history(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_history)

    @property
    def current_state(self) -> TmpNestedArgs:
        return self.nested_args(
            (self.idx_current, HistoryNode.idx_state)
        )

    def goal_achieved(self) -> bool:
        state = self.current_state.apply().cast(State)
        return state.goal_achieved()

    def node_types(self) -> tuple[type[INode], ...]:
        meta = self.meta.apply().cast(MetaInfo)
        all_types_wrappers = meta.all_types.apply().cast(GeneralTypeGroup).as_tuple
        all_types = tuple(wrapper.type for wrapper in all_types_wrappers)
        return all_types

###########################################################
###################### MAIN INDICES #######################
###########################################################

class FullStateIndex(IFullStateIndex[FullState, T], typing.Generic[T], ABC):

    @classmethod
    def outer_type(cls):
        return FullState

class FullStateIntIndex(IFullStateIntIndex[FullState, T], typing.Generic[T], ABC):

    @classmethod
    def outer_type(cls):
        return FullState

class FullStateMainIndex(NodeMainBaseIndex, IFullStateIndex[FullState, INode], IInstantiable):

    @classmethod
    def outer_type(cls):
        return FullState

    def find_in_outer_node(self, node: FullState):
        return self.find_in_node(node)

    def replace_in_outer_target(self, target: FullState, new_node: INode):
        return self.replace_in_target(target, new_node)

class FullStateArgIndex(NodeArgBaseIndex, IFullStateIndex[FullState, INode], IInstantiable):

    @classmethod
    def outer_type(cls):
        return FullState

    def find_in_outer_node(self, node: FullState):
        return self.find_in_node(node)

    def replace_in_outer_target(self, target: FullState, new_node: INode):
        return self.replace_in_target(target, new_node)

class FullStateGroupBaseIndex(FullStateIntIndex[T], ABC):

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArgs:
        raise NotImplementedError

    def find_in_outer_node(self, node: FullState):
        return self.find_arg(self.group(node).apply())

    def replace_in_outer_target(self, target: FullState, new_node: T):
        raise NotImplementedError

class FullStateReadonlyGroupBaseIndex(FullStateGroupBaseIndex[T], ABC):

    def replace_in_outer_target(self, target: FullState, new_node: T):
        return Optional.create()

class FullStateGroupTypeBaseIndex(FullStateReadonlyGroupBaseIndex[TypeNode[T]], ABC):

    @classmethod
    def item_type(cls) -> type[TypeNode[T]]:
        return TypeNode

    @classmethod
    def inner_item_type(cls) -> type[T]:
        raise NotImplementedError

###########################################################
###################### META INDICES #######################
###########################################################

class MetaAllTypesTypeIndex(
    FullStateGroupTypeBaseIndex[INode],
    IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return INode

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArgs:
        return full_state.nested_args((
            FullState.idx_meta,
            MetaInfo.idx_all_types,
        ))

class MetaTypesDetailsTypeIndex(
    FullStateGroupTypeBaseIndex[DetailedType],
    IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return DetailedType

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArgs:
        return full_state.nested_args((
            FullState.idx_meta,
            MetaInfo.idx_all_types_details,
        ))

class MetaAllowedBasicActionsTypeIndex(
    FullStateGroupTypeBaseIndex[IBasicAction[FullState]],
    IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IBasicAction

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArgs:
        return full_state.nested_args((
            FullState.idx_meta,
            MetaInfo.idx_allowed_basic_actions,
        ))

class MetaAllowedActionsTypeIndex(
    FullStateGroupTypeBaseIndex[IAction[FullState]],
    IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IAction

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArgs:
        return full_state.nested_args((
            FullState.idx_meta,
            MetaInfo.idx_allowed_actions,
        ))

class MetaDefaultTypeIndex(FullStateGroupTypeBaseIndex[IDefault], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IDefault

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArgs:
        return full_state.nested_args((
            FullState.idx_meta,
            MetaInfo.idx_default_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaFromIntTypeIndex(FullStateGroupTypeBaseIndex[IFromInt], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IFromInt

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArgs:
        return full_state.nested_args((
            FullState.idx_meta,
            MetaInfo.idx_from_int_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaIntTypeIndex(FullStateGroupTypeBaseIndex[IInt], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IInt

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArgs:
        return full_state.nested_args((
            FullState.idx_meta,
            MetaInfo.idx_int_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaNodeIndexTypeIndex(FullStateGroupTypeBaseIndex[INodeIndex], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return INodeIndex

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArgs:
        return full_state.nested_args((
            FullState.idx_meta,
            MetaInfo.idx_node_index_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaFullStateIndexTypeIndex(FullStateGroupTypeBaseIndex[FullStateIndex], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return FullStateIndex

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArgs:
        return full_state.nested_args((
            FullState.idx_meta,
            MetaInfo.idx_full_state_index_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaFullStateIntIndexTypeIndex(FullStateGroupTypeBaseIndex[FullStateIntIndex], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return FullStateIntIndex

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArgs:
        return full_state.nested_args((
            FullState.idx_meta,
            MetaInfo.idx_full_state_int_index_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaSingleChildTypeIndex(FullStateGroupTypeBaseIndex[IFromSingleChild], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IFromSingleChild

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArgs:
        return full_state.nested_args((
            FullState.idx_meta,
            MetaInfo.idx_single_child_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaGroupTypeIndex(FullStateGroupTypeBaseIndex[IGroup], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IGroup

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArgs:
        return full_state.nested_args((
            FullState.idx_meta,
            MetaInfo.idx_group_outer_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaFunctionTypeIndex(FullStateGroupTypeBaseIndex[IFunction], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IFunction

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArgs:
        return full_state.nested_args((
            FullState.idx_meta,
            MetaInfo.idx_function_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaBooleanTypeIndex(FullStateGroupTypeBaseIndex[IBoolean], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IBoolean

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArgs:
        return full_state.nested_args((
            FullState.idx_meta,
            MetaInfo.idx_boolean_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

O = typing.TypeVar('O', bound=IActionOutput)

class MetaAllActionsTypeIndex(
    FullStateGroupTypeBaseIndex[IAction[FullState]],
    IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IAction

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArgs:
        return full_state.nested_args((
            FullState.idx_meta,
            MetaInfo.idx_all_actions,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaBasicActionsTypeIndex(
    FullStateGroupTypeBaseIndex[IBasicAction[FullState]],
    IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IBasicAction

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArgs:
        return full_state.nested_args((
            FullState.idx_meta,
            MetaInfo.idx_basic_actions,
            SubtypeOuterGroup.idx_subtypes,
        ))

###########################################################
################## CURRENT STATE INDICES ##################
###########################################################

class CurrentStateScratchIndex(FullStateReadonlyGroupBaseIndex[Scratch], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return Scratch

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArgs:
        return full_state.nested_args((
            FullState.idx_current,
            HistoryNode.idx_state,
            State.idx_scratch_group,
        ))

class CurrentStateArgsOuterGroupIndex(
    FullStateReadonlyGroupBaseIndex[PartialArgsGroup],
    IInstantiable,
):

    @classmethod
    def item_type(cls):
        return PartialArgsGroup

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArgs:
        return full_state.nested_args((
            FullState.idx_current,
            HistoryNode.idx_state,
            State.idx_args_outer_group,
        ))

class CurrentStateDefinitionIndex(FullStateReadonlyGroupBaseIndex[StateDefinition], IInstantiable):

    @classmethod
    def item_type(cls):
        return StateDefinition

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArgs:
        return full_state.nested_args((
            FullState.idx_current,
            HistoryNode.idx_state,
            State.idx_definition_group,
        ))
