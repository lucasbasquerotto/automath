import typing
from abc import ABC
from environment.core import (
    INode,
    InheritableNode,
    IDefault,
    BaseGroup,
    IFromInt,
    IInt,
    INodeIndex,
    NodeMainIndex,
    NodeArgIndex,
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
from environment.state import (
    State,
    Scratch,
    PartialArgsGroup,
    StateDefinition)
from environment.meta_env import (
    MetaInfo,
    IMetaData,
    IFullState,
    IFullStateIndex,
    IFullStateIntIndex,
    IAction,
    IBasicAction,
    SubtypeOuterGroup,
    IActionOutput)

T = typing.TypeVar('T', bound=INode)

###########################################################
################# FULL STATE DEFINITIONS ##################
###########################################################

class HistoryNode(InheritableNode, IInstantiable):

    idx_state = 0
    idx_meta_data = 1

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            State,
            IOptional[IMetaData],
        ]))

class HistoryGroupNode(BaseGroup[HistoryNode], IInstantiable):

    idx_state_nodes = 0

    @classmethod
    def item_type(cls) -> type[HistoryNode]:
        return HistoryNode

###########################################################
####################### FULL STATE ########################
###########################################################

class FullState(InheritableNode, IFullState, IInstantiable):

    idx_meta = 0
    idx_current = 1
    idx_history = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            MetaInfo,
            HistoryNode,
            HistoryGroupNode,
        ]))

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

class FullStateMainIndex(NodeMainIndex, IInstantiable):
    pass

class FullStateArgIndex(NodeArgIndex, IInstantiable):
    pass

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

class MetaAllowedActionsTypeIndex(
    FullStateGroupTypeBaseIndex[IAction[FullState, O]],
    typing.Generic[O],
    IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IAction

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArgs:
        return full_state.nested_args((
            FullState.idx_meta,
            MetaInfo.idx_allowed_actions,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaBasicActionsTypeIndex(
    FullStateGroupTypeBaseIndex[IBasicAction[FullState, O]],
    typing.Generic[O],
    IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IAction

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

class FullStateNode(InheritableNode, ABC):

    idx_state_type = 0
    idx_state_value = 1

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            TypeNode,
            State,
        ]))

class FullStateGroup(BaseGroup[FullStateNode], IInstantiable):

    idx_state_nodes = 0

    @classmethod
    def item_type(cls):
        return FullStateNode

    @classmethod
    def from_nodes(cls, nodes: typing.Sequence[FullStateNode]) -> typing.Self:
        return cls.from_items(nodes)

    def to_state_nodes(self) -> typing.Sequence[FullStateNode]:
        return typing.cast(typing.Sequence[FullStateNode], self.args[self.idx_state_nodes])
