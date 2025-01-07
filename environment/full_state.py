import typing
from abc import ABC
from environment.core import (
    INode,
    InheritableNode,
    BaseInt,
    IDefault,
    BaseGroup,
    ITypedIndex,
    ITypedIntIndex,
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
    IInstantiable)
from environment.state import (
    State,
    Scratch,
    ScratchGroup,
    PartialArgsGroup,
    PartialArgsOuterGroup,
    StateDefinition,
    StateDefinitionGroup)
from environment.meta_env import MetaInfo, IMetaData
from environment.action import BaseAction


T = typing.TypeVar('T', bound=INode)

###########################################################
################# FULL STATE DEFINITIONS ##################
###########################################################

class HistoryNode(InheritableNode, IInstantiable):
    def __init__(self, state: State, meta_data: IOptional[IMetaData]):
        super().__init__(state, meta_data)

    @property
    def state(self) -> State:
        state = self.args[0]
        return typing.cast(State, state)

    @property
    def meta_data(self) -> IOptional[IMetaData]:
        meta_data = self.args[1]
        return typing.cast(IOptional[IMetaData], meta_data)

class HistoryGroupNode(BaseGroup[HistoryNode], IInstantiable):

    @classmethod
    def item_type(cls) -> type[HistoryNode]:
        return HistoryNode

###########################################################
####################### FULL STATE ########################
###########################################################

class FullState(InheritableNode, IInstantiable):
    def __init__(self, meta: MetaInfo, current: HistoryNode, history: HistoryGroupNode):
        super().__init__(meta, current, history)

    @property
    def meta(self) -> MetaInfo:
        meta = self.args[0]
        return typing.cast(MetaInfo, meta)

    @property
    def current(self) -> HistoryNode:
        current = self.args[1]
        return typing.cast(HistoryNode, current)

    @property
    def history(self) -> HistoryGroupNode:
        history = self.args[2]
        return typing.cast(HistoryGroupNode, history)

###########################################################
###################### MAIN INDICES #######################
###########################################################

class IFullStateIndex(ITypedIndex[FullState, T], typing.Generic[T], ABC):

    @classmethod
    def outer_type(cls) -> type[FullState]:
        return FullState

    @classmethod
    def item_type(cls) -> type[T]:
        raise NotImplementedError

class FullStateIntIndex(
    BaseInt,
    IFullStateIndex[T],
    ITypedIntIndex[FullState, T],
    typing.Generic[T],
    ABC,
):
    pass

class FullStateMainIndex(NodeMainIndex, IInstantiable):
    pass

class FullStateArgIndex(NodeArgIndex, IInstantiable):
    pass

class FullStateGroupBaseIndex(FullStateIntIndex[T], ABC):

    @classmethod
    def group(cls, full_state: FullState) -> BaseGroup[T]:
        raise NotImplementedError

    def find_in_outer_node(self, node: FullState):
        return self.find_arg(self.group(node))

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
    def group(cls, full_state: FullState):
        return full_state.meta.default_group.subtypes

class MetaFromIntTypeIndex(FullStateGroupTypeBaseIndex[IFromInt], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IFromInt

    @classmethod
    def group(cls, full_state: FullState):
        return full_state.meta.from_int_group.subtypes

class MetaIntTypeIndex(FullStateGroupTypeBaseIndex[IInt], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IInt

    @classmethod
    def group(cls, full_state: FullState):
        return full_state.meta.int_group.subtypes

class MetaNodeIndexTypeIndex(FullStateGroupTypeBaseIndex[INodeIndex], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return INodeIndex

    @classmethod
    def group(cls, full_state: FullState):
        return full_state.meta.node_index_group.subtypes

class MetaFullStateIndexTypeIndex(FullStateGroupTypeBaseIndex[IFullStateIndex], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IFullStateIndex

    @classmethod
    def group(cls, full_state: FullState):
        return full_state.meta.full_state_index_group.subtypes

class MetaFullStateIntIndexTypeIndex(FullStateGroupTypeBaseIndex[FullStateIntIndex], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IFullStateIndex

    @classmethod
    def group(cls, full_state: FullState):
        return full_state.meta.full_state_int_index_group.subtypes

class MetaSingleChildTypeIndex(FullStateGroupTypeBaseIndex[IFromSingleChild], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IFromSingleChild

    @classmethod
    def group(cls, full_state: FullState):
        return full_state.meta.single_child_group.subtypes

class MetaGroupTypeIndex(FullStateGroupTypeBaseIndex[IGroup], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IGroup

    @classmethod
    def group(cls, full_state: FullState):
        return full_state.meta.group_outer_group.subtypes

class MetaFunctionTypeIndex(FullStateGroupTypeBaseIndex[IFunction], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IFunction

    @classmethod
    def group(cls, full_state: FullState):
        return full_state.meta.function_group.subtypes

class MetaBooleanTypeIndex(FullStateGroupTypeBaseIndex[IBoolean], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IBoolean

    @classmethod
    def group(cls, full_state: FullState):
        return full_state.meta.boolean_group.subtypes

class MetaAllowedActionsTypeIndex(FullStateGroupTypeBaseIndex[BaseAction], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return BaseAction

    @classmethod
    def group(cls, full_state: FullState):
        return full_state.meta.allowed_actions.subtypes

###########################################################
################## CURRENT STATE INDICES ##################
###########################################################

class CurrentStateScratchIndex(FullStateReadonlyGroupBaseIndex[Scratch], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return Scratch

    @classmethod
    def group(cls, full_state: FullState) -> ScratchGroup:
        return full_state.current.state.scratch_group

class CurrentStateArgsOuterGroupIndex(FullStateReadonlyGroupBaseIndex[PartialArgsGroup], IInstantiable):

    @classmethod
    def item_type(cls):
        return PartialArgsGroup

    @classmethod
    def group(cls, full_state: FullState) -> PartialArgsOuterGroup:
        return full_state.current.state.args_outer_group

class CurrentStateDefinitionIndex(FullStateReadonlyGroupBaseIndex[StateDefinition], IInstantiable):

    @classmethod
    def item_type(cls):
        return StateDefinition

    @classmethod
    def group(cls, full_state: FullState) -> StateDefinitionGroup:
        return full_state.current.state.definition_group
