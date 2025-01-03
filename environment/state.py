import typing
from abc import ABC
from environment.core import (
    INode,
    IFunction,
    InheritableNode,
    BaseGroup,
    NodeIntBaseIndex,
    NodeMainIndex,
    NodeArgIndex,
    OpaqueScope,
    Integer,
    ScopeId,
    OptionalValueGroup,
    ExtendedTypeGroup,
    RestTypeGroup,
    UnknownType,
    FunctionExpr,
    SimpleScope,
    ITypedIndex,
    ITypedIntIndex,
)

T = typing.TypeVar('T', bound=INode)
K = typing.TypeVar('K', bound=INode)

###########################################################
#################### STATE DEFINITIONS ####################
###########################################################

class Scratch(OpaqueScope[INode]):
    def __init__(self, id: ScopeId, child: INode):
        assert isinstance(child, INode)
        super().__init__(id, child)

class ScratchGroup(BaseGroup[Scratch]):
    @classmethod
    def item_type(cls):
        return OpaqueScope[Scratch]

    @classmethod
    def from_raw_items(cls, items: tuple[INode, ...]) -> typing.Self:
        return cls.from_items([Scratch.with_content(s) for s in items])

    def to_raw_items(self) -> tuple[INode, ...]:
        return tuple(s.child for s in self.as_tuple)

class PartialArgsGroup(FunctionExpr[OptionalValueGroup[T]], typing.Generic[T]):

    def __init__(
        self,
        param_type_group: ExtendedTypeGroup,
        scope: OpaqueScope[OptionalValueGroup[T]],
    ):
        assert isinstance(param_type_group, ExtendedTypeGroup)
        assert isinstance(scope, OpaqueScope)
        assert isinstance(scope.child, OptionalValueGroup)
        super().__init__(param_type_group, scope)

    @property
    def param_type_group(self) -> ExtendedTypeGroup:
        param_type_group = self.args[0]
        assert isinstance(param_type_group, ExtendedTypeGroup)
        return param_type_group

    @property
    def scope(self) -> OpaqueScope[OptionalValueGroup[T]]:
        scope = self.args[1]
        assert isinstance(scope, OpaqueScope)
        assert isinstance(scope.child, OptionalValueGroup)
        return scope

    @property
    def inner_args_group(self) -> OptionalValueGroup[T]:
        inner_args = self.scope.child
        return typing.cast(OptionalValueGroup[T], inner_args)

    @property
    def inner_args(self) -> tuple[INode, ...]:
        return self.inner_args_group.as_tuple

    @classmethod
    def default(cls) -> typing.Self:
        return cls(
            ExtendedTypeGroup(RestTypeGroup(UnknownType())),
            OpaqueScope.with_content(OptionalValueGroup()))

class PartialArgsOuterGroup(BaseGroup[PartialArgsGroup]):
    @classmethod
    def item_type(cls):
        return PartialArgsGroup

class IDefinitionKey(ABC, INode):
    pass

D = typing.TypeVar('D', bound=IDefinitionKey)

class FunctionId(Integer, IDefinitionKey, IFunction):
    pass

class StateDefinition(InheritableNode, typing.Generic[D, T]):

    def __init__(self, definition_key: D, definition_expr: T):
        super().__init__(definition_key, definition_expr)

    @property
    def definition_key(self) -> D:
        definition_key = self.args[0]
        return typing.cast(D, definition_key)

    @property
    def definition_expr(self) -> T:
        definition_expr = self.args[1]
        return typing.cast(T, definition_expr)

class FunctionDefinition(StateDefinition[FunctionId, FunctionExpr[T]], typing.Generic[T]):

    @property
    def scope(self) -> SimpleScope[T]:
        return self.definition_expr.scope

class StateDefinitionGroup(BaseGroup[StateDefinition]):
    @classmethod
    def item_type(cls):
        return StateDefinition

###########################################################
########################## STATE ##########################
###########################################################

class State(InheritableNode):
    def __init__(
        self,
        definition_group: StateDefinitionGroup,
        args_outer_group: PartialArgsOuterGroup,
        scratch_group: ScratchGroup,
    ):
        super().__init__(definition_group, args_outer_group, scratch_group)

    @property
    def definition_group(self) -> StateDefinitionGroup:
        definition_group = self.args[0]
        return typing.cast(StateDefinitionGroup, definition_group)

    @property
    def args_outer_group(self) -> PartialArgsOuterGroup:
        args_outer_group = self.args[2]
        assert isinstance(args_outer_group, PartialArgsOuterGroup)
        return args_outer_group

    @property
    def scratch_group(self) -> ScratchGroup:
        scratch_group = self.args[1]
        assert isinstance(scratch_group, ScratchGroup)
        return scratch_group

    @classmethod
    def from_raw(
        cls,
        definitions: tuple[StateDefinition, ...],
        args_outer_groups: tuple[PartialArgsGroup, ...],
        scratchs: tuple[INode, ...],
    ) -> typing.Self:
        return cls(
            StateDefinitionGroup.from_items(definitions),
            PartialArgsOuterGroup.from_items(args_outer_groups),
            ScratchGroup.from_raw_items(scratchs))

###########################################################
######################### INDICES #########################
###########################################################

class IStateIndex(ITypedIndex[State, T], typing.Generic[T]):

    @classmethod
    def outer_type(cls) -> type[State]:
        return State

    @classmethod
    def item_type(cls) -> type[T]:
        raise NotImplementedError

    def find_in_outer_node(self, node: State) -> T | None:
        raise NotImplementedError

    def replace_in_outer_target(self, target: State, new_node: T) -> State | None:
        raise NotImplementedError

class StateIntIndex(ABC, Integer, IStateIndex[T], ITypedIntIndex[State, T], typing.Generic[T]):
    pass

class StateDefinitionIndex(StateIntIndex[StateDefinition]):

    @classmethod
    def item_type(cls):
        return StateDefinition

    def find_in_outer_node(self, node: State) -> StateDefinition | None:
        return self.find_arg(node.definition_group)

    def replace_in_outer_target(self, target: State, new_node: StateDefinition) -> State | None:
        definitions = list(target.definition_group.as_tuple)
        for i, definition in enumerate(definitions):
            if self.value == (i+1):
                assert new_node.definition_key == definition.definition_key
                definitions[i] = new_node
                return State(
                    definition_group=StateDefinitionGroup.from_items(definitions),
                    args_outer_group=target.args_outer_group,
                    scratch_group=target.scratch_group)
        return target

class StateScratchIndex(StateIntIndex[Scratch]):

    @classmethod
    def item_type(cls):
        return Scratch

    def find_in_outer_node(self, node: State) -> Scratch | None:
        return self.find_arg(node.scratch_group)

    def replace_in_outer_target(self, target: State, new_node: Scratch) -> State | None:
        group = self.replace_arg(target.scratch_group, new_node)
        if group is None:
            return None
        return State(
            definition_group=target.definition_group,
            args_outer_group=target.args_outer_group,
            scratch_group=group)

class ScratchNodeIndex(NodeIntBaseIndex):

    def find_in_node(self, node: INode) -> INode | None:
        assert isinstance(node, Scratch)
        return NodeMainIndex(self.to_int).find_in_node(node.child)

    def replace_in_target(self, target_node: INode, new_node: INode) -> Scratch | None:
        assert isinstance(target_node, Scratch)
        child = NodeMainIndex(self.to_int).replace_in_target(target_node.child, new_node)
        if child is None:
            return None
        return Scratch(target_node.id, child)

class StateArgsGroupIndex(StateIntIndex[PartialArgsGroup]):

    @classmethod
    def item_type(cls):
        return PartialArgsGroup

    def find_in_outer_node(self, node: State) -> PartialArgsGroup | None:
        return self.find_arg(node.args_outer_group)

    def replace_in_outer_target(self, target: State, new_node: PartialArgsGroup) -> State | None:
        group = self.replace_arg(target.args_outer_group, new_node)
        if group is None:
            return None
        return State(
            definition_group=target.definition_group,
            args_outer_group=group,
            scratch_group=target.scratch_group)

class StateArgsGroupArgIndex(InheritableNode, IStateIndex[INode]):

    @classmethod
    def item_type(cls):
        return INode

    def __init__(self, group_index: StateArgsGroupIndex, arg_index: NodeArgIndex):
        assert isinstance(group_index, StateArgsGroupIndex)
        assert isinstance(arg_index, NodeArgIndex)
        super().__init__(group_index, arg_index)

    @property
    def group_index(self) -> StateArgsGroupIndex:
        group_index = self.args[0]
        assert isinstance(group_index, StateArgsGroupIndex)
        return group_index

    @property
    def arg_index(self) -> NodeArgIndex:
        arg_index = self.args[1]
        assert isinstance(arg_index, NodeArgIndex)
        return arg_index

    def find_in_outer_node(self, node: State) -> INode | None:
        group = self.group_index.find_in_outer_node(node)
        if group is None:
            return None
        return group[self.arg_index]

    def replace_in_outer_target(self, target: State, new_node: INode) -> State | None:
        group = self.group_index.find_in_outer_node(target)
        if group is None:
            return target
        new_group = self.arg_index.replace_in_target(
            target_node=group,
            new_node=new_node)
        if new_group is None:
            return None
        return self.group_index.replace_in_outer_target(target, new_group)
