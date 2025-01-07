import typing
from abc import ABC
from environment.core import (
    INode,
    IDefault,
    IFunction,
    IOptional,
    Optional,
    InheritableNode,
    BaseGroup,
    NodeIntBaseIndex,
    NodeMainIndex,
    NodeArgIndex,
    OpaqueScope,
    BaseInt,
    ScopeId,
    OptionalValueGroup,
    ExtendedTypeGroup,
    RestTypeGroup,
    UnknownType,
    FunctionExpr,
    SimpleScope,
    ITypedIndex,
    ITypedIntIndex,
    IInstantiable,
)

T = typing.TypeVar('T', bound=INode)
K = typing.TypeVar('K', bound=INode)

###########################################################
#################### STATE DEFINITIONS ####################
###########################################################

class Scratch(OpaqueScope[IOptional[INode]], IInstantiable):
    def __init__(self, id: ScopeId, child: IOptional[INode]):
        super().__init__(id, child)

class ScratchGroup(BaseGroup[Scratch], IInstantiable):
    @classmethod
    def item_type(cls):
        return OpaqueScope[Scratch]

    @classmethod
    def from_raw_items(cls, items: tuple[INode | None, ...]) -> typing.Self:
        return cls.from_items([Scratch.with_content(Optional(s)) for s in items])

    def to_raw_items(self) -> tuple[INode, ...]:
        return tuple(s.child for s in self.as_tuple)

class PartialArgsGroup(
    FunctionExpr[OptionalValueGroup[T]], IDefault, typing.Generic[T],
    IInstantiable,
):

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
    def create(cls) -> typing.Self:
        return cls(
            ExtendedTypeGroup(RestTypeGroup(UnknownType())),
            OpaqueScope.with_content(OptionalValueGroup()))

class PartialArgsOuterGroup(BaseGroup[PartialArgsGroup], IInstantiable):
    @classmethod
    def item_type(cls):
        return PartialArgsGroup

class IDefinitionKey(INode, ABC):
    pass

D = typing.TypeVar('D', bound=IDefinitionKey)

class FunctionId(BaseInt, IDefinitionKey, IFunction, IInstantiable):
    pass

class StateDefinition(InheritableNode, typing.Generic[D, T], ABC):

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

class FunctionDefinition(
    StateDefinition[FunctionId, FunctionExpr[T]],
    typing.Generic[T],
    IInstantiable,
):

    @property
    def scope(self) -> SimpleScope[T]:
        return self.definition_expr.scope

class StateDefinitionGroup(BaseGroup[StateDefinition], IInstantiable):
    @classmethod
    def item_type(cls):
        return StateDefinition

###########################################################
########################## STATE ##########################
###########################################################

class State(InheritableNode, IInstantiable):
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

class IStateIndex(ITypedIndex[State, T], typing.Generic[T], ABC):

    @classmethod
    def outer_type(cls) -> type[State]:
        return State

    @classmethod
    def item_type(cls) -> type[T]:
        raise NotImplementedError

    def find_in_outer_node(self, node: State) -> IOptional[T]:
        raise NotImplementedError

    def replace_in_outer_target(self, target: State, new_node: T) -> IOptional[State]:
        raise NotImplementedError

    def remove_in_outer_target(self, target: State) -> IOptional[State]:
        raise NotImplementedError

class StateIntIndex(BaseInt, IStateIndex[T], ITypedIntIndex[State, T], typing.Generic[T], ABC):
    pass

class StateDefinitionIndex(StateIntIndex[StateDefinition], IInstantiable):

    @classmethod
    def item_type(cls):
        return StateDefinition

    def find_in_outer_node(self, node: State):
        return self.find_arg(node.definition_group)

    def replace_in_outer_target(self, target: State, new_node: StateDefinition):
        definitions = list(target.definition_group.as_tuple)
        for i, definition in enumerate(definitions):
            if self.value == (i+1):
                assert new_node.definition_key == definition.definition_key
                definitions[i] = new_node
                return Optional(State(
                    definition_group=StateDefinitionGroup.from_items(definitions),
                    args_outer_group=target.args_outer_group,
                    scratch_group=target.scratch_group))
        return Optional.create()

    def remove_in_outer_target(self, target: State):
        definitions = list(target.definition_group.as_tuple)
        for i, _ in enumerate(definitions):
            if self.value == (i+1):
                definitions.pop(i)
                return Optional(State(
                    definition_group=StateDefinitionGroup.from_items(definitions),
                    args_outer_group=target.args_outer_group,
                    scratch_group=target.scratch_group))
        return Optional.create()

class StateScratchIndex(StateIntIndex[Scratch], IInstantiable):

    @classmethod
    def item_type(cls):
        return Scratch

    def find_in_outer_node(self, node: State):
        return self.find_arg(node.scratch_group)

    def replace_in_outer_target(self, target: State, new_node: Scratch):
        group = self.replace_arg(target.scratch_group, new_node).value
        if group is None:
            return Optional.create()
        return Optional(State(
            definition_group=target.definition_group,
            args_outer_group=target.args_outer_group,
            scratch_group=group))

    def remove_in_outer_target(self, target: State):
        group = self.remove_arg(target.scratch_group).value
        if group is None:
            return Optional.create()
        return Optional(State(
            definition_group=target.definition_group,
            args_outer_group=target.args_outer_group,
            scratch_group=group))

class ScratchNodeIndex(NodeIntBaseIndex, IInstantiable):

    def find_in_node(self, node: INode):
        assert isinstance(node, Scratch)
        content = node.child.value
        if content is None:
            return Optional.create()
        return NodeMainIndex(self.to_int).find_in_node(content)

    def replace_in_target(self, target_node: INode, new_node: INode):
        assert isinstance(target_node, Scratch)
        assert isinstance(new_node, INode)
        if self.to_int == 1:
            return Optional(new_node)
        old_content = target_node.child.value_or_raise
        content = NodeMainIndex(self.to_int).replace_in_target(old_content, new_node)
        return content

    def remove_in_target(self, target_node: INode):
        assert isinstance(target_node, Scratch)
        if self.to_int == 1:
            return Optional.create()
        old_content = target_node.child.value_or_raise
        content = NodeMainIndex(self.to_int).remove_in_target(old_content)
        return content

class StateArgsGroupIndex(StateIntIndex[PartialArgsGroup], IInstantiable):

    @classmethod
    def item_type(cls):
        return PartialArgsGroup

    def find_in_outer_node(self, node: State):
        return self.find_arg(node.args_outer_group)

    def replace_in_outer_target(self, target: State, new_node: PartialArgsGroup):
        group = self.replace_arg(target.args_outer_group, new_node)
        args_outer_group = group.value
        if args_outer_group is None:
            return Optional.create()
        return State(
            definition_group=target.definition_group,
            args_outer_group=args_outer_group,
            scratch_group=target.scratch_group)

    def remove_in_outer_target(self, target: State):
        group = self.remove_arg(target.args_outer_group)
        args_outer_group = group.value
        if args_outer_group is None:
            return Optional.create()
        return State(
            definition_group=target.definition_group,
            args_outer_group=args_outer_group,
            scratch_group=target.scratch_group)

class StateArgsGroupArgIndex(InheritableNode, IStateIndex[INode], IInstantiable):

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

    def find_in_outer_node(self, node: State):
        group = self.group_index.find_in_outer_node(node).value
        if group is None:
            return Optional.create()
        return Optional(group[self.arg_index])

    def replace_in_outer_target(self, target: State, new_node: INode):
        group = self.group_index.find_in_outer_node(target).value
        if group is None:
            return Optional.create()
        new_group = self.arg_index.replace_in_target(
            target_node=group,
            new_node=new_node,
        ).value
        if new_group is None:
            return Optional.create()
        result = self.group_index.replace_in_outer_target(target, new_group)
        return Optional(result)

    def remove_in_outer_target(self, target: State):
        group = self.group_index.find_in_outer_node(target).value
        if group is None:
            return Optional.create()
        new_group = self.arg_index.remove_in_target(group).value
        if new_group is None:
            return Optional.create()
        result = self.group_index.replace_in_outer_target(target, new_group)
        return Optional(result)
