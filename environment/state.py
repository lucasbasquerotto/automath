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
    SimpleScope,
    BaseInt,
    ScopeId,
    OptionalValueGroup,
    ExtendedTypeGroup,
    SingleValueTypeGroup,
    UnknownType,
    FunctionExpr,
    ITypedIndex,
    ITypedIntIndex,
    CountableTypeGroup,
    TmpNestedArg,
    TmpNestedArgs,
    IInstantiable,
)

T = typing.TypeVar('T', bound=INode)
K = typing.TypeVar('K', bound=INode)

###########################################################
#################### STATE DEFINITIONS ####################
###########################################################

class Scratch(OpaqueScope[IOptional[INode]], IInstantiable):

    idx_id = 0
    idx_child = 1

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            ScopeId,
            IOptional[INode],
        ]))

class ScratchGroup(BaseGroup[Scratch], IInstantiable):

    @classmethod
    def item_type(cls):
        return OpaqueScope[Scratch]

    @classmethod
    def from_raw_items(cls, items: tuple[INode | None, ...]) -> typing.Self:
        return cls.from_items([
            Scratch.with_content(Optional(s) if s is not None else Optional())
            for s in items
        ])

    def to_raw_items(self) -> tuple[INode, ...]:
        return tuple(s.child.apply() for s in self.as_tuple)

class PartialArgsGroup(
    FunctionExpr[OptionalValueGroup],
    IDefault,
    IInstantiable,
):

    idx_param_type_group = 0
    idx_scope = 1

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            ExtendedTypeGroup,
            OpaqueScope[OptionalValueGroup],
        ]))

    @property
    def param_type_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_param_type_group)

    @property
    def scope(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_scope)

    @property
    def scope_child(self) -> TmpNestedArgs:
        return self.nested_args((self.idx_scope, SimpleScope.idx_child))

    @property
    def inner_args(self) -> tuple[INode, ...]:
        return self.scope_child.apply().cast(OptionalValueGroup).as_tuple

    @classmethod
    def create(cls) -> typing.Self:
        return cls(
            ExtendedTypeGroup(SingleValueTypeGroup(UnknownType())),
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

    idx_definition_key = 0
    idx_definition_expr = 1

    @property
    def definition_key(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_definition_key)

    @property
    def definition_expr(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_definition_expr)

class FunctionDefinition(
    StateDefinition[FunctionId, FunctionExpr[T]],
    typing.Generic[T],
    IInstantiable,
):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            FunctionId,
            FunctionExpr[T],
        ]))

    @property
    def scope(self) -> TmpNestedArgs:
        return self.nested_args(
            (self.idx_definition_expr, FunctionExpr.idx_scope)
        )

class StateDefinitionGroup(BaseGroup[StateDefinition], IInstantiable):
    @classmethod
    def item_type(cls):
        return StateDefinition

###########################################################
########################## STATE ##########################
###########################################################

class State(InheritableNode, IInstantiable):

    idx_definition_group = 0
    idx_args_outer_group = 1
    idx_scratch_group = 2

    @property
    def definition_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_definition_group)

    @property
    def args_outer_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_args_outer_group)

    @property
    def scratch_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_scratch_group)

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
        return self.find_arg(node.definition_group.apply())

    def replace_in_outer_target(self, target: State, new_node: StateDefinition):
        definitions = list(target.definition_group.apply().cast(StateDefinitionGroup).as_tuple)
        for i, definition in enumerate(definitions):
            if self.value == (i+1):
                assert new_node.definition_key.apply() == definition.definition_key.apply()
                definitions[i] = new_node
                return Optional(State(
                    StateDefinitionGroup.from_items(definitions),
                    target.nested_arg(target.idx_args_outer_group).apply(),
                    target.nested_arg(target.idx_scratch_group).apply()))
        return Optional.create()

    def remove_in_outer_target(self, target: State):
        definitions = list(target.definition_group.apply().cast(StateDefinitionGroup).as_tuple)
        for i, _ in enumerate(definitions):
            if self.value == (i+1):
                definitions.pop(i)
                return Optional(State(
                    StateDefinitionGroup.from_items(definitions),
                    target.args_outer_group.apply(),
                    target.scratch_group.apply()))
        return Optional.create()

class StateScratchIndex(StateIntIndex[Scratch], IInstantiable):

    @classmethod
    def item_type(cls):
        return Scratch

    def find_in_outer_node(self, node: State):
        return self.find_arg(node.scratch_group.apply())

    def replace_in_outer_target(self, target: State, new_node: Scratch):
        group = self.replace_arg(target.scratch_group.apply(), new_node).value
        if group is None:
            return Optional.create()
        return Optional(State(
            target.definition_group.apply(),
            target.args_outer_group.apply(),
            group))

    def remove_in_outer_target(self, target: State):
        group = self.remove_arg(target.scratch_group.apply()).value
        if group is None:
            return Optional.create()
        return Optional(State(
            target.definition_group.apply(),
            target.args_outer_group.apply(),
            group))

class ScratchNodeIndex(NodeIntBaseIndex, IInstantiable):

    def find_in_node(self, node: INode):
        assert isinstance(node, Scratch)
        content = node.child.apply().cast(IOptional).value
        if content is None:
            return Optional.create()
        return NodeMainIndex(self.to_int).find_in_node(content)

    def replace_in_target(self, target_node: INode, new_node: INode):
        assert isinstance(target_node, Scratch)
        assert isinstance(new_node, INode)
        if self.to_int == 1:
            return Optional(new_node)
        old_content = target_node.child.apply().cast(IOptional).value_or_raise
        content = NodeMainIndex(self.to_int).replace_in_target(old_content, new_node)
        return content

    def remove_in_target(self, target_node: INode):
        assert isinstance(target_node, Scratch)
        if self.to_int == 1:
            return Optional.create()
        old_content = target_node.child.apply().cast(IOptional).value_or_raise
        content = NodeMainIndex(self.to_int).remove_in_target(old_content)
        return content

class StateArgsGroupIndex(StateIntIndex[PartialArgsGroup], IInstantiable):

    @classmethod
    def item_type(cls):
        return PartialArgsGroup

    def find_in_outer_node(self, node: State):
        return self.find_arg(node.args_outer_group.apply())

    def replace_in_outer_target(self, target: State, new_node: PartialArgsGroup):
        group = self.replace_arg(target.args_outer_group.apply(), new_node)
        args_outer_group = group.value
        if args_outer_group is None:
            return Optional.create()
        return State(
            target.definition_group.apply(),
            args_outer_group,
            target.scratch_group.apply())

    def remove_in_outer_target(self, target: State):
        group = self.remove_arg(target.args_outer_group.apply())
        args_outer_group = group.value
        if args_outer_group is None:
            return Optional.create()
        return State(
            target.definition_group.apply(),
            args_outer_group,
            target.scratch_group.apply())

class StateArgsGroupArgIndex(InheritableNode, IStateIndex[INode], IInstantiable):

    idx_group_index = 0
    idx_arg_index = 1

    @classmethod
    def item_type(cls):
        return INode

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            StateArgsGroupIndex,
            NodeArgIndex,
        ]))

    @property
    def group_index(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_group_index)

    @property
    def arg_index(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_arg_index)

    def find_in_outer_node(self, node: State):
        group_index = self.group_index.apply().cast(StateArgsGroupIndex)
        arg_index = self.arg_index.apply().cast(NodeArgIndex)

        group = group_index.find_in_outer_node(node).value
        if group is None:
            return Optional.create()

        return Optional(group[arg_index])

    def replace_in_outer_target(self, target: State, new_node: INode):
        group_index = self.group_index.apply().cast(StateArgsGroupIndex)
        arg_index = self.arg_index.apply().cast(NodeArgIndex)

        group = group_index.find_in_outer_node(target).value
        if group is None:
            return Optional.create()
        new_group = arg_index.replace_in_target(
            target_node=group,
            new_node=new_node,
        ).value
        if new_group is None:
            return Optional.create()
        result = group_index.replace_in_outer_target(target, new_group)

        return Optional(result)

    def remove_in_outer_target(self, target: State):
        group_index = self.group_index.apply().cast(StateArgsGroupIndex)
        arg_index = self.arg_index.apply().cast(NodeArgIndex)

        group = group_index.find_in_outer_node(target).value
        if group is None:
            return Optional.create()
        new_group = arg_index.remove_in_target(group).value
        if new_group is None:
            return Optional.create()
        result = group_index.replace_in_outer_target(target, new_group)

        return Optional(result)
