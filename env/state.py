import typing
from abc import ABC
from env.core import (
    INode,
    IDefault,
    IFunction,
    IOptional,
    OptionalBase,
    Optional,
    InheritableNode,
    BaseGroup,
    NodeIntBaseIndex,
    NodeMainIndex,
    NodeArgIndex,
    OpaqueScope,
    LaxOpaqueScope,
    SimpleBaseScope,
    BaseInt,
    ScopeId,
    OptionalValueGroup,
    ExtendedTypeGroup,
    SingleValueTypeGroup,
    UnknownType,
    FunctionExprBase,
    ITypedIndex,
    ITypedIntIndex,
    CountableTypeGroup,
    DefaultGroup,
    Eq,
    TmpNestedArg,
    TmpNestedArgs,
    IInstantiable,
)

T = typing.TypeVar('T', bound=INode)
K = typing.TypeVar('K', bound=INode)

###########################################################
#################### STATE DEFINITIONS ####################
###########################################################


class IContext(INode, ABC):
    pass

class OptionalContext(OptionalBase[T], IContext, IInstantiable, typing.Generic[T]):
    pass


class Scratch(OpaqueScope[OptionalContext[INode]], IInstantiable):

    idx_id = 1
    idx_child = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            ScopeId,
            OptionalContext[INode],
        ]))

    @classmethod
    def with_optional(cls, optional: IOptional[INode]) -> typing.Self:
        return cls.with_content(OptionalContext.from_optional(optional))

class ScratchGroup(BaseGroup[Scratch], IInstantiable):

    @classmethod
    def item_type(cls):
        return OpaqueScope[Scratch]

    @classmethod
    def from_raw_items(cls, items: tuple[INode | None, ...]) -> typing.Self:
        return cls.from_items([
            Scratch.with_content(OptionalContext(s) if s is not None else OptionalContext())
            for s in items
        ])

    def to_raw_items(self) -> tuple[INode, ...]:
        return tuple(s.child.apply() for s in self.as_tuple)

class PartialArgsGroup(
    FunctionExprBase[OptionalValueGroup],
    IDefault,
    IInstantiable,
):

    idx_param_type_group = 1
    idx_scope = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            ExtendedTypeGroup,
            LaxOpaqueScope[OptionalValueGroup],
        ]))

    @classmethod
    def from_args(
        cls,
        param_type_group: ExtendedTypeGroup,
        scope: LaxOpaqueScope[OptionalValueGroup],
    ) -> typing.Self:
        return cls(param_type_group, scope)

    @property
    def param_type_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_param_type_group)

    @property
    def scope(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_scope)

    @property
    def scope_child(self) -> TmpNestedArgs:
        return self.nested_args((self.idx_scope, SimpleBaseScope.idx_child))

    @property
    def inner_args(self) -> tuple[INode, ...]:
        return self.scope_child.apply().cast(OptionalValueGroup).as_tuple

    def fill_with_void(self) -> DefaultGroup:
        return self.scope_child.apply().cast(OptionalValueGroup).fill_with_void()

    @classmethod
    def create(cls) -> typing.Self:
        return cls(
            ExtendedTypeGroup(SingleValueTypeGroup(UnknownType())),
            LaxOpaqueScope.with_content(OptionalValueGroup()))

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

    idx_definition_key = 1
    idx_definition_expr = 2

    @property
    def definition_key(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_definition_key)

    @property
    def definition_expr(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_definition_expr)

class FunctionDefinition(
    StateDefinition[FunctionId, FunctionExprBase[T]],
    IInstantiable,
    typing.Generic[T],
):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            FunctionId,
            FunctionExprBase[T],
        ]))

    @property
    def scope(self) -> TmpNestedArgs:
        return self.nested_args(
            (self.idx_definition_expr, FunctionExprBase.idx_scope)
        )

class StateDefinitionGroup(BaseGroup[StateDefinition], IInstantiable):

    @classmethod
    def item_type(cls):
        return StateDefinition

###########################################################
########################## STATE ##########################
###########################################################

class State(InheritableNode, IDefault, IInstantiable):

    idx_scratch_group = 1
    idx_args_outer_group = 2
    idx_definition_group = 3

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            ScratchGroup,
            PartialArgsOuterGroup,
            StateDefinitionGroup,
        ]))

    @classmethod
    def create(cls) -> typing.Self:
        return cls(
            ScratchGroup(),
            PartialArgsOuterGroup(),
            StateDefinitionGroup(),
        )

    @property
    def scratch_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_scratch_group)

    @property
    def args_outer_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_args_outer_group)

    @property
    def definition_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_definition_group)

    def with_new_args(
        self,
        scratch_group: ScratchGroup | None = None,
        args_outer_group: PartialArgsOuterGroup | None = None,
        definition_group: StateDefinitionGroup | None = None,
    ) -> typing.Self:
        return self.func(
            scratch_group or self.scratch_group.apply(),
            args_outer_group or self.args_outer_group.apply(),
            definition_group or self.definition_group.apply(),
        )

    @classmethod
    def from_args(
        cls,
        scratch_group: ScratchGroup | None = None,
        args_outer_group: PartialArgsOuterGroup | None = None,
        definition_group: StateDefinitionGroup | None = None,
    ) -> typing.Self:
        return cls(
            scratch_group or ScratchGroup(),
            args_outer_group or PartialArgsOuterGroup(),
            definition_group or StateDefinitionGroup())

    @classmethod
    def from_raw(
        cls,
        scratchs: tuple[INode | None, ...] | None = None,
        args_groups: tuple[PartialArgsGroup, ...] | None = None,
        definitions: tuple[StateDefinition, ...] | None = None,
    ) -> typing.Self:
        scratchs = scratchs or tuple()
        args_groups = args_groups or tuple()
        definitions = definitions or tuple()
        return cls(
            ScratchGroup.from_raw_items(scratchs),
            PartialArgsOuterGroup.from_items(args_groups),
            StateDefinitionGroup.from_items(definitions))

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

    def find_in_outer_node(self, node: State) -> IOptional[StateDefinition]:
        return self.find_arg(node.definition_group.apply())

    def replace_in_outer_target(
        self,
        target: State,
        new_node: StateDefinition,
    ) -> IOptional[State]:
        definitions = list(target.definition_group.apply().cast(StateDefinitionGroup).as_tuple)
        for i, definition in enumerate(definitions):
            if self.as_int == (i+1):
                Eq(
                    new_node.definition_key.apply(),
                    definition.definition_key.apply(),
                ).raise_on_false()
                definitions[i] = new_node
                return Optional(target.with_new_args(
                    definition_group=StateDefinitionGroup.from_items(definitions),
                ))
        return Optional.create()

    def remove_in_outer_target(self, target: State) -> IOptional[State]:
        definitions = list(target.definition_group.apply().cast(StateDefinitionGroup).as_tuple)
        for i, _ in enumerate(definitions):
            if self.as_int == (i+1):
                definitions.pop(i)
                return Optional(target.with_new_args(
                    definition_group=StateDefinitionGroup.from_items(definitions),
                ))
        return Optional.create()

class StateScratchIndex(StateIntIndex[Scratch], IInstantiable):

    @classmethod
    def item_type(cls):
        return Scratch

    def find_in_outer_node(self, node: State) -> IOptional[Scratch]:
        return self.find_arg(node.scratch_group.apply())

    def replace_in_outer_target(self, target: State, new_node: Scratch) -> IOptional[State]:
        group = self.replace_arg(target.scratch_group.apply(), new_node).value
        if group is None:
            return Optional.create()
        assert isinstance(group, ScratchGroup)
        return Optional(target.with_new_args(
            scratch_group=group,
        ))

    def remove_in_outer_target(self, target: State) -> IOptional[State]:
        group = self.remove_arg(target.scratch_group.apply()).value
        if group is None:
            return Optional.create()
        assert isinstance(group, ScratchGroup)
        return Optional(target.with_new_args(
            scratch_group=group,
        ))

class ScratchNodeIndex(NodeIntBaseIndex, IInstantiable):

    def find_in_node(self, node: INode):
        assert isinstance(node, Scratch)
        content = node.child.apply().cast(IOptional).value
        if content is None:
            return Optional.create()
        return NodeMainIndex(self.as_int).find_in_node(content)

    def replace_in_target(self, target_node: INode, new_node: INode):
        assert isinstance(target_node, Scratch)
        assert isinstance(new_node, INode)
        if self.as_int == 1:
            return Optional(new_node)
        old_content = target_node.child.apply().cast(IOptional).value_or_raise
        content = NodeMainIndex(self.as_int).replace_in_target(old_content, new_node)
        return content

    def remove_in_target(self, target_node: INode):
        assert isinstance(target_node, Scratch)
        if self.as_int == 1:
            return Optional.create()
        old_content = target_node.child.apply().cast(IOptional).value_or_raise
        content = NodeMainIndex(self.as_int).remove_in_target(old_content)
        return content

class StateArgsGroupIndex(StateIntIndex[PartialArgsGroup], IInstantiable):

    @classmethod
    def item_type(cls):
        return PartialArgsGroup

    def find_in_outer_node(self, node: State) -> IOptional[PartialArgsGroup]:
        return self.find_arg(node.args_outer_group.apply())

    def replace_in_outer_target(
        self,
        target: State,
        new_node: PartialArgsGroup,
    ) -> IOptional[State]:
        group = self.replace_arg(target.args_outer_group.apply(), new_node)
        args_outer_group = group.value
        if args_outer_group is None:
            return Optional.create()
        assert isinstance(args_outer_group, PartialArgsOuterGroup)
        return Optional(target.with_new_args(args_outer_group=args_outer_group))

    def remove_in_outer_target(self, target: State) -> IOptional[State]:
        group = self.remove_arg(target.args_outer_group.apply())
        args_outer_group = group.value
        if args_outer_group is None:
            return Optional.create()
        assert isinstance(args_outer_group, PartialArgsOuterGroup)
        return Optional(target.with_new_args(args_outer_group=args_outer_group))

class StateArgsGroupArgIndex(InheritableNode, IStateIndex[INode], IInstantiable):

    idx_group_index = 1
    idx_arg_index = 2

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

        return arg_index.find_in_node(group)

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
