import typing
from abc import ABC
from environment.core import (
    INode,
    InheritableNode,
    BaseValueGroup,
    StrictGroup,
    INodeIndex,
    NodeIntBaseIndex,
    NodeMainIndex,
    NodeArgIndex,
    OpaqueScope,
    Integer,
    ScopeId,
    OptionalValueGroup,
    ExtendedTypeGroup,
    RestTypeGroup,
    UnknownTypeNode,
    Function,
    FunctionDefinition,
    IFunction,
)

T = typing.TypeVar('T', bound=INode)
K = typing.TypeVar('K', bound=INode)

class Scratch(OpaqueScope[INode]):
    def __init__(self, id: ScopeId, child: INode):
        assert isinstance(child, INode)
        super().__init__(id, child)

class ScratchGroup(BaseValueGroup[Scratch]):
    @classmethod
    def item_type(cls):
        return OpaqueScope[Scratch]

    @classmethod
    def from_raw_items(cls, items: tuple[INode, ...]) -> typing.Self:
        return cls.from_items([Scratch.create(s) for s in items])

    def to_raw_items(self) -> tuple[INode, ...]:
        return tuple(s.child for s in self.as_tuple)

class PartialArgsGroup(Function[OptionalValueGroup[T]], typing.Generic[T]):

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

    def apply_to(self, function: IFunction) -> Function:
        assert isinstance(function, IFunction)
        self.validate()
        scope = function.scope
        group = self.inner_args_group.fill_with_void()
        return Function(
            self.param_type_group,
            scope.func(
                scope.id,
                function.with_args(*group.as_tuple),
            ),
        )

    @classmethod
    def default(cls) -> typing.Self:
        return cls(
            ExtendedTypeGroup(RestTypeGroup(UnknownTypeNode())),
            OpaqueScope.create(OptionalValueGroup()))

class PartialArgsOuterGroup(BaseValueGroup[PartialArgsGroup]):
    @classmethod
    def item_type(cls):
        return PartialArgsGroup

class FunctionDefinitionGroup(StrictGroup[FunctionDefinition]):
    @classmethod
    def item_type(cls):
        return FunctionDefinition

class State(InheritableNode):
    def __init__(
        self,
        function_group: FunctionDefinitionGroup,
        args_outer_group: PartialArgsOuterGroup,
        scratch_group: ScratchGroup,
    ):
        super().__init__(function_group, args_outer_group, scratch_group)

    @property
    def function_group(self) -> FunctionDefinitionGroup:
        function_group = self.args[0]
        assert isinstance(function_group, FunctionDefinitionGroup)
        return function_group

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
        functions: tuple[FunctionDefinition, ...],
        args_outer_groups: tuple[PartialArgsGroup, ...],
        scratchs: tuple[INode, ...],
    ) -> typing.Self:
        return cls(
            FunctionDefinitionGroup.from_items(functions),
            PartialArgsOuterGroup.from_items(args_outer_groups),
            ScratchGroup.from_raw_items(scratchs))

class IStateIndex(INodeIndex, typing.Generic[T]):
    @classmethod
    def item_type(cls) -> type[T]:
        raise NotImplementedError

    def from_node(self, node: INode) -> INode | None:
        assert isinstance(node, State)
        return self.find_in_state(node)

    def replace_target(self, target_node: INode, new_node: INode) -> INode | None:
        assert isinstance(target_node, State)
        assert isinstance(new_node, self.item_type())
        return self.replace_in_state(target_node, new_node)

    def find_in_state(self, state: State) -> T | None:
        raise NotImplementedError

    def replace_in_state(self, state: State, new_node: T) -> State | None:
        raise NotImplementedError

class StateIntIndex(Integer, ABC, IStateIndex[T], typing.Generic[T]):

    def find_arg(self, node: INode) -> T | None:
        result = NodeArgIndex(self.to_int).from_node(node)
        if result is None:
            return None
        assert isinstance(result, self.item_type())
        return result

    def replace_arg(self, node: K, new_node: T) -> K | None:
        result = NodeArgIndex(self.to_int).replace_target(node, new_node)
        if result is None:
            return None
        return result

class StateDefinitionIndex(StateIntIndex[FunctionDefinition]):
    @classmethod
    def item_type(cls):
        return FunctionDefinition

    def find_in_state(self, state: State) -> FunctionDefinition | None:
        return self.find_arg(state.function_group)

    def replace_in_state(self, state: State, new_node: FunctionDefinition) -> State | None:
        definitions = list(state.function_group.as_tuple)
        for i, _ in enumerate(definitions):
            if self.value == (i+1):
                assert new_node.function_id == definitions[i].function_id
                definitions[i] = new_node
                return State(
                    function_group=FunctionDefinitionGroup.from_items(definitions),
                    args_outer_group=state.args_outer_group,
                    scratch_group=state.scratch_group)
        return state

class StateScratchIndex(StateIntIndex[Scratch]):
    @classmethod
    def item_type(cls):
        return Scratch

    def find_in_state(self, state: State) -> Scratch | None:
        return self.find_arg(state.scratch_group)

    def replace_in_state(self, state: State, new_node: Scratch) -> State | None:
        result = self.replace_arg(state.scratch_group, new_node)
        if result is None:
            return None
        return State(
            function_group=state.function_group,
            args_outer_group=state.args_outer_group,
            scratch_group=result)

class ScratchNodeIndex(NodeIntBaseIndex):
    def from_node(self, node: INode) -> INode | None:
        assert isinstance(node, Scratch)
        return NodeMainIndex(self.to_int).from_node(node.child)

    def replace_target(self, target_node: INode, new_node: INode) -> Scratch | None:
        assert isinstance(target_node, Scratch)
        child = NodeMainIndex(self.to_int).replace_target(target_node.child, new_node)
        if child is None:
            return None
        return Scratch(target_node.id, child)

class StateArgsGroupIndex(StateIntIndex[PartialArgsGroup]):
    @classmethod
    def item_type(cls):
        return PartialArgsGroup

    def find_in_state(self, state: State) -> PartialArgsGroup | None:
        return self.find_arg(state.args_outer_group)

    def replace_in_state(self, state: State, new_node: PartialArgsGroup) -> State | None:
        result = self.replace_arg(state.args_outer_group, new_node)
        if result is None:
            return None
        return State(
            function_group=state.function_group,
            args_outer_group=result,
            scratch_group=state.scratch_group)

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

    def find_in_state(self, state: State) -> INode | None:
        group = self.group_index.find_in_state(state)
        if group is None:
            return None
        return group[self.arg_index]

    def replace_in_state(self, state: State, new_node: INode) -> State | None:
        group = self.group_index.find_in_state(state)
        if group is None:
            return state
        new_group = self.arg_index.replace_target(
            target_node=group,
            new_node=new_node)
        if new_group is None:
            return None
        return self.group_index.replace_in_state(state, new_group)
