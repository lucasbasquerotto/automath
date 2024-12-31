import typing
from environment.core import (
    BaseNode,
    InheritableNode,
    PartialArgsGroup,
    BaseValueGroup,
    StrictGroup,
    BaseNodeIndex,
    BaseNodeIntIndex,
    BaseNodeMainIndex,
    BaseNodeArgIndex,
    OpaqueScope,
    Integer,
    ScopeId,
    FunctionDefinition)

T = typing.TypeVar('T', bound=BaseNode)
K = typing.TypeVar('K', bound=BaseNode)

class Scratch(OpaqueScope[BaseNode]):
    def __init__(self, id: ScopeId, child: BaseNode):
        assert isinstance(child, BaseNode)
        super().__init__(id, child)

class ScratchGroup(BaseValueGroup[Scratch]):
    @classmethod
    def item_type(cls):
        return OpaqueScope[Scratch]

    @classmethod
    def from_raw_items(cls, items: tuple[BaseNode, ...]) -> typing.Self:
        return cls.from_items([Scratch.create(s) for s in items])

    def to_raw_items(self) -> tuple[BaseNode, ...]:
        return tuple(s.child for s in self.as_tuple)

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
        args_group: PartialArgsOuterGroup,
        scratch_group: ScratchGroup,
    ):
        super().__init__(function_group, args_group, scratch_group)

    @property
    def function_group(self) -> FunctionDefinitionGroup:
        function_group = self.args[0]
        assert isinstance(function_group, FunctionDefinitionGroup)
        return function_group

    @property
    def args_group(self) -> PartialArgsOuterGroup:
        args_group = self.args[2]
        assert isinstance(args_group, PartialArgsOuterGroup)
        return args_group

    @property
    def scratch_group(self) -> ScratchGroup:
        scratch_group = self.args[1]
        assert isinstance(scratch_group, ScratchGroup)
        return scratch_group

    @classmethod
    def from_raw(
        cls,
        definitions: tuple[FunctionDefinition, ...],
        arg_groups: tuple[PartialArgsGroup, ...],
        scratchs: tuple[BaseNode, ...],
    ) -> typing.Self:
        return cls(
            FunctionDefinitionGroup.from_items(definitions),
            PartialArgsOuterGroup.from_items(arg_groups),
            ScratchGroup.from_raw_items(scratchs))

class StateIndex(BaseNodeIndex, typing.Generic[T]):
    @classmethod
    def item_type(cls) -> type[T]:
        raise NotImplementedError

    def from_node(self, node: BaseNode) -> BaseNode | None:
        assert isinstance(node, State)
        return self.find_in_state(node)

    def replace_target(self, target_node: BaseNode, new_node: BaseNode) -> BaseNode | None:
        assert isinstance(target_node, State)
        assert isinstance(new_node, self.item_type())
        return self.replace_in_state(target_node, new_node)

    def find_in_state(self, state: State) -> T | None:
        raise NotImplementedError

    def replace_in_state(self, state: State, new_node: T) -> State | None:
        raise NotImplementedError

class StateIntIndex(StateIndex[T], typing.Generic[T]):
    @classmethod
    def item_type(cls) -> type[T]:
        raise NotImplementedError

    def __init__(self, index: Integer):
        assert isinstance(index, Integer)
        super().__init__(index)

    @property
    def index(self) -> Integer:
        index = self.args[0]
        assert isinstance(index, Integer)
        return index

    @property
    def value(self) -> int:
        return self.index.value

    def find_in_state(self, state: State) -> T | None:
        raise NotImplementedError

    def replace_in_state(self, state: State, new_node: T) -> State | None:
        raise NotImplementedError

    def find_arg(self, node: BaseNode) -> T | None:
        result = BaseNodeArgIndex(self.index).from_node(node)
        if result is None:
            return None
        assert isinstance(result, self.item_type())
        return result

    def replace_arg(self, node: K, new_node: T) -> K | None:
        result = BaseNodeArgIndex(self.index).replace_target(node, new_node)
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
                    args_group=state.args_group,
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
            args_group=state.args_group,
            scratch_group=result)

class ScratchNodeIndex(BaseNodeIntIndex):
    def from_node(self, node: BaseNode) -> BaseNode | None:
        assert isinstance(node, Scratch)
        return BaseNodeMainIndex(self.index).from_node(node.child)

    def replace_target(self, target_node: BaseNode, new_node: BaseNode) -> Scratch | None:
        assert isinstance(target_node, Scratch)
        child = BaseNodeMainIndex(self.index).replace_target(target_node.child, new_node)
        if child is None:
            return None
        return Scratch(target_node.id, child)

class StateArgsGroupIndex(StateIntIndex[PartialArgsGroup]):
    @classmethod
    def item_type(cls):
        return PartialArgsGroup

    def find_in_state(self, state: State) -> PartialArgsGroup | None:
        return self.find_arg(state.args_group)

    def replace_in_state(self, state: State, new_node: PartialArgsGroup) -> State | None:
        result = self.replace_arg(state.args_group, new_node)
        if result is None:
            return None
        return State(
            function_group=state.function_group,
            args_group=result,
            scratch_group=state.scratch_group)

class StateArgsGroupArgIndex(StateIndex[BaseNode]):
    @classmethod
    def item_type(cls):
        return BaseNode

    def __init__(self, group_index: StateArgsGroupIndex, arg_index: BaseNodeArgIndex):
        assert isinstance(group_index, StateArgsGroupIndex)
        assert isinstance(arg_index, BaseNodeArgIndex)
        super().__init__(group_index, arg_index)

    @property
    def group_index(self) -> StateArgsGroupIndex:
        group_index = self.args[0]
        assert isinstance(group_index, StateArgsGroupIndex)
        return group_index

    @property
    def arg_index(self) -> BaseNodeArgIndex:
        arg_index = self.args[1]
        assert isinstance(arg_index, BaseNodeArgIndex)
        return arg_index

    def find_in_state(self, state: State) -> BaseNode | None:
        group = self.group_index.find_in_state(state)
        if group is None:
            return None
        return group[self.arg_index]

    def replace_in_state(self, state: State, new_node: BaseNode) -> State | None:
        group = self.group_index.find_in_state(state)
        if group is None:
            return state
        new_group = self.arg_index.replace_target(
            target_node=group,
            new_node=new_node)
        if new_group is None:
            return None
        return self.group_index.replace_in_state(state, new_group)
