import typing
from environment.core import (
    BaseNode,
    InheritableNode,
    FunctionDefinition,
    FunctionInfo,
    ParamsArgsGroup,
    StrictGroup,
    OptionalGroup,
    BaseNodeIndex,
    BaseNodeArgIndex,
    Integer)

T = typing.TypeVar('T', bound=BaseNode)
K = typing.TypeVar('K', bound=BaseNode)

class FunctionDefinitionNode(InheritableNode):
    def __init__(self, definition_key: FunctionDefinition, function_info: FunctionInfo):
        assert isinstance(definition_key, FunctionDefinition)
        assert isinstance(function_info, FunctionInfo)
        super().__init__(definition_key, function_info)

    @property
    def definition_key(self) -> FunctionDefinition:
        key = self.args[0]
        assert isinstance(key, FunctionDefinition)
        return key

    @property
    def function_info(self) -> FunctionInfo:
        f = self.args[1]
        assert isinstance(f, FunctionInfo)
        return f

class FunctionDefinitionGroup(StrictGroup[FunctionDefinitionNode]):
    @classmethod
    def item_type(cls):
        return FunctionDefinitionNode

class PartialDefinitionGroup(OptionalGroup[FunctionInfo]):
    @classmethod
    def item_type(cls):
        return FunctionInfo

class ParamsArgsOuterGroup(StrictGroup[ParamsArgsGroup]):
    @classmethod
    def item_type(cls):
        return ParamsArgsGroup

class State(InheritableNode):
    def __init__(
        self,
        definitions: FunctionDefinitionGroup,
        partial_definitions: PartialDefinitionGroup,
        arg_groups: ParamsArgsOuterGroup,
    ):
        super().__init__(definitions, partial_definitions, arg_groups)

    @property
    def definitions(self) -> FunctionDefinitionGroup:
        definition_group = self.args[0]
        assert isinstance(definition_group, FunctionDefinitionGroup)
        return definition_group

    @property
    def partial_definitions(self) -> PartialDefinitionGroup:
        partial_definition_group = self.args[1]
        assert isinstance(partial_definition_group, PartialDefinitionGroup)
        return partial_definition_group

    @property
    def arg_groups(self) -> ParamsArgsOuterGroup:
        args_outer_group = self.args[2]
        assert isinstance(args_outer_group, ParamsArgsOuterGroup)
        return args_outer_group

    @classmethod
    def from_raw(
        cls,
        definitions: tuple[FunctionDefinitionNode, ...],
        partial_definitions: tuple[FunctionInfo | None, ...],
        arg_groups: tuple[ParamsArgsGroup, ...],
    ) -> 'State':
        return cls(
            FunctionDefinitionGroup(*definitions),
            PartialDefinitionGroup.from_items(partial_definitions),
            ParamsArgsOuterGroup(*arg_groups))

class StateIndex(BaseNodeIndex, typing.Generic[T]):
    @classmethod
    def item_type(cls) -> type[T]:
        raise NotImplementedError()

    def from_item(self, node: BaseNode) -> BaseNode | None:
        assert isinstance(node, State)
        return self.find_in_state(node)

    def replace_target(self, target_node: BaseNode, new_node: BaseNode) -> BaseNode | None:
        assert isinstance(target_node, State)
        assert isinstance(new_node, self.item_type())
        return self.replace_in_state(target_node, new_node)

    def find_in_state(self, state: State) -> T | None:
        raise NotImplementedError()

    def replace_in_state(self, state: State, new_node: T) -> State | None:
        raise NotImplementedError()

class StateIntIndex(StateIndex[T], typing.Generic[T]):
    @classmethod
    def item_type(cls) -> type[T]:
        raise NotImplementedError()

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
        raise NotImplementedError()

    def replace_in_state(self, state: State, new_node: T) -> State | None:
        raise NotImplementedError()

    def find_arg(self, node: BaseNode) -> T | None:
        result = BaseNodeArgIndex(self.index).from_item(node)
        if result is None:
            return None
        assert isinstance(result, self.item_type())
        return result

    def replace_arg(self, node: K, new_node: T) -> K | None:
        result = BaseNodeArgIndex(self.index).replace_target(node, new_node)
        if result is None:
            return None
        return result

class StateDefinitionIndex(StateIntIndex[FunctionDefinitionNode]):
    @classmethod
    def item_type(cls):
        return FunctionDefinitionNode

    def find_in_state(self, state: State) -> FunctionDefinitionNode | None:
        return self.find_arg(state.definitions)

    def replace_in_state(self, state: State, new_node: FunctionDefinitionNode) -> State | None:
        definitions = list(state.definitions.as_tuple)
        for i, _ in enumerate(definitions):
            if self.value == (i+1):
                assert new_node.definition_key == definitions[i].definition_key
                definitions[i] = new_node
                return State(
                    definitions=FunctionDefinitionGroup.from_items(definitions),
                    partial_definitions=state.partial_definitions,
                    arg_groups=state.arg_groups)
        return state

class StatePartialDefinitionIndex(StateIntIndex[FunctionInfo]):
    @classmethod
    def item_type(cls):
        return FunctionInfo

    def find_in_state(self, state: State) -> FunctionInfo | None:
        return self.find_arg(state.partial_definitions)

    def replace_in_state(self, state: State, new_node: FunctionInfo) -> State | None:
        result = self.replace_arg(state.partial_definitions, new_node)
        if result is None:
            return None
        return State(
            definitions=state.definitions,
            partial_definitions=result,
            arg_groups=state.arg_groups)

class StateArgGroupIndex(StateIntIndex[ParamsArgsGroup]):
    @classmethod
    def item_type(cls):
        return ParamsArgsGroup

    def find_in_state(self, state: State) -> ParamsArgsGroup | None:
        return self.find_arg(state.arg_groups)

    def replace_in_state(self, state: State, new_node: ParamsArgsGroup) -> State | None:
        result = self.replace_arg(state.arg_groups, new_node)
        if result is None:
            return None
        return State(
            definitions=state.definitions,
            partial_definitions=state.partial_definitions,
            arg_groups=result)

class StateArgGroupArgIndex(StateIndex[BaseNode]):
    @classmethod
    def item_type(cls):
        return BaseNode

    def __init__(self, group_index: StateArgGroupIndex, arg_index: BaseNodeArgIndex):
        assert isinstance(group_index, StateArgGroupIndex)
        assert isinstance(arg_index, BaseNodeArgIndex)
        super().__init__(group_index, arg_index)

    @property
    def group_index(self) -> StateArgGroupIndex:
        group_index = self.args[0]
        assert isinstance(group_index, StateArgGroupIndex)
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
