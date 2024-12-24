import typing
from environment.core import (
    BaseNode,
    InheritableNode,
    FunctionDefinition,
    FunctionInfo,
    ParamsArgsGroup,
    StrictGroup,
    OptionalGroup,
    BaseNodeIndex)

T = typing.TypeVar('T', bound=BaseNode)

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

class StateIndex(BaseNodeIndex, typing.Generic[T]):
    def find_in_state(self, state: 'State') -> T | None:
        raise NotImplementedError()

    def replace_in_state(self, state: 'State', new_node: T) -> 'State':
        raise NotImplementedError()

class StateDefinitionIndex(StateIndex[FunctionDefinitionNode]):
    def find_in_state(self, state: 'State') -> FunctionDefinitionNode | None:
        definitions = state.definitions
        for i, d in enumerate(definitions):
            if self.value == (i+1):
                return d
        return None

    def replace_in_state(self, state: 'State', new_node: FunctionDefinitionNode) -> 'State':
        definitions = list(state.definitions)
        for i, _ in enumerate(definitions):
            if self.value == (i+1):
                assert new_node.definition_key == definitions[i].definition_key
                definitions[i] = new_node
                return State.from_raw(
                    definitions=tuple(definitions),
                    partial_definitions=state.partial_definitions,
                    arg_groups=state.arg_groups)
        return state

class StatePartialDefinitionIndex(StateIndex[FunctionInfo]):
    def find_in_state(self, state: 'State') -> FunctionInfo | None:
        partial_definitions = state.partial_definitions
        for i, d in enumerate(partial_definitions):
            if self.value == (i+1):
                return d
        return None

    def replace_in_state(self, state: 'State', new_node: FunctionInfo) -> 'State':
        partial_definitions = list(state.partial_definitions)
        for i, _ in enumerate(partial_definitions):
            if self.value == (i+1):
                partial_definitions[i] = new_node
                return State.from_raw(
                    definitions=state.definitions,
                    partial_definitions=tuple(partial_definitions),
                    arg_groups=state.arg_groups)
        return state

class StateArgGroupIndex(StateIndex[ParamsArgsGroup]):
    def find_in_state(self, state: 'State') -> ParamsArgsGroup | None:
        arg_groups = state.arg_groups
        for i, a in enumerate(arg_groups):
            if self.value == (i+1):
                return a
        return None

    def replace_in_state(self, state: 'State', new_node: ParamsArgsGroup) -> 'State':
        arg_groups = list(state.arg_groups)
        for i, _ in enumerate(arg_groups):
            if self.value == (i+1):
                arg_groups[i] = new_node
                return State.from_raw(
                    definitions=state.definitions,
                    partial_definitions=state.partial_definitions,
                    arg_groups=tuple(arg_groups))
        return state

class State(InheritableNode):
    def __init__(
        self,
        definitions: FunctionDefinitionGroup,
        partial_definitions: PartialDefinitionGroup,
        arg_groups: ParamsArgsOuterGroup,
    ):
        super().__init__(definitions, partial_definitions, arg_groups)

    @property
    def definition_group(self) -> FunctionDefinitionGroup:
        definition_group = self.args[0]
        assert isinstance(definition_group, FunctionDefinitionGroup)
        return definition_group

    @property
    def definitions(self) -> tuple[FunctionDefinitionNode, ...]:
        return self.definition_group.as_tuple

    @property
    def partial_definition_group(self) -> PartialDefinitionGroup:
        partial_definition_group = self.args[1]
        assert isinstance(partial_definition_group, PartialDefinitionGroup)
        return partial_definition_group

    @property
    def partial_definitions(self) -> tuple[FunctionInfo | None, ...]:
        return self.partial_definition_group.as_tuple

    @property
    def args_outer_group(self) -> ParamsArgsOuterGroup:
        args_outer_group = self.args[2]
        assert isinstance(args_outer_group, ParamsArgsOuterGroup)
        return args_outer_group

    @property
    def arg_groups(self) -> tuple[ParamsArgsGroup, ...]:
        return self.args_outer_group.as_tuple

    def __getitem__(self, index: BaseNodeIndex) -> BaseNode | None:
        if isinstance(index, StateIndex):
            return index.find_in_state(self)
        return super().__getitem__(index)

    def replace(self, index: 'BaseNodeIndex', new_node: BaseNode) -> BaseNode | None:
        if isinstance(index, StateIndex):
            return index.replace_in_state(self, new_node)
        return super().replace(index, new_node)

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
