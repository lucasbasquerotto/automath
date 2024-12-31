import typing
from environment.core import (
    BaseNodeMainIndex,
    InheritableNode,
    Integer,
    BaseNode,
    Function)
from environment.state import (
    State,
    Scratch,
    StateScratchIndex,
    ScratchNodeIndex,
    StateArgsGroupIndex)

class ActionOutput(InheritableNode):
    def run(self, state: State) -> State:
        raise NotImplementedError

O = typing.TypeVar('O', bound=ActionOutput)

class BaseAction(InheritableNode, typing.Generic[O]):
    def run(self, state: State) -> O:
        raise NotImplementedError

class BasicActionGenerator:
    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        raise NotImplementedError

class ScratchBaseActionOutput(ActionOutput):
    def __init__(self, index: StateScratchIndex, node: BaseNode):
        assert isinstance(index, StateScratchIndex)
        assert isinstance(node, BaseNode)
        super().__init__(index, node)

    @property
    def index(self) -> StateScratchIndex:
        index = self.args[0]
        assert isinstance(index, StateScratchIndex)
        return index

    @property
    def node(self) -> BaseNode:
        node = self.args[1]
        assert isinstance(node, BaseNode)
        return node

    def run(self, state: State) -> State:
        raise NotImplementedError

###########################################################
##################### CREATE SCRATCH ######################
###########################################################

class CreateScratchOutput(ScratchBaseActionOutput):
    def run(self, state: State) -> State:
        index = self.index
        assert index.value == len(state.scratch_group.as_tuple) + 1
        scratch_group = state.scratch_group
        new_args = list(scratch_group.as_tuple) + [Scratch.create(self.node)]
        return State(
            function_group=state.function_group,
            args_group=state.args_group,
            scratch_group=scratch_group.func(*new_args),
        )

class CreateScratchFromNode(BaseAction[CreateScratchOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        state_index = BaseNodeMainIndex(Integer(arg1))
        assert arg2 == 0
        assert arg3 == 0
        return cls(state_index)

    def __init__(self, state_index: BaseNodeMainIndex):
        assert isinstance(state_index, BaseNodeMainIndex)
        super().__init__(state_index)

    @property
    def state_index(self) -> BaseNodeMainIndex:
        state_index = self.args[0]
        assert isinstance(state_index, BaseNodeMainIndex)
        return state_index

    def run(self, state: State) -> CreateScratchOutput:
        node = self.state_index.from_node(state)
        assert node is not None
        index = StateScratchIndex(Integer(len(state.scratch_group.as_tuple)))
        return CreateScratchOutput(index, node)

class CreateScratchFromFunction(BaseAction[CreateScratchOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        function_index = BaseNodeMainIndex(Integer(arg1))
        args_group_index = StateArgsGroupIndex(Integer(arg2))
        assert arg3 == 0
        return cls(function_index, args_group_index)

    def __init__(self, function_index: BaseNodeMainIndex, args_group_index: StateArgsGroupIndex):
        assert isinstance(function_index, BaseNodeMainIndex)
        assert isinstance(args_group_index, StateArgsGroupIndex)
        super().__init__(function_index, args_group_index)

    @property
    def function_index(self) -> BaseNodeMainIndex:
        function_index = self.args[0]
        assert isinstance(function_index, BaseNodeMainIndex)
        return function_index

    @property
    def args_group_index(self) -> StateArgsGroupIndex:
        args_group_index = self.args[1]
        assert isinstance(args_group_index, StateArgsGroupIndex)
        return args_group_index

    def run(self, state: State) -> CreateScratchOutput:
        function = self.function_index.from_node(state)
        assert isinstance(function, Function)
        function.validate()

        args_group = self.args_group_index.find_in_state(state)
        assert args_group is not None
        args_group.validate()

        node = args_group.apply_to(function)
        index = StateScratchIndex(Integer(len(state.scratch_group.as_tuple)))

        return CreateScratchOutput(index, node)

###########################################################
##################### DEFINE SCRATCH ######################
###########################################################

class DefineScratchOutput(ScratchBaseActionOutput):
    def run(self, state: State) -> State:
        scratch = Scratch.create(self.node)
        new_state = self.index.replace_in_state(state, scratch)
        assert new_state is not None
        return new_state

class DefineScratchFromNode(BaseAction[DefineScratchOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(Integer(arg1))
        state_index = BaseNodeMainIndex(Integer(arg2))
        assert arg3 == 0
        return cls(scratch_index, state_index)

    def __init__(
        self,
        scratch_index: StateScratchIndex,
        state_index: BaseNodeMainIndex,
    ):
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(state_index, BaseNodeMainIndex)
        super().__init__(scratch_index, state_index)

    @property
    def scratch_index(self) -> StateScratchIndex:
        scratch_index = self.args[0]
        assert isinstance(scratch_index, StateScratchIndex)
        return scratch_index

    @property
    def state_index(self) -> BaseNodeMainIndex:
        state_index = self.args[1]
        assert isinstance(state_index, BaseNodeMainIndex)
        return state_index

    def run(self, state: State) -> DefineScratchOutput:
        scratch_index = self.scratch_index
        scratch = scratch_index.find_in_state(state)
        assert scratch is not None

        node = self.state_index.from_node(state)
        assert node is not None

        return DefineScratchOutput(scratch_index, node)

class DefineScratchFromFunction(BaseAction[DefineScratchOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(Integer(arg1))
        function_index = BaseNodeMainIndex(Integer(arg2))
        args_group_index = StateArgsGroupIndex(Integer(arg3))
        return cls(scratch_index, function_index, args_group_index)

    def __init__(
        self,
        scratch_index: StateScratchIndex,
        function_index: BaseNodeMainIndex,
        args_group_index: StateArgsGroupIndex,
    ):
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(function_index, BaseNodeMainIndex)
        assert isinstance(args_group_index, StateArgsGroupIndex)
        super().__init__(scratch_index, function_index, args_group_index)

    @property
    def scratch_index(self) -> StateScratchIndex:
        scratch_index = self.args[0]
        assert isinstance(scratch_index, StateScratchIndex)
        return scratch_index

    @property
    def function_index(self) -> BaseNodeMainIndex:
        function_index = self.args[1]
        assert isinstance(function_index, BaseNodeMainIndex)
        return function_index

    @property
    def args_group_index(self) -> StateArgsGroupIndex:
        args_group_index = self.args[2]
        assert isinstance(args_group_index, StateArgsGroupIndex)
        return args_group_index

    def run(self, state: State) -> DefineScratchOutput:
        scratch_index = self.scratch_index
        scratch = scratch_index.find_in_state(state)
        assert scratch is not None

        function = self.function_index.from_node(state)
        assert isinstance(function, Function)
        function.validate()

        args_group = self.args_group_index.find_in_state(state)
        assert args_group is not None
        args_group.validate()

        node = args_group.apply_to(function)

        return DefineScratchOutput(scratch_index, node)

class UpdateScratchFromNode(BaseAction[DefineScratchOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(Integer(arg1))
        scratch_inner_index = ScratchNodeIndex(Integer(arg2))
        state_index = BaseNodeMainIndex(Integer(arg3))
        return cls(scratch_index, scratch_inner_index, state_index)

    def __init__(
        self,
        scratch_index: StateScratchIndex,
        scratch_inner_index: ScratchNodeIndex,
        state_index: BaseNodeMainIndex,
    ):
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(scratch_inner_index, ScratchNodeIndex)
        assert isinstance(state_index, BaseNodeMainIndex)
        super().__init__(scratch_index, scratch_inner_index, state_index)

    @property
    def scratch_index(self) -> StateScratchIndex:
        scratch_index = self.args[0]
        assert isinstance(scratch_index, StateScratchIndex)
        return scratch_index

    @property
    def scratch_inner_index(self) -> ScratchNodeIndex:
        scratch_inner_index = self.args[1]
        assert isinstance(scratch_inner_index, ScratchNodeIndex)
        return scratch_inner_index

    @property
    def state_index(self) -> BaseNodeMainIndex:
        state_index = self.args[2]
        assert isinstance(state_index, BaseNodeMainIndex)
        return state_index

    def run(self, state: State) -> DefineScratchOutput:
        scratch_index = self.scratch_index
        scratch = scratch_index.find_in_state(state)
        assert scratch is not None

        inner_node = self.state_index.from_node(state)
        assert inner_node is not None

        new_scratch = self.scratch_inner_index.replace_target(scratch, inner_node)
        assert new_scratch is not None

        return DefineScratchOutput(scratch_index, new_scratch.child)

class UpdateScratchFromFunction(BaseAction[DefineScratchOutput]):
    def __init__(
        self,
        scratch_index: StateScratchIndex,
        scratch_inner_index: ScratchNodeIndex,
        function_index: BaseNodeMainIndex,
        args_group_index: StateArgsGroupIndex,
    ):
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(scratch_inner_index, ScratchNodeIndex)
        assert isinstance(function_index, BaseNodeMainIndex)
        assert isinstance(args_group_index, StateArgsGroupIndex)
        super().__init__(scratch_index, scratch_inner_index, function_index, args_group_index)

    @property
    def scratch_index(self) -> StateScratchIndex:
        scratch_index = self.args[0]
        assert isinstance(scratch_index, StateScratchIndex)
        return scratch_index

    @property
    def scratch_inner_index(self) -> ScratchNodeIndex:
        scratch_inner_index = self.args[1]
        assert isinstance(scratch_inner_index, ScratchNodeIndex)
        return scratch_inner_index

    @property
    def function_index(self) -> BaseNodeMainIndex:
        function_index = self.args[2]
        assert isinstance(function_index, BaseNodeMainIndex)
        return function_index

    @property
    def args_group_index(self) -> StateArgsGroupIndex:
        args_group_index = self.args[3]
        assert isinstance(args_group_index, StateArgsGroupIndex)
        return args_group_index

    def run(self, state: State) -> DefineScratchOutput:
        scratch_index = self.scratch_index
        scratch = scratch_index.find_in_state(state)
        assert scratch is not None

        function = self.function_index.from_node(state)
        assert isinstance(function, Function)
        function.validate()

        args_group = self.args_group_index.find_in_state(state)
        assert args_group is not None
        args_group.validate()

        inner_node = args_group.apply_to(function)
        new_scratch = self.scratch_inner_index.replace_target(scratch, inner_node)
        assert new_scratch is not None

        return DefineScratchOutput(scratch_index, new_scratch.child)
