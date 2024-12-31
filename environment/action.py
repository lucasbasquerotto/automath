import typing
from environment.core import (
    BaseNodeMainIndex,
    BaseNodeArgIndex,
    InheritableNode,
    Integer,
    Optional,
    BaseNode,
    ExtendedTypeGroup,
    Function,
    OptionalValueGroup,
    OpaqueScope,
    TypeNode,
    Param,
    ScopeId,
    UnknownTypeNode,
    FunctionDefinition)
from environment.state import (
    State,
    Scratch,
    StateScratchIndex,
    ScratchNodeIndex,
    StateArgsGroupIndex,
    PartialArgsGroup)

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

###########################################################
################### BASE ACTION OUTPUTS ###################
###########################################################

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
            args_outer_group=state.args_outer_group,
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

        args_group = self.args_group_index.find_in_state(state)
        assert args_group is not None

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

        args_group = self.args_group_index.find_in_state(state)
        assert args_group is not None

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

        args_group = self.args_group_index.find_in_state(state)
        assert args_group is not None

        function_node = self.function_index.from_node(state)
        assert function_node is not None

        if isinstance(function_node, Function) or isinstance(function_node, FunctionDefinition):
            inner_node = args_group.apply_to(function_node)
        elif isinstance(function_node, TypeNode):
            t = function_node.type
            param_type_group = t.arg_type_group()
            assert isinstance(param_type_group, ExtendedTypeGroup)
            arg_strict_group = args_group.inner_args_group.value_group()
            param_type_group.validate_values(arg_strict_group)
            params_amount = len(arg_strict_group.as_tuple)
            content = t.new(*[
                Param(ScopeId(1), Integer(i+1), UnknownTypeNode())
                for i in range(params_amount)])
            function = Function(
                param_type_group,
                OpaqueScope.create(content),
            )
            inner_node = args_group.apply_to(function)
        else:
            raise ValueError(f"Invalid function node type: {type(function_node)}")

        new_scratch = self.scratch_inner_index.replace_target(scratch, inner_node)
        assert new_scratch is not None

        return DefineScratchOutput(scratch_index, new_scratch.child)

###########################################################
#################### CREATE ARGS GROUP ####################
###########################################################

class CreateArgsGroupOutput(ActionOutput):
    def __init__(
        self,
        index: StateArgsGroupIndex,
        new_args_group: PartialArgsGroup,
    ):
        assert isinstance(index, StateArgsGroupIndex)
        assert isinstance(new_args_group, PartialArgsGroup)
        super().__init__(index, new_args_group)

    @property
    def index(self) -> StateArgsGroupIndex:
        index = self.args[0]
        assert isinstance(index, StateArgsGroupIndex)
        return index

    @property
    def new_args_group(self) -> PartialArgsGroup:
        new_args_group = self.args[1]
        assert isinstance(new_args_group, PartialArgsGroup)
        return new_args_group

    def run(self, state: State) -> State:
        index = self.index
        new_args_group = self.new_args_group

        assert index.value == len(state.args_outer_group.as_tuple) + 1
        for arg in new_args_group.scope.child.as_tuple:
            arg.validate()

        args_group = state.args_outer_group
        new_args = list(args_group.as_tuple) + [new_args_group]

        return State(
            function_group=state.function_group,
            args_outer_group=args_group.func(*new_args),
            scratch_group=state.scratch_group,
        )

class CreateArgsGroupFromAmountsAction(BaseAction[CreateArgsGroupOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        args_amount = Integer(arg1)
        params_amount = None if arg2 == 0 else Integer(arg2)
        assert arg3 == 0
        return cls(args_amount, Optional(params_amount))

    def __init__(
        self,
        args_amount: Integer,
        params_amount: Optional[Integer],
    ):
        assert isinstance(args_amount, Integer)
        assert isinstance(params_amount, Optional)
        assert params_amount is None or isinstance(params_amount.value, Integer)
        super().__init__(args_amount, params_amount)

    @property
    def args_amount(self) -> Integer:
        args_amount = self.args[0]
        assert isinstance(args_amount, Integer)
        return args_amount

    @property
    def params_amount(self) -> Optional[Integer]:
        params_amount = self.args[1]
        return typing.cast(Optional[Integer], params_amount)

    def run(self, state: State) -> CreateArgsGroupOutput:
        args_amount = self.args_amount
        params_amount = self.params_amount
        new_args_group = PartialArgsGroup(
                ExtendedTypeGroup.default(params_amount),
                OpaqueScope.create(OptionalValueGroup.default(args_amount)),
        )
        index = StateArgsGroupIndex(Integer(len(state.args_outer_group.as_tuple)))
        return CreateArgsGroupOutput(index, new_args_group)

class CreateArgsGroupDynamicAction(BaseAction[CreateArgsGroupOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        node_index = BaseNodeMainIndex(Integer(arg1))
        assert arg2 == 0
        assert arg3 == 0
        return cls(node_index)

    def __init__(self, node_index: BaseNodeMainIndex):
        assert isinstance(node_index, BaseNodeMainIndex)
        super().__init__(node_index)

    @property
    def node_index(self) -> BaseNodeMainIndex:
        node_index = self.args[0]
        assert isinstance(node_index, BaseNodeMainIndex)
        return node_index

    def run(self, state: State) -> CreateArgsGroupOutput:
        node_index = self.node_index
        new_args_group = node_index.from_node(state)
        assert isinstance(new_args_group, PartialArgsGroup)

        index = StateArgsGroupIndex(Integer(len(state.args_outer_group.as_tuple)))

        return CreateArgsGroupOutput(index, new_args_group)

###########################################################
################## DEFINE ARGS GROUP ARG ##################
###########################################################

class DefineArgsGroupArgOutput(ActionOutput):
    def __init__(
        self,
        index: StateArgsGroupIndex,
        arg_index: BaseNodeArgIndex,
        new_arg: BaseNode,
    ):
        assert isinstance(index, StateArgsGroupIndex)
        assert isinstance(arg_index, BaseNodeArgIndex)
        assert isinstance(new_arg, BaseNode)
        super().__init__(index, arg_index, new_arg)

    @property
    def index(self) -> StateArgsGroupIndex:
        index = self.args[0]
        assert isinstance(index, StateArgsGroupIndex)
        return index

    @property
    def arg_index(self) -> BaseNodeArgIndex:
        arg_index = self.args[1]
        assert isinstance(arg_index, BaseNodeArgIndex)
        return arg_index

    @property
    def new_arg(self) -> BaseNode:
        new_arg = self.args[2]
        assert isinstance(new_arg, BaseNode)
        return new_arg

    def run(self, state: State) -> State:
        index = self.index
        arg_index = self.arg_index
        new_arg = self.new_arg

        new_arg.validate()

        args_group = index.find_in_state(state)
        assert args_group is not None

        new_args_group = arg_index.replace_target(args_group, new_arg)
        assert isinstance(new_args_group, PartialArgsGroup)
        new_args_group = PartialArgsGroup(
            new_args_group.param_type_group,
            new_args_group.scope.normalize())

        new_state = index.replace_in_state(state, new_args_group)
        assert new_state is not None

        return new_state

class DefineArgsGroupArgFromNodeAction(BaseAction[DefineArgsGroupArgOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        args_group_index = StateArgsGroupIndex(Integer(arg1))
        arg_index = BaseNodeArgIndex(Integer(arg2))
        node_index = BaseNodeMainIndex(Integer(arg3))
        return cls(args_group_index, arg_index, node_index)

    def __init__(
        self,
        args_group_index: StateArgsGroupIndex,
        arg_index: BaseNodeArgIndex,
        node_index: BaseNodeMainIndex,
    ):
        assert isinstance(args_group_index, StateArgsGroupIndex)
        assert isinstance(arg_index, BaseNodeArgIndex)
        assert isinstance(node_index, BaseNodeMainIndex)
        super().__init__(args_group_index, arg_index, node_index)

    @property
    def args_group_index(self) -> StateArgsGroupIndex:
        args_group_index = self.args[0]
        assert isinstance(args_group_index, StateArgsGroupIndex)
        return args_group_index

    @property
    def arg_index(self) -> BaseNodeArgIndex:
        arg_index = self.args[1]
        assert isinstance(arg_index, BaseNodeArgIndex)
        return arg_index

    @property
    def node_index(self) -> BaseNodeMainIndex:
        node_index = self.args[2]
        assert isinstance(node_index, BaseNodeMainIndex)
        return node_index

    def run(self, state: State) -> DefineArgsGroupArgOutput:
        args_group_index = self.args_group_index
        arg_index = self.arg_index
        node_index = self.node_index

        args_group = args_group_index.find_in_state(state)
        assert args_group is not None

        new_arg = node_index.from_node(state)
        assert new_arg is not None

        return DefineArgsGroupArgOutput(args_group_index, arg_index, new_arg)
