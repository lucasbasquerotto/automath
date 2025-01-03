import typing
from environment.core import (
    INode,
    NodeMainIndex,
    NodeArgIndex,
    IInt,
    Integer,
    Optional,
    ExtendedTypeGroup,
    IFunction,
    OptionalValueGroup,
    OpaqueScope,
    FunctionCall,
    TypeNode,
    IDefault,
    IFromInt,
    IsEmpty)
from environment.state import (
    State,
    Scratch,
    StateScratchIndex,
    ScratchNodeIndex,
    StateArgsGroupIndex,
    PartialArgsGroup)
from environment.full_state import (
    FullState,
    IFullStateIndex,
    FullStateIntIndex,
    MetaDefaultTypeIndex,
    MetaFromIntTypeIndex,
    MetaFullStateIntIndexTypeIndex)
from environment.action import (
    BaseAction,
    ActionOutput)

class BasicActionGenerator(INode):
    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        raise NotImplementedError

###########################################################
################### BASE ACTION OUTPUTS ###################
###########################################################

class ScratchBaseActionOutput(ActionOutput):
    def __init__(self, index: StateScratchIndex, node: INode):
        assert isinstance(index, StateScratchIndex)
        assert isinstance(node, INode)
        super().__init__(index, node)

    @property
    def index(self) -> StateScratchIndex:
        index = self.args[0]
        assert isinstance(index, StateScratchIndex)
        return index

    @property
    def node(self) -> INode:
        node = self.args[1]
        assert isinstance(node, INode)
        return node

    def apply(self, full_state: FullState) -> State:
        raise NotImplementedError

###########################################################
##################### CREATE SCRATCH ######################
###########################################################

class CreateScratchOutput(ScratchBaseActionOutput):
    def apply(self, full_state: FullState) -> State:
        state = full_state.current.state
        index = self.index
        assert index.value == len(state.scratch_group.as_tuple) + 1
        scratch_group = state.scratch_group
        new_args = list(scratch_group.as_tuple) + [Scratch.with_content(self.node)]
        return State(
            definition_group=state.definition_group,
            args_outer_group=state.args_outer_group,
            scratch_group=scratch_group.func(*new_args),
        )

class CreateScratchFromDefault(BaseAction[CreateScratchOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        type_index = MetaDefaultTypeIndex(arg1)
        assert arg2 == 0
        assert arg3 == 0
        return cls(type_index)

    def __init__(self, type_index: MetaDefaultTypeIndex):
        super().__init__(type_index)

    @property
    def type_index(self) -> MetaDefaultTypeIndex:
        type_index = self.args[0]
        assert isinstance(type_index, MetaDefaultTypeIndex)
        return type_index

    def _run(self, full_state: FullState) -> CreateScratchOutput:
        state = full_state.current.state
        node_type = self.type_index.find_in_outer_node(full_state)
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, IDefault)
        node = node_type.type.create()
        index = StateScratchIndex(len(state.scratch_group.as_tuple))
        return CreateScratchOutput(index, node)

class CreateScratchFromInt(BaseAction[CreateScratchOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        type_index = MetaFromIntTypeIndex(arg1)
        index_value = Integer(arg2)
        assert arg3 == 0
        return cls(type_index, index_value)

    def __init__(self, type_index: MetaFromIntTypeIndex, index_value: IInt):
        super().__init__(type_index, index_value)

    @property
    def type_index(self) -> MetaFromIntTypeIndex:
        type_index = self.args[0]
        assert isinstance(type_index, MetaFromIntTypeIndex)
        return type_index

    @property
    def index_value(self) -> Integer:
        index_value = self.args[1]
        assert isinstance(index_value, Integer)
        return index_value

    def _run(self, full_state: FullState) -> CreateScratchOutput:
        state = full_state.current.state
        node_type = self.type_index.find_in_outer_node(full_state)
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, IFromInt)
        node = node_type.type.from_int(self.index_value.to_int)
        index = StateScratchIndex(len(state.scratch_group.as_tuple))
        return CreateScratchOutput(index, node)

class CreateScratchFromIntIndex(BaseAction[CreateScratchOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        type_index = MetaFullStateIntIndexTypeIndex(arg1)
        index_value = Integer(arg2)
        assert arg3 == 0
        return cls(type_index, index_value)

    def __init__(self, type_index: MetaFullStateIntIndexTypeIndex, index_value: Integer):
        super().__init__(type_index, index_value)

    @property
    def type_index(self) -> MetaFullStateIntIndexTypeIndex:
        type_index = self.args[0]
        assert isinstance(type_index, MetaFullStateIntIndexTypeIndex)
        return type_index

    @property
    def index_value(self) -> Integer:
        index_value = self.args[1]
        assert isinstance(index_value, Integer)
        return index_value

    def _run(self, full_state: FullState) -> CreateScratchOutput:
        state = full_state.current.state
        type_index = self.type_index
        index_value = self.index_value
        node_type = type_index.find_in_outer_node(full_state)
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, FullStateIntIndex)
        node_index = typing.cast(
            FullStateIntIndex[INode],
            node_type.type.from_int(index_value.to_int))
        node = node_index.find_in_outer_node(full_state)
        node = IsEmpty.with_optional(node).value_or_raise
        index = StateScratchIndex(len(state.scratch_group.as_tuple))
        return CreateScratchOutput(index, node)

class CreateScratchFromIndex(BaseAction[CreateScratchOutput]):

    def __init__(self, node_index: IFullStateIndex[INode]):
        super().__init__(node_index)

    @property
    def node_index(self) -> IFullStateIndex[INode]:
        node_index = self.args[0]
        return typing.cast(IFullStateIndex[INode], node_index)

    def _run(self, full_state: FullState) -> CreateScratchOutput:
        state = full_state.current.state
        node_index = self.node_index
        node = node_index.find_in_outer_node(full_state)
        node = IsEmpty.with_optional(node).value_or_raise
        index = StateScratchIndex(len(state.scratch_group.as_tuple))
        return CreateScratchOutput(index, node)

class CreateScratchFromBasicFunction(BaseAction[CreateScratchOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        type_index = MetaFullStateIntIndexTypeIndex(arg1)
        index_value = Integer(arg2)
        args_group_index = StateArgsGroupIndex(arg3)
        return cls(type_index, index_value, args_group_index)

    def __init__(
        self,
        type_index: MetaFullStateIntIndexTypeIndex,
        index_value: Integer,
        args_group_index: StateArgsGroupIndex,
    ):
        super().__init__(type_index, index_value, args_group_index)

    @property
    def type_index(self) -> MetaFullStateIntIndexTypeIndex:
        type_index = self.args[0]
        assert isinstance(type_index, MetaFullStateIntIndexTypeIndex)
        return type_index

    @property
    def index_value(self) -> Integer:
        index_value = self.args[1]
        assert isinstance(index_value, Integer)
        return index_value

    @property
    def args_group_index(self) -> StateArgsGroupIndex:
        args_group_index = self.args[2]
        assert isinstance(args_group_index, StateArgsGroupIndex)
        return args_group_index

    def _run(self, full_state: FullState) -> CreateScratchOutput:
        state = full_state.current.state
        type_index = self.type_index
        index_value = self.index_value
        node_type = type_index.find_in_outer_node(full_state)
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, FullStateIntIndex)
        node_index = typing.cast(
            FullStateIntIndex[INode],
            node_type.type.from_int(index_value.to_int))
        function_node = node_index.find_in_outer_node(full_state)
        function_node = Optional(function_node).value_or_raise

        args_group = self.args_group_index.find_in_outer_node(state)
        assert isinstance(args_group, PartialArgsGroup)

        function_node.as_node.validate()
        args_group.validate()
        node = FunctionCall(
            typing.cast(IFunction, function_node),
            args_group.inner_args_group.fill_with_void())

        index = StateScratchIndex(len(state.scratch_group.as_tuple))

        return CreateScratchOutput(index, node)

class CreateScratchFromFunction(BaseAction[CreateScratchOutput]):

    def __init__(
        self,
        node_index: IFullStateIndex[INode],
        args_group_index: StateArgsGroupIndex,
    ):
        super().__init__(node_index, args_group_index)

    @property
    def node_index(self) -> IFullStateIndex[INode]:
        node_index = self.args[0]
        return typing.cast(IFullStateIndex[INode], node_index)

    @property
    def args_group_index(self) -> StateArgsGroupIndex:
        args_group_index = self.args[1]
        assert isinstance(args_group_index, StateArgsGroupIndex)
        return args_group_index

    def _run(self, full_state: FullState) -> CreateScratchOutput:
        state = full_state.current.state
        node_index = self.node_index
        function_node = node_index.find_in_outer_node(full_state)
        function_node = Optional(function_node).value_or_raise

        args_group = self.args_group_index.find_in_outer_node(state)
        assert isinstance(args_group, PartialArgsGroup)

        function_node.as_node.validate()
        args_group.validate()
        node = FunctionCall(
            typing.cast(IFunction, function_node),
            args_group.inner_args_group.fill_with_void())

        index = StateScratchIndex(len(state.scratch_group.as_tuple))

        return CreateScratchOutput(index, node)

class CreateScratchDynamicAction(BaseAction[CreateScratchOutput]):

    def __init__(self, node: INode):
        super().__init__(node)

    @property
    def node(self) -> INode:
        node = self.args[0]
        assert isinstance(node, INode)
        return node

    def _run(self, full_state: FullState) -> CreateScratchOutput:
        state = full_state.current.state
        node = self.node
        index = StateScratchIndex(len(state.scratch_group.as_tuple))
        return CreateScratchOutput(index, node)

###########################################################
##################### DEFINE SCRATCH ######################
###########################################################

class DefineScratchOutput(ScratchBaseActionOutput):
    def apply(self, full_state: FullState) -> State:
        state = full_state.current.state
        scratch = Scratch.with_content(self.node)
        new_state = self.index.replace_in_outer_target(state, scratch)
        assert new_state is not None
        return new_state

class DefineScratchFromNode(BaseAction[DefineScratchOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        node_index = NodeMainIndex(arg2)
        assert arg3 == 0
        return cls(scratch_index, node_index)

    def __init__(
        self,
        scratch_index: StateScratchIndex,
        node_index: NodeMainIndex,
    ):
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(node_index, NodeMainIndex)
        super().__init__(scratch_index, node_index)

    @property
    def scratch_index(self) -> StateScratchIndex:
        scratch_index = self.args[0]
        assert isinstance(scratch_index, StateScratchIndex)
        return scratch_index

    @property
    def node_index(self) -> NodeMainIndex:
        node_index = self.args[1]
        assert isinstance(node_index, NodeMainIndex)
        return node_index

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        state = full_state.current.state
        scratch_index = self.scratch_index
        scratch = scratch_index.find_in_outer_node(state)
        assert scratch is not None

        node = self.node_index.find_in_node(full_state)
        assert node is not None

        return DefineScratchOutput(scratch_index, node)

class DefineScratchFromFunction(BaseAction[DefineScratchOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        function_index = NodeMainIndex(arg2)
        args_group_index = StateArgsGroupIndex(arg3)
        return cls(scratch_index, function_index, args_group_index)

    def __init__(
        self,
        scratch_index: StateScratchIndex,
        function_index: NodeMainIndex,
        args_group_index: StateArgsGroupIndex,
    ):
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(function_index, NodeMainIndex)
        assert isinstance(args_group_index, StateArgsGroupIndex)
        super().__init__(scratch_index, function_index, args_group_index)

    @property
    def scratch_index(self) -> StateScratchIndex:
        scratch_index = self.args[0]
        assert isinstance(scratch_index, StateScratchIndex)
        return scratch_index

    @property
    def function_index(self) -> NodeMainIndex:
        function_index = self.args[1]
        assert isinstance(function_index, NodeMainIndex)
        return function_index

    @property
    def args_group_index(self) -> StateArgsGroupIndex:
        args_group_index = self.args[2]
        assert isinstance(args_group_index, StateArgsGroupIndex)
        return args_group_index

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        state = full_state.current.state
        scratch_index = self.scratch_index
        scratch = scratch_index.find_in_outer_node(state)
        assert scratch is not None

        function = self.function_index.find_in_node(full_state)
        assert isinstance(function, IFunction)

        args_group = self.args_group_index.find_in_outer_node(state)
        assert args_group is not None

        function.as_node.validate()
        args_group.validate()
        node = FunctionCall(function, args_group.inner_args_group.fill_with_void())

        return DefineScratchOutput(scratch_index, node)

class UpdateScratchFromNode(BaseAction[DefineScratchOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        scratch_inner_index = ScratchNodeIndex(arg2)
        node_index = NodeMainIndex(arg3)
        return cls(scratch_index, scratch_inner_index, node_index)

    def __init__(
        self,
        scratch_index: StateScratchIndex,
        scratch_inner_index: ScratchNodeIndex,
        node_index: NodeMainIndex,
    ):
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(scratch_inner_index, ScratchNodeIndex)
        assert isinstance(node_index, NodeMainIndex)
        super().__init__(scratch_index, scratch_inner_index, node_index)

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
    def node_index(self) -> NodeMainIndex:
        node_index = self.args[2]
        assert isinstance(node_index, NodeMainIndex)
        return node_index

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        state = full_state.current.state
        scratch_index = self.scratch_index
        scratch = scratch_index.find_in_outer_node(state)
        assert scratch is not None

        inner_node = self.node_index.find_in_node(full_state)
        assert inner_node is not None

        new_scratch = self.scratch_inner_index.replace_in_target(scratch, inner_node)
        assert new_scratch is not None

        return DefineScratchOutput(scratch_index, new_scratch.child)

class UpdateScratchFromFunction(BaseAction[DefineScratchOutput]):
    def __init__(
        self,
        scratch_index: StateScratchIndex,
        scratch_inner_index: ScratchNodeIndex,
        function_index: NodeMainIndex,
        args_group_index: StateArgsGroupIndex,
    ):
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(scratch_inner_index, ScratchNodeIndex)
        assert isinstance(function_index, NodeMainIndex)
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
    def function_index(self) -> NodeMainIndex:
        function_index = self.args[2]
        assert isinstance(function_index, NodeMainIndex)
        return function_index

    @property
    def args_group_index(self) -> StateArgsGroupIndex:
        args_group_index = self.args[3]
        assert isinstance(args_group_index, StateArgsGroupIndex)
        return args_group_index

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        state = full_state.current.state
        scratch_index = self.scratch_index
        scratch = scratch_index.find_in_outer_node(state)
        assert scratch is not None

        args_group = self.args_group_index.find_in_outer_node(state)
        assert args_group is not None

        function = self.function_index.find_in_node(full_state)
        assert isinstance(function, IFunction)

        function.as_node.validate()
        args_group.validate()
        inner_node = FunctionCall(function, args_group.inner_args_group.fill_with_void())

        new_scratch = self.scratch_inner_index.replace_in_target(scratch, inner_node)
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

    def apply(self, full_state: FullState) -> State:
        state = full_state.current.state
        index = self.index
        new_args_group = self.new_args_group

        assert index.value == len(state.args_outer_group.as_tuple) + 1
        for arg in new_args_group.scope.child.as_tuple:
            arg.as_node.validate()

        args_group = state.args_outer_group
        new_args = list(args_group.as_tuple) + [new_args_group]

        return State(
            definition_group=state.definition_group,
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

    def _run(self, full_state: FullState) -> CreateArgsGroupOutput:
        state = full_state.current.state
        args_amount = self.args_amount.to_int
        params_amount = self.params_amount
        optional_group: OptionalValueGroup[INode] = OptionalValueGroup.from_int(args_amount)
        new_args_group = PartialArgsGroup(
                ExtendedTypeGroup.default(params_amount),
                OpaqueScope.with_content(optional_group),
        )
        index = StateArgsGroupIndex(len(state.args_outer_group.as_tuple))
        return CreateArgsGroupOutput(index, new_args_group)

class CreateArgsGroupDynamicAction(BaseAction[CreateArgsGroupOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        node_index = NodeMainIndex(arg1)
        assert arg2 == 0
        assert arg3 == 0
        return cls(node_index)

    def __init__(self, node_index: NodeMainIndex):
        assert isinstance(node_index, NodeMainIndex)
        super().__init__(node_index)

    @property
    def node_index(self) -> NodeMainIndex:
        node_index = self.args[0]
        assert isinstance(node_index, NodeMainIndex)
        return node_index

    def _run(self, full_state: FullState) -> CreateArgsGroupOutput:
        state = full_state.current.state
        node_index = self.node_index
        new_args_group = node_index.find_in_node(full_state)
        assert isinstance(new_args_group, PartialArgsGroup)

        index = StateArgsGroupIndex(len(state.args_outer_group.as_tuple))

        return CreateArgsGroupOutput(index, new_args_group)

###########################################################
################## DEFINE ARGS GROUP ARG ##################
###########################################################

class DefineArgsGroupArgOutput(ActionOutput):
    def __init__(
        self,
        index: StateArgsGroupIndex,
        arg_index: NodeArgIndex,
        new_arg: INode,
    ):
        assert isinstance(index, StateArgsGroupIndex)
        assert isinstance(arg_index, NodeArgIndex)
        assert isinstance(new_arg, INode)
        super().__init__(index, arg_index, new_arg)

    @property
    def index(self) -> StateArgsGroupIndex:
        index = self.args[0]
        assert isinstance(index, StateArgsGroupIndex)
        return index

    @property
    def arg_index(self) -> NodeArgIndex:
        arg_index = self.args[1]
        assert isinstance(arg_index, NodeArgIndex)
        return arg_index

    @property
    def new_arg(self) -> INode:
        new_arg = self.args[2]
        assert isinstance(new_arg, INode)
        return new_arg

    def apply(self, full_state: FullState) -> State:
        state = full_state.current.state
        index = self.index
        arg_index = self.arg_index
        new_arg = self.new_arg

        new_arg.as_node.validate()

        args_group = index.find_in_outer_node(state)
        assert args_group is not None

        new_args_group = arg_index.replace_in_target(args_group, new_arg)
        assert isinstance(new_args_group, PartialArgsGroup)
        new_args_group = PartialArgsGroup(
            new_args_group.param_type_group,
            new_args_group.scope.normalize())

        new_state = index.replace_in_outer_target(state, new_args_group)
        assert new_state is not None

        return new_state

class DefineArgsGroupArgFromNodeAction(BaseAction[DefineArgsGroupArgOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        args_group_index = StateArgsGroupIndex(arg1)
        arg_index = NodeArgIndex(arg2)
        node_index = NodeMainIndex(arg3)
        return cls(args_group_index, arg_index, node_index)

    def __init__(
        self,
        args_group_index: StateArgsGroupIndex,
        arg_index: NodeArgIndex,
        node_index: NodeMainIndex,
    ):
        assert isinstance(args_group_index, StateArgsGroupIndex)
        assert isinstance(arg_index, NodeArgIndex)
        assert isinstance(node_index, NodeMainIndex)
        super().__init__(args_group_index, arg_index, node_index)

    @property
    def args_group_index(self) -> StateArgsGroupIndex:
        args_group_index = self.args[0]
        assert isinstance(args_group_index, StateArgsGroupIndex)
        return args_group_index

    @property
    def arg_index(self) -> NodeArgIndex:
        arg_index = self.args[1]
        assert isinstance(arg_index, NodeArgIndex)
        return arg_index

    @property
    def node_index(self) -> NodeMainIndex:
        node_index = self.args[2]
        assert isinstance(node_index, NodeMainIndex)
        return node_index

    def _run(self, full_state: FullState) -> DefineArgsGroupArgOutput:
        state = full_state.current.state
        args_group_index = self.args_group_index
        arg_index = self.arg_index
        node_index = self.node_index

        args_group = args_group_index.find_in_outer_node(state)
        assert args_group is not None

        new_arg = node_index.find_in_node(full_state)
        assert new_arg is not None

        return DefineArgsGroupArgOutput(args_group_index, arg_index, new_arg)
