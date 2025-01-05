import typing
from environment.core import (
    INode,
    NodeArgIndex,
    DefaultGroup,
    Integer,
    Optional,
    ExtendedTypeGroup,
    IFunction,
    ISingleChild,
    OptionalValueGroup,
    OpaqueScope,
    FunctionCall,
    TypeNode,
    IDefault,
    ISingleOptionalChild,
    IFromInt,
    IFromSingleChild,
    IsEmpty,
    CountableTypeGroup)
from environment.state import (
    State,
    Scratch,
    StateScratchIndex,
    ScratchNodeIndex,
    StateArgsGroupIndex,
    PartialArgsGroup)
from environment.full_state import (
    FullState,
    FullStateIntIndex,
    MetaDefaultTypeIndex,
    MetaFromIntTypeIndex,
    MetaSingleChildTypeIndex,
    MetaFullStateIntIndexTypeIndex)
from environment.action import (
    BaseAction,
    GeneralAction)

class BasicActionGenerator(INode):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        raise NotImplementedError

###########################################################
################## ASCRATCH BASE ACTIONS ##################
###########################################################

class ScratchBaseActionOutput(GeneralAction, ISingleChild[StateScratchIndex]):

    @classmethod
    def with_child(cls, child: StateScratchIndex) -> typing.Self:
        return cls.new(child)

    def __init__(self, index: StateScratchIndex):
        super().__init__(index)

    @property
    def index(self) -> StateScratchIndex:
        index = self.args[0]
        assert isinstance(index, StateScratchIndex)
        return index

    @property
    def child(self):
        return self.index

    def apply(self, full_state: FullState) -> State:
        raise NotImplementedError

class ScratchWithNodeBaseActionOutput(GeneralAction):

    def __init__(self, index: StateScratchIndex, node: Optional[INode]):
        super().__init__(index, node)

    @property
    def index(self) -> StateScratchIndex:
        index = self.args[0]
        assert isinstance(index, StateScratchIndex)
        return index

    @property
    def node(self) -> Optional:
        node = self.args[1]
        assert isinstance(node, Optional)
        return node

    def apply(self, full_state: FullState) -> State:
        raise NotImplementedError

###########################################################
##################### CREATE SCRATCH ######################
###########################################################

class CreateScratchOutput(ScratchWithNodeBaseActionOutput):

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

class CreateScratch(
    BaseAction[CreateScratchOutput],
    IDefault,
    ISingleOptionalChild[StateScratchIndex],
    BasicActionGenerator,
):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        clone_index = Optional.create() if arg1 == 0 else Optional(StateScratchIndex(arg1))
        assert arg2 == 0
        assert arg3 == 0
        return cls(clone_index)

    @classmethod
    def create(cls) -> typing.Self:
        return cls(Optional.create())

    def __init__(self, clone_index: Optional[StateScratchIndex]):
        super().__init__(clone_index)

    @property
    def clone_index(self) -> Optional[StateScratchIndex]:
        clone_index = self.args[0]
        assert isinstance(clone_index, Optional)
        if clone_index.value is not None:
            assert isinstance(clone_index.value, StateScratchIndex)
        return clone_index

    @property
    def child(self):
        return self.clone_index

    def _run(self, full_state: FullState) -> CreateScratchOutput:
        state = full_state.current.state
        index = StateScratchIndex(len(state.scratch_group.as_tuple))
        if self.clone_index.value is None:
            return CreateScratchOutput(index, Optional.create())
        scratch = self.clone_index.value.find_in_outer_node(state).value_or_raise
        assert isinstance(scratch, Scratch)
        scratch.validate()
        node = scratch.child.value
        return CreateScratchOutput(index, Optional(node))

class DeleteScratchOutput(ScratchBaseActionOutput):

    def apply(self, full_state: FullState) -> State:
        state = full_state.current.state
        new_state = self.index.remove_in_outer_target(state).value_or_raise
        return new_state

###########################################################
##################### DEFINE SCRATCH ######################
###########################################################

class DefineScratchOutput(ScratchWithNodeBaseActionOutput):

    def apply(self, full_state: FullState) -> State:
        state = full_state.current.state
        scratch = Scratch.with_content(Optional(self.node))
        new_state = self.index.replace_in_outer_target(state, scratch).value_or_raise
        assert new_state is not None
        return new_state

class ClearScratch(BaseAction[DefineScratchOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        assert arg2 == 0
        assert arg3 == 0
        return cls(scratch_index)

    def __init__(self, scratch_index: StateScratchIndex):
        super().__init__(scratch_index)

    @property
    def scratch_index(self) -> StateScratchIndex:
        scratch_index = self.args[0]
        assert isinstance(scratch_index, StateScratchIndex)
        return scratch_index

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.scratch_index
        return DefineScratchOutput(scratch_index, Optional.create())

class DefineScratchFromDefault(BaseAction[DefineScratchOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        type_index = MetaDefaultTypeIndex(arg2)
        assert arg3 == 0
        return cls(scratch_index, type_index)

    def __init__(self, scratch_index: StateScratchIndex, type_index: MetaDefaultTypeIndex):
        super().__init__(scratch_index, type_index)

    @property
    def scratch_index(self) -> StateScratchIndex:
        scratch_index = self.args[0]
        assert isinstance(scratch_index, StateScratchIndex)
        return scratch_index

    @property
    def type_index(self) -> MetaDefaultTypeIndex:
        type_index = self.args[1]
        assert isinstance(type_index, MetaDefaultTypeIndex)
        return type_index

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.scratch_index
        node_type = self.type_index.find_in_outer_node(full_state).value_or_raise
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, IDefault)
        content = node_type.type.create()
        return DefineScratchOutput(scratch_index, Optional(content))

class DefineScratchFromInt(BaseAction[DefineScratchOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        type_index = MetaFromIntTypeIndex(arg2)
        index_value = Integer(arg3)
        return cls(scratch_index, type_index, index_value)

    def __init__(
        self,
        scratch_index: StateScratchIndex,
        type_index: MetaFromIntTypeIndex,
        index_value: Integer,
    ):
        super().__init__(scratch_index, type_index, index_value)

    @property
    def scratch_index(self) -> StateScratchIndex:
        scratch_index = self.args[0]
        assert isinstance(scratch_index, StateScratchIndex)
        return scratch_index

    @property
    def type_index(self) -> MetaFromIntTypeIndex:
        type_index = self.args[1]
        assert isinstance(type_index, MetaFromIntTypeIndex)
        return type_index

    @property
    def index_value(self) -> Integer:
        index_value = self.args[2]
        assert isinstance(index_value, Integer)
        return index_value

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.scratch_index
        node_type = self.type_index.find_in_outer_node(full_state).value_or_raise
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, IFromInt)
        content = node_type.type.from_int(self.index_value.to_int)
        return DefineScratchOutput(scratch_index, Optional(content))

class DefineScratchFromSingleArg(BaseAction[DefineScratchOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        type_index = MetaSingleChildTypeIndex(arg2)
        arg = StateScratchIndex(arg3)
        return cls(scratch_index, type_index, arg)

    def __init__(
        self,
        scratch_index: StateScratchIndex,
        type_index: MetaSingleChildTypeIndex,
        arg: StateScratchIndex,
    ):
        super().__init__(scratch_index, type_index, arg)

    @property
    def scratch_index(self) -> StateScratchIndex:
        scratch_index = self.args[0]
        assert isinstance(scratch_index, StateScratchIndex)
        return scratch_index

    @property
    def type_index(self) -> MetaSingleChildTypeIndex:
        type_index = self.args[1]
        assert isinstance(type_index, MetaSingleChildTypeIndex)
        return type_index

    @property
    def arg(self) -> StateScratchIndex:
        arg = self.args[2]
        assert isinstance(arg, StateScratchIndex)
        return arg

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.scratch_index
        node_type = self.type_index.find_in_outer_node(full_state).value_or_raise
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, IFromSingleChild)
        content = node_type.type.with_child(self.arg)
        return DefineScratchOutput(scratch_index, Optional(content))

class DefineScratchFromIntIndex(BaseAction[DefineScratchOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        type_index = MetaFullStateIntIndexTypeIndex(arg2)
        index_value = Integer(arg3)
        return cls(scratch_index, type_index, index_value)

    def __init__(
        self,
        scratch_index: StateScratchIndex,
        type_index: MetaFullStateIntIndexTypeIndex,
        index_value: Integer,
    ):
        super().__init__(scratch_index, type_index, index_value)

    @property
    def scratch_index(self) -> StateScratchIndex:
        scratch_index = self.args[0]
        assert isinstance(scratch_index, StateScratchIndex)
        return scratch_index

    @property
    def type_index(self) -> MetaFullStateIntIndexTypeIndex:
        type_index = self.args[1]
        assert isinstance(type_index, MetaFullStateIntIndexTypeIndex)
        return type_index

    @property
    def index_value(self) -> Integer:
        index_value = self.args[2]
        assert isinstance(index_value, Integer)
        return index_value

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.scratch_index
        node_type = self.type_index.find_in_outer_node(full_state).value_or_raise
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, FullStateIntIndex)
        node_index = typing.cast(
            FullStateIntIndex[INode],
            node_type.type.from_int(self.index_value.to_int))
        content = node_index.find_in_outer_node(full_state).value_or_raise
        content = IsEmpty.with_optional(content).value_or_raise
        return DefineScratchOutput(scratch_index, Optional(content))

class DefineScratchFromFunctionWithIntArg(BaseAction[DefineScratchOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        source_index = StateScratchIndex(arg2)
        int_arg = Integer(arg3)
        return cls(scratch_index, source_index, int_arg)

    def __init__(
        self,
        scratch_index: StateScratchIndex,
        source_index: StateScratchIndex,
        int_arg: Integer,
    ):
        super().__init__(scratch_index, source_index, int_arg)

    @property
    def scratch_index(self) -> StateScratchIndex:
        scratch_index = self.args[0]
        assert isinstance(scratch_index, StateScratchIndex)
        return scratch_index

    @property
    def source_index(self) -> StateScratchIndex:
        source_index = self.args[1]
        assert isinstance(source_index, StateScratchIndex)
        return source_index

    @property
    def int_arg(self) -> Integer:
        int_arg = self.args[2]
        assert isinstance(int_arg, Integer)
        return int_arg

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        state = full_state.current.state
        scratch_index = self.scratch_index
        source_index = self.source_index
        int_arg = self.int_arg

        source_scratch = source_index.find_in_outer_node(state).value_or_raise
        assert isinstance(source_scratch, Scratch)
        source_scratch.validate()

        content = source_scratch.child.value
        assert content is not None

        content = FunctionCall(
            typing.cast(IFunction, content),
            DefaultGroup(int_arg))

        return DefineScratchOutput(scratch_index, Optional(content))

class DefineScratchFromFunctionWithSingleArg(BaseAction[DefineScratchOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        source_index = StateScratchIndex(arg2)
        single_arg = StateScratchIndex(arg3)
        return cls(scratch_index, source_index, single_arg)

    def __init__(
        self,
        scratch_index: StateScratchIndex,
        source_index: StateScratchIndex,
        single_arg: StateScratchIndex,
    ):
        super().__init__(scratch_index, source_index, single_arg)

    @property
    def scratch_index(self) -> StateScratchIndex:
        scratch_index = self.args[0]
        assert isinstance(scratch_index, StateScratchIndex)
        return scratch_index

    @property
    def source_index(self) -> StateScratchIndex:
        source_index = self.args[1]
        assert isinstance(source_index, StateScratchIndex)
        return source_index

    @property
    def single_arg(self) -> StateScratchIndex:
        single_arg = self.args[2]
        assert isinstance(single_arg, StateScratchIndex)
        return single_arg

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        state = full_state.current.state
        scratch_index = self.scratch_index
        source_index = self.source_index
        single_arg = self.single_arg

        source_scratch = source_index.find_in_outer_node(state).value_or_raise
        assert isinstance(source_scratch, Scratch)
        source_scratch.validate()

        content = source_scratch.child.value
        assert content is not None

        content = FunctionCall(
            typing.cast(IFunction, content),
            DefaultGroup(single_arg))

        return DefineScratchOutput(scratch_index, Optional(content))

class DefineScratchFromFunctionWithArgs(BaseAction[DefineScratchOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        source_index = StateScratchIndex(arg2)
        args_group_index = Optional.create() if arg3 == 0 else Optional(StateArgsGroupIndex(arg3))
        return cls(scratch_index, source_index, args_group_index)

    def __init__(
        self,
        scratch_index: StateScratchIndex,
        source_index: StateScratchIndex,
        args_group_index: Optional[StateArgsGroupIndex],
    ):
        super().__init__(scratch_index, source_index, args_group_index)

    @property
    def scratch_index(self) -> StateScratchIndex:
        scratch_index = self.args[0]
        assert isinstance(scratch_index, StateScratchIndex)
        return scratch_index

    @property
    def source_index(self) -> StateScratchIndex:
        source_index = self.args[1]
        assert isinstance(source_index, StateScratchIndex)
        return source_index

    @property
    def args_group_index(self) -> Optional[StateArgsGroupIndex]:
        args_group_index = self.args[2]
        assert isinstance(args_group_index, Optional)
        if args_group_index.value is not None:
            assert isinstance(args_group_index.value, StateArgsGroupIndex)
        return args_group_index

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        state = full_state.current.state
        scratch_index = self.scratch_index
        source_index = self.source_index
        args_group_index = self.args_group_index.value

        source_scratch = source_index.find_in_outer_node(state).value_or_raise
        assert isinstance(source_scratch, Scratch)
        source_scratch.validate()

        content = source_scratch.child.value
        assert content is not None

        if args_group_index is not None:
            args_group = args_group_index.find_in_outer_node(state).value_or_raise
            assert isinstance(args_group, PartialArgsGroup)
            args_group.validate()
        else:
            args_group = PartialArgsGroup.create()

        content = FunctionCall(
            typing.cast(IFunction, content),
            args_group.inner_args_group.fill_with_void())

        return DefineScratchOutput(scratch_index, Optional(content))

class UpdateScratchFromAnother(BaseAction[DefineScratchOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        scratch_inner_index = ScratchNodeIndex(arg2)
        source_index = StateScratchIndex(arg2)
        return cls(scratch_index, scratch_inner_index, source_index)

    def __init__(
        self,
        scratch_index: StateScratchIndex,
        scratch_inner_index: ScratchNodeIndex,
        source_index: StateScratchIndex,
    ):
        super().__init__(scratch_index, scratch_inner_index, source_index)

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
    def source_index(self) -> StateScratchIndex:
        source_index = self.args[2]
        assert isinstance(source_index, StateScratchIndex)
        return source_index

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        state = full_state.current.state
        scratch_index = self.scratch_index
        scratch = scratch_index.find_in_outer_node(state).value_or_raise
        source_scratch = self.source_index.find_in_outer_node(state).value_or_raise
        inner_content = source_scratch.child.value_or_raise

        new_content = self.scratch_inner_index.replace_in_target(
            scratch,
            inner_content,
        ).value_or_raise

        return DefineScratchOutput(scratch_index, Optional(new_content))

###########################################################
#################### CREATE ARGS GROUP ####################
###########################################################

class CreateArgsGroupOutput(GeneralAction):

    def __init__(
        self,
        index: StateArgsGroupIndex,
        new_args_group: PartialArgsGroup,
    ):
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

class CreateArgsGroup(BaseAction[CreateArgsGroupOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        args_amount = Integer(arg1)
        param_types_index = Optional.create() if arg2 == 0 else Optional(StateScratchIndex(arg2))
        args_group_source_index = (
            Optional.create()
            if arg3 == 0
            else Optional(StateArgsGroupIndex(arg3)))
        return cls(args_amount, param_types_index, args_group_source_index)

    def __init__(
        self,
        args_amount: Integer,
        param_types_index: Optional[StateScratchIndex],
        args_group_source_index: Optional[StateArgsGroupIndex],
    ):
        super().__init__(args_amount, param_types_index, args_group_source_index)

    @property
    def args_amount(self) -> Integer:
        args_amount = self.args[0]
        assert isinstance(args_amount, Integer)
        return args_amount

    @property
    def param_types_index(self) -> Optional[StateScratchIndex]:
        param_types_index = self.args[1]
        assert isinstance(param_types_index, Optional)
        if param_types_index.value is not None:
            assert isinstance(param_types_index.value, StateScratchIndex)
        return typing.cast(Optional[StateScratchIndex], param_types_index)

    @property
    def args_group_source_index(self) -> Optional[StateArgsGroupIndex]:
        args_group_source_index = self.args[2]
        assert isinstance(args_group_source_index, Optional)
        if args_group_source_index.value is not None:
            assert isinstance(args_group_source_index.value, StateArgsGroupIndex)
        return typing.cast(Optional[StateArgsGroupIndex], args_group_source_index)

    def _run(self, full_state: FullState) -> CreateArgsGroupOutput:
        state = full_state.current.state
        args_amount = self.args_amount.to_int
        param_types_index = self.param_types_index.value
        args_group_source_index = self.args_group_source_index.value

        if args_group_source_index is not None:
            args_group_source = args_group_source_index.find_in_outer_node(state).value_or_raise
            assert isinstance(args_group_source, PartialArgsGroup)
            args_group_source.validate()
            type_group = args_group_source.param_type_group
            optional_group = args_group_source.inner_args_group

            if args_amount != len(optional_group.as_tuple):
                type_group = type_group.new_amount(args_amount)
                optional_group = optional_group.new_amount(args_amount)
        else:
            type_group = ExtendedTypeGroup.create()
            optional_group = OptionalValueGroup.from_int(args_amount)

        if param_types_index is not None:
            param_types = param_types_index.find_in_outer_node(state).value_or_raise
            assert isinstance(param_types, Scratch)
            param_types.validate()

            type_group_aux = param_types.child.value
            assert isinstance(type_group_aux, ExtendedTypeGroup)
            type_group = type_group_aux

            inner_group = type_group.group
            if isinstance(inner_group, CountableTypeGroup):
                assert len(inner_group.as_tuple) == args_amount

        scope = OpaqueScope.with_content(optional_group)
        new_args_group = PartialArgsGroup(type_group, scope)

        index = StateArgsGroupIndex(len(state.args_outer_group.as_tuple))

        return CreateArgsGroupOutput(index, new_args_group)

###########################################################
################## DEFINE ARGS GROUP ARG ##################
###########################################################

class DefineArgsGroupArgOutput(GeneralAction):

    def __init__(
        self,
        index: StateArgsGroupIndex,
        arg_index: NodeArgIndex,
        new_arg: INode,
    ):
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

        args_group = index.find_in_outer_node(state).value_or_raise

        new_args_group = arg_index.replace_in_target(args_group, new_arg).value_or_raise
        assert isinstance(new_args_group, PartialArgsGroup)
        new_args_group = PartialArgsGroup(
            new_args_group.param_type_group,
            new_args_group.scope.normalize())

        new_state = index.replace_in_outer_target(state, new_args_group).value_or_raise

        return new_state

class DefineArgsGroup(BaseAction[DefineArgsGroupArgOutput], BasicActionGenerator):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        args_group_index = StateArgsGroupIndex(arg1)
        arg_index = NodeArgIndex(arg2)
        scratch_index = StateScratchIndex(arg3)
        return cls(args_group_index, arg_index, scratch_index)

    def __init__(
        self,
        args_group_index: StateArgsGroupIndex,
        arg_index: NodeArgIndex,
        scratch_index: StateScratchIndex,
    ):
        super().__init__(args_group_index, arg_index, scratch_index)

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
    def scratch_index(self) -> StateScratchIndex:
        scratch_index = self.args[2]
        assert isinstance(scratch_index, StateScratchIndex)
        return scratch_index

    def _run(self, full_state: FullState) -> DefineArgsGroupArgOutput:
        state = full_state.current.state
        args_group_index = self.args_group_index
        arg_index = self.arg_index
        scratch_index = self.scratch_index

        args_group_index.find_in_outer_node(state).raise_if_empty()
        new_arg = scratch_index.find_in_outer_node(state).value_or_raise

        return DefineArgsGroupArgOutput(args_group_index, arg_index, new_arg)
