import typing
from abc import ABC
from env.core import (
    INode,
    NodeArgIndex,
    DefaultGroup,
    Integer,
    Optional,
    ExtendedTypeGroup,
    IFunction,
    ISingleChild,
    OptionalValueGroup,
    LaxOpaqueScope,
    FunctionCall,
    TypeNode,
    IDefault,
    ISingleOptionalChild,
    IFromInt,
    IFromSingleChild,
    IsEmpty,
    CountableTypeGroup,
    IOptional,
    IInstantiable)
from env.state import (
    State,
    Scratch,
    ScratchGroup,
    StateScratchIndex,
    ScratchNodeIndex,
    StateArgsGroupIndex,
    PartialArgsOuterGroup,
    PartialArgsGroup)
from env.full_state import (
    FullState,
    FullStateIntIndex,
    MetaDefaultTypeIndex,
    MetaFromIntTypeIndex,
    MetaSingleChildTypeIndex,
    MetaFullStateIntIndexTypeIndex)
from env.action import (
    BasicAction,
    GeneralAction)

###########################################################
################## SCRATCH BASE ACTIONS ###################
###########################################################

class ScratchBaseActionOutput(GeneralAction, ISingleChild[StateScratchIndex], ABC):

    @classmethod
    def with_child(cls, child: StateScratchIndex) -> typing.Self:
        return cls.new(child)

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            StateScratchIndex,
        ]))

    @property
    def idx_index(self) -> int:
        return 0

    @property
    def child(self):
        return self.args[self.idx_index]

    def apply(self, full_state: FullState) -> State:
        raise NotImplementedError

class ScratchWithNodeBaseActionOutput(GeneralAction, ABC):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            StateScratchIndex,
            Optional[INode],
        ]))

    @property
    def idx_index(self) -> int:
        return 0

    @property
    def idx_node(self) -> int:
        return 1

    def apply(self, full_state: FullState) -> State:
        raise NotImplementedError

###########################################################
##################### CREATE SCRATCH ######################
###########################################################

class CreateScratchOutput(ScratchWithNodeBaseActionOutput, IInstantiable):

    def apply(self, full_state: FullState) -> State:
        index = typing.cast(StateScratchIndex, self.args[self.idx_index])
        node = typing.cast(Optional[INode], self.args[self.idx_node])

        state = full_state.current_state.apply().cast(State)
        scratch_group = state.scratch_group.apply().cast(ScratchGroup)
        assert index.as_int == len(scratch_group.as_tuple) + 1
        new_args = list(scratch_group.as_tuple) + [Scratch.with_content(node)]

        return State(
            state.definition_group.apply(),
            state.args_outer_group.apply(),
            scratch_group.func(*new_args),
        )

class CreateScratch(
    BasicAction[CreateScratchOutput],
    IDefault,
    ISingleOptionalChild[StateScratchIndex],
    IInstantiable,
):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        clone_index: Optional[StateScratchIndex] = (
            Optional.create()
            if arg1 == 0
            else Optional(StateScratchIndex(arg1)))
        assert arg2 == 0
        assert arg3 == 0
        return cls(clone_index)

    @classmethod
    def create(cls) -> typing.Self:
        return cls(Optional.create())

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            Optional[StateScratchIndex],
        ]))

    @property
    def idx_clone_index(self) -> int:
        return 0

    @property
    def child(self):
        return self.args[self.idx_clone_index]

    def _run(self, full_state: FullState) -> CreateScratchOutput:
        clone_index = typing.cast(Optional[StateScratchIndex], self.args[self.idx_clone_index])

        state = full_state.current_state.apply().cast(State)
        scratch_group = state.scratch_group.apply().cast(ScratchGroup)
        index = StateScratchIndex(len(scratch_group.as_tuple))
        if clone_index.value is None:
            return CreateScratchOutput(index, Optional.create())
        scratch = clone_index.value.find_in_outer_node(state).value_or_raise
        assert isinstance(scratch, Scratch)
        scratch.validate()
        node = scratch.child.apply().cast(IOptional).value

        return CreateScratchOutput(index, Optional(node) if node is not None else Optional())

class DeleteScratchOutput(ScratchBaseActionOutput, IInstantiable):

    def apply(self, full_state: FullState) -> State:
        index = typing.cast(StateScratchIndex, self.args[self.idx_index])
        state = full_state.current_state.apply().cast(State)
        new_state = index.remove_in_outer_target(state).value_or_raise
        return new_state

###########################################################
##################### DEFINE SCRATCH ######################
###########################################################

class DefineScratchOutput(ScratchWithNodeBaseActionOutput, IInstantiable):

    def apply(self, full_state: FullState) -> State:
        index = typing.cast(StateScratchIndex, self.args[self.idx_index])
        node = typing.cast(Optional[INode], self.args[self.idx_node])

        state = full_state.current_state.apply().cast(State)
        scratch = Scratch.with_content(Optional(node))
        new_state = index.replace_in_outer_target(state, scratch).value_or_raise
        assert new_state is not None

        return new_state

class ClearScratch(BasicAction[DefineScratchOutput], IInstantiable):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        assert arg2 == 0
        assert arg3 == 0
        return cls(scratch_index)

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            StateScratchIndex,
        ]))

    @property
    def idx_scratch_index(self) -> int:
        return 0

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = typing.cast(StateScratchIndex, self.args[self.idx_scratch_index])
        return DefineScratchOutput(scratch_index, Optional.create())

class DefineScratchFromDefault(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        type_index = MetaDefaultTypeIndex(arg2)
        assert arg3 == 0
        return cls(scratch_index, type_index)

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            StateScratchIndex,
            MetaDefaultTypeIndex,
        ]))

    @property
    def idx_scratch_index(self) -> int:
        return 0

    @property
    def idx_type_index(self) -> int:
        return 1

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = typing.cast(StateScratchIndex, self.args[self.idx_scratch_index])
        type_index = typing.cast(MetaDefaultTypeIndex, self.args[self.idx_type_index])

        node_type = type_index.find_in_outer_node(full_state).value_or_raise
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, IDefault)
        content = node_type.type.create()

        return DefineScratchOutput(scratch_index, Optional(content))

class DefineScratchFromInt(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        type_index = MetaFromIntTypeIndex(arg2)
        index_value = Integer(arg3)
        return cls(scratch_index, type_index, index_value)

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            StateScratchIndex,
            MetaFromIntTypeIndex,
            Integer,
        ]))

    @property
    def idx_scratch_index(self) -> int:
        return 0

    @property
    def idx_type_index(self) -> int:
        return 1

    @property
    def idx_index_value(self) -> int:
        return 2

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = typing.cast(StateScratchIndex, self.args[self.idx_scratch_index])
        type_index = typing.cast(MetaFromIntTypeIndex, self.args[self.idx_type_index])
        index_value = typing.cast(Integer, self.args[self.idx_index_value])

        node_type = type_index.find_in_outer_node(full_state).value_or_raise
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, IFromInt)
        content = node_type.type.from_int(index_value.as_int)

        return DefineScratchOutput(scratch_index, Optional(content))

class DefineScratchFromSingleArg(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        type_index = MetaSingleChildTypeIndex(arg2)
        arg = StateScratchIndex(arg3)
        return cls(scratch_index, type_index, arg)

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            StateScratchIndex,
            MetaSingleChildTypeIndex,
            StateScratchIndex,
        ]))

    @property
    def idx_scratch_index(self) -> int:
        return 0

    @property
    def idx_type_index(self) -> int:
        return 1

    @property
    def idx_arg(self) -> int:
        return 2

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = typing.cast(StateScratchIndex, self.args[self.idx_scratch_index])
        type_index = typing.cast(MetaSingleChildTypeIndex, self.args[self.idx_type_index])
        arg = typing.cast(StateScratchIndex, self.args[self.idx_arg])

        node_type = type_index.find_in_outer_node(full_state).value_or_raise
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, IFromSingleChild)
        content = node_type.type.with_child(arg)

        return DefineScratchOutput(scratch_index, Optional(content))

class DefineScratchFromIntIndex(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        type_index = MetaFullStateIntIndexTypeIndex(arg2)
        index_value = Integer(arg3)
        return cls(scratch_index, type_index, index_value)

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            StateScratchIndex,
            MetaFullStateIntIndexTypeIndex,
            Integer,
        ]))

    @property
    def idx_scratch_index(self) -> int:
        return 0

    @property
    def idx_type_index(self) -> int:
        return 1

    @property
    def idx_index_value(self) -> int:
        return 2

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = typing.cast(StateScratchIndex, self.args[self.idx_scratch_index])
        type_index = typing.cast(MetaFullStateIntIndexTypeIndex, self.args[self.idx_type_index])
        index_value = typing.cast(Integer, self.args[self.idx_index_value])

        node_type = type_index.find_in_outer_node(full_state).value_or_raise
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, FullStateIntIndex)
        node_index = typing.cast(
            FullStateIntIndex[INode],
            node_type.type.from_int(index_value.as_int))
        content = node_index.find_in_outer_node(full_state).value_or_raise
        content = IsEmpty.with_optional(content).value_or_raise

        return DefineScratchOutput(scratch_index, Optional(content))

class DefineScratchFromFunctionWithIntArg(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        source_index = StateScratchIndex(arg2)
        int_arg = Integer(arg3)
        return cls(scratch_index, source_index, int_arg)

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            StateScratchIndex,
            StateScratchIndex,
            Integer,
        ]))

    @property
    def idx_scratch_index(self) -> int:
        return 0

    @property
    def idx_source_index(self) -> int:
        return 1

    @property
    def idx_int_arg(self) -> int:
        return 2

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = typing.cast(StateScratchIndex, self.args[self.idx_scratch_index])
        source_index = typing.cast(StateScratchIndex, self.args[self.idx_source_index])
        int_arg = typing.cast(Integer, self.args[self.idx_int_arg])

        state = full_state.current_state.apply().cast(State)
        source_scratch = source_index.find_in_outer_node(state).value_or_raise
        assert isinstance(source_scratch, Scratch)
        source_scratch.validate()

        content = source_scratch.child.apply().cast(IOptional).value
        assert content is not None

        content = FunctionCall(
            typing.cast(IFunction, content),
            DefaultGroup(int_arg))

        return DefineScratchOutput(scratch_index, Optional(content))

class DefineScratchFromFunctionWithSingleArg(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        source_index = StateScratchIndex(arg2)
        single_arg = StateScratchIndex(arg3)
        return cls(scratch_index, source_index, single_arg)

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            StateScratchIndex,
            StateScratchIndex,
            StateScratchIndex,
        ]))

    @property
    def idx_scratch_index(self) -> int:
        return 0

    @property
    def idx_source_index(self) -> int:
        return 1

    @property
    def idx_single_arg_index(self) -> int:
        return 2

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = typing.cast(StateScratchIndex, self.args[self.idx_scratch_index])
        source_index = typing.cast(StateScratchIndex, self.args[self.idx_source_index])
        single_arg_index = typing.cast(StateScratchIndex, self.args[self.idx_single_arg_index])

        state = full_state.current_state.apply().cast(State)
        source_scratch = source_index.find_in_outer_node(state).value_or_raise
        assert isinstance(source_scratch, Scratch)
        source_scratch.validate()

        content = source_scratch.child.apply().cast(IOptional).value
        assert content is not None

        single_arg_outer = single_arg_index.find_in_outer_node(state).value_or_raise
        assert isinstance(single_arg_outer, Scratch)
        single_arg_outer.validate()

        single_arg = single_arg_outer.child.apply().cast(IOptional).value
        assert single_arg is not None

        content = FunctionCall(
            typing.cast(IFunction, content),
            DefaultGroup(single_arg))

        return DefineScratchOutput(scratch_index, Optional(content))

class DefineScratchFromFunctionWithArgs(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        source_index = StateScratchIndex(arg2)
        args_group_index: Optional[StateArgsGroupIndex] = (
            Optional.create()
            if arg3 == 0
            else Optional(StateArgsGroupIndex(arg3)))
        return cls(scratch_index, source_index, args_group_index)

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            StateScratchIndex,
            StateScratchIndex,
            Optional[StateArgsGroupIndex],
        ]))

    @property
    def idx_scratch_index(self) -> int:
        return 0

    @property
    def idx_source_index(self) -> int:
        return 1

    @property
    def idx_args_group_index(self) -> int:
        return 2

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = typing.cast(StateScratchIndex, self.args[self.idx_scratch_index])
        source_index = typing.cast(StateScratchIndex, self.args[self.idx_source_index])
        args_group_index = typing.cast(
            Optional[StateArgsGroupIndex],
            self.args[self.idx_args_group_index]).value

        state = full_state.current_state.apply().cast(State)
        source_scratch = source_index.find_in_outer_node(state).value_or_raise
        assert isinstance(source_scratch, Scratch)
        source_scratch.validate()

        content = source_scratch.child.apply().cast(IOptional).value
        assert content is not None

        if args_group_index is not None:
            args_group = args_group_index.find_in_outer_node(state).value_or_raise
            assert isinstance(args_group, PartialArgsGroup)
            args_group.validate()
        else:
            args_group = PartialArgsGroup.create()

        content = FunctionCall(
            typing.cast(IFunction, content),
            args_group.scope_child.fill_with_void())

        return DefineScratchOutput(scratch_index, Optional(content))

class UpdateScratchFromAnother(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        scratch_inner_index = ScratchNodeIndex(arg2)
        source_index = StateScratchIndex(arg3)
        return cls(scratch_index, scratch_inner_index, source_index)

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            StateScratchIndex,
            ScratchNodeIndex,
            StateScratchIndex,
        ]))

    @property
    def idx_scratch_index(self) -> int:
        return 0

    @property
    def idx_scratch_inner_index(self) -> int:
        return 1

    @property
    def idx_source_index(self) -> int:
        return 2

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = typing.cast(StateScratchIndex, self.args[self.idx_scratch_index])
        scratch_inner_index = typing.cast(ScratchNodeIndex, self.args[self.idx_scratch_inner_index])
        source_index = typing.cast(StateScratchIndex, self.args[self.idx_source_index])

        state = full_state.current_state.apply().cast(State)
        scratch = scratch_index.find_in_outer_node(state).value_or_raise
        source_scratch = source_index.find_in_outer_node(state).value_or_raise
        inner_content = source_scratch.child.value_or_raise

        new_content = scratch_inner_index.replace_in_target(
            scratch,
            inner_content,
        ).value_or_raise

        return DefineScratchOutput(scratch_index, Optional(new_content))

###########################################################
#################### CREATE ARGS GROUP ####################
###########################################################

class CreateArgsGroupOutput(GeneralAction, IInstantiable):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            StateArgsGroupIndex,
            PartialArgsGroup,
        ]))

    @property
    def idx_index(self) -> int:
        return 0

    @property
    def idx_new_args_group(self) -> int:
        return 1

    def apply(self, full_state: FullState) -> State:
        index = typing.cast(StateArgsGroupIndex, self.args[self.idx_index])
        new_args_group = typing.cast(PartialArgsGroup, self.args[self.idx_new_args_group])

        state = full_state.current_state.apply().cast(State)
        args_outer_group = state.args_outer_group.apply().cast(PartialArgsOuterGroup)
        assert index.as_int == len(args_outer_group.as_tuple) + 1
        scope_child = new_args_group.scope_child.apply().cast(OptionalValueGroup)
        for arg in scope_child.as_tuple:
            arg.as_node.validate()

        args_outer_group = state.args_outer_group.apply().cast(PartialArgsOuterGroup)
        new_args = list(args_outer_group.as_tuple) + [new_args_group]

        return State(
            state.definition_group.apply(),
            args_outer_group.func(*new_args),
            state.scratch_group.apply(),
        )

class CreateArgsGroup(
    BasicAction[CreateArgsGroupOutput],
    IInstantiable,
):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        args_amount = Integer(arg1)
        param_types_index: Optional[StateScratchIndex] = (
            Optional.create()
            if arg2 == 0
            else Optional(StateScratchIndex(arg2)))
        args_group_source_index: Optional[StateArgsGroupIndex] = (
            Optional.create()
            if arg3 == 0
            else Optional(StateArgsGroupIndex(arg3)))
        return cls(args_amount, param_types_index, args_group_source_index)

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            Integer,
            Optional[StateScratchIndex],
            Optional[StateArgsGroupIndex],
        ]))

    @property
    def idx_args_amount(self) -> int:
        return 0

    @property
    def idx_param_types_index(self) -> int:
        return 1

    @property
    def idx_args_group_source_index(self) -> int:
        return 2

    def _run(self, full_state: FullState) -> CreateArgsGroupOutput:
        args_amount = typing.cast(Integer, self.args[self.idx_args_amount]).as_int
        param_types_index = typing.cast(
            Optional[StateScratchIndex],
            self.args[self.idx_param_types_index],
        ).value
        args_group_source_index = typing.cast(
            Optional[StateArgsGroupIndex],
            self.args[self.idx_args_group_source_index],
        ).value

        state = full_state.current_state.apply().cast(State)

        if args_group_source_index is not None:
            args_group_source = args_group_source_index.find_in_outer_node(state).value_or_raise
            assert isinstance(args_group_source, PartialArgsGroup)
            args_group_source.validate()
            type_group = args_group_source.param_type_group.apply().cast(ExtendedTypeGroup)
            optional_group = args_group_source.scope_child.apply().cast(OptionalValueGroup)

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

            type_group_aux = param_types.child.apply().cast(IOptional).value
            assert isinstance(type_group_aux, ExtendedTypeGroup)
            type_group = type_group_aux

            inner_group = type_group.group
            if isinstance(inner_group, CountableTypeGroup):
                assert len(inner_group.as_tuple) == args_amount

        scope = LaxOpaqueScope.with_content(optional_group)
        new_args_group = PartialArgsGroup(type_group, scope)

        index = StateArgsGroupIndex(
            len(state.args_outer_group.apply().cast(PartialArgsOuterGroup).as_tuple))

        return CreateArgsGroupOutput(index, new_args_group)

###########################################################
################## DEFINE ARGS GROUP ARG ##################
###########################################################

class DefineArgsGroupArgOutput(GeneralAction, IInstantiable):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            StateArgsGroupIndex,
            NodeArgIndex,
            INode,
        ]))

    @property
    def idx_group_index(self) -> int:
        return 0

    @property
    def idx_arg_index(self) -> int:
        return 1

    @property
    def idx_new_arg(self) -> int:
        return 2

    def apply(self, full_state: FullState) -> State:
        group_index = typing.cast(StateArgsGroupIndex, self.args[self.idx_group_index])
        arg_index = typing.cast(NodeArgIndex, self.args[self.idx_arg_index])
        new_arg = typing.cast(INode, self.args[self.idx_new_arg])

        state = full_state.current_state.apply().cast(State)
        new_arg.as_node.validate()
        args_group = group_index.find_in_outer_node(state).value_or_raise

        new_args_group = arg_index.replace_in_target(args_group, new_arg).value_or_raise
        assert isinstance(new_args_group, PartialArgsGroup)
        new_args_group = PartialArgsGroup(
            new_args_group.param_type_group.apply(),
            new_args_group.scope.apply())

        new_state = group_index.replace_in_outer_target(state, new_args_group).value_or_raise

        return new_state

class DefineArgsGroup(
    BasicAction[DefineArgsGroupArgOutput],
    IInstantiable,
):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        args_group_index = StateArgsGroupIndex(arg1)
        arg_index = NodeArgIndex(arg2)
        scratch_index = StateScratchIndex(arg3)
        return cls(args_group_index, arg_index, scratch_index)

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            StateArgsGroupIndex,
            NodeArgIndex,
            StateScratchIndex,
        ]))

    @property
    def idx_args_group_index(self) -> int:
        return 0

    @property
    def idx_arg_index(self) -> int:
        return 1

    @property
    def idx_scratch_index(self) -> int:
        return 2

    def _run(self, full_state: FullState) -> DefineArgsGroupArgOutput:
        args_group_index = typing.cast(StateArgsGroupIndex, self.args[self.idx_args_group_index])
        arg_index = typing.cast(NodeArgIndex, self.args[self.idx_arg_index])
        scratch_index = typing.cast(StateScratchIndex, self.args[self.idx_scratch_index])

        state = full_state.current_state.apply().cast(State)
        args_group_index.find_in_outer_node(state).raise_if_empty()
        new_arg = scratch_index.find_in_outer_node(state).value_or_raise

        return DefineArgsGroupArgOutput(args_group_index, arg_index, new_arg)
