import typing
from abc import ABC
from env.core import (
    INode,
    NodeArgIndex,
    DefaultGroup,
    Integer,
    Optional,
    ExtendedTypeGroup,
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

    idx_index = 1

    @classmethod
    def with_child(cls, child: StateScratchIndex) -> typing.Self:
        return cls.new(child)

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            StateScratchIndex,
        ]))

    @property
    def child(self):
        return self.nested_arg(self.idx_index).apply()

    def apply(self, full_state: FullState) -> State:
        raise NotImplementedError

class ScratchWithNodeBaseActionOutput(GeneralAction, ABC):

    idx_index = 1
    idx_node = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            StateScratchIndex,
            IOptional[INode],
        ]))

    @property
    def child(self):
        return self.nested_arg(self.idx_index).apply()

    def apply(self, full_state: FullState) -> State:
        raise NotImplementedError

###########################################################
##################### CREATE SCRATCH ######################
###########################################################

class CreateScratchOutput(ScratchWithNodeBaseActionOutput, IInstantiable):

    def apply(self, full_state: FullState) -> State:
        index = self.nested_arg(self.idx_index).apply()
        node = self.nested_arg(self.idx_node).apply()
        assert isinstance(index, StateScratchIndex)
        assert isinstance(node, IOptional)

        state = full_state.current_state.apply().cast(State)
        scratch_group = state.scratch_group.apply().cast(ScratchGroup)
        assert index.as_int == len(scratch_group.as_tuple) + 1
        new_node = Scratch.with_optional(node)
        new_args = list(scratch_group.as_tuple) + [new_node]

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

    idx_clone_index = 1

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
    def child(self):
        return self.nested_arg(self.idx_clone_index).apply()

    def _run(self, full_state: FullState) -> CreateScratchOutput:
        clone_index = self.nested_arg(self.idx_clone_index).apply()
        assert isinstance(clone_index, Optional)

        state = full_state.current_state.apply().cast(State)
        scratch_group = state.scratch_group.apply().cast(ScratchGroup)
        index = StateScratchIndex(len(scratch_group.as_tuple))
        if clone_index.value is None:
            return CreateScratchOutput(index, Optional.create())
        scratch = clone_index.value.find_in_outer_node(state).value_or_raise
        assert isinstance(scratch, Scratch)
        scratch.validate()
        node = scratch.child.apply().cast(IOptional)

        return CreateScratchOutput(index, node)

class DeleteScratchOutput(ScratchBaseActionOutput, IInstantiable):

    def apply(self, full_state: FullState) -> State:
        index = self.nested_arg(self.idx_index).apply()
        assert isinstance(index, StateScratchIndex)
        state = full_state.current_state.apply().cast(State)
        new_state = index.remove_in_outer_target(state).value_or_raise
        return new_state

###########################################################
##################### DEFINE SCRATCH ######################
###########################################################

class DefineScratchOutput(ScratchWithNodeBaseActionOutput, IInstantiable):

    def apply(self, full_state: FullState) -> State:
        index = self.nested_arg(self.idx_index).apply()
        node = self.nested_arg(self.idx_node).apply()
        assert isinstance(index, StateScratchIndex)
        assert isinstance(node, IOptional)

        state = full_state.current_state.apply().cast(State)
        scratch = Scratch.with_optional(node)
        new_state = index.replace_in_outer_target(state, scratch).value_or_raise
        assert new_state is not None

        return new_state

class ClearScratch(BasicAction[DefineScratchOutput], IInstantiable):

    idx_scratch_index = 1

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

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.nested_arg(self.idx_scratch_index).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        return DefineScratchOutput(scratch_index, Optional.create())

class DefineScratchFromDefault(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    idx_scratch_index = 1
    idx_type_index = 2

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

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.nested_arg(self.idx_scratch_index).apply()
        type_index = self.nested_arg(self.idx_type_index).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(type_index, MetaDefaultTypeIndex)

        node_type = type_index.find_in_outer_node(full_state).value_or_raise
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, IDefault)
        content = node_type.type.create()

        return DefineScratchOutput(scratch_index, Optional(content))

class DefineScratchFromInt(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    idx_scratch_index = 1
    idx_type_index = 2
    idx_index_value = 3

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

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.nested_arg(self.idx_scratch_index).apply()
        type_index = self.nested_arg(self.idx_type_index).apply()
        index_value = self.nested_arg(self.idx_index_value).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(type_index, MetaFromIntTypeIndex)
        assert isinstance(index_value, Integer)

        node_type = type_index.find_in_outer_node(full_state).value_or_raise
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, IFromInt)
        content = node_type.type.from_int(index_value.as_int)

        return DefineScratchOutput(scratch_index, Optional(content))

class DefineScratchFromSingleArg(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    idx_scratch_index = 1
    idx_type_index = 2
    idx_arg = 3

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

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.nested_arg(self.idx_scratch_index).apply()
        type_index = self.nested_arg(self.idx_type_index).apply()
        arg = self.nested_arg(self.idx_arg).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(type_index, MetaSingleChildTypeIndex)
        assert isinstance(arg, StateScratchIndex)

        node_type = type_index.find_in_outer_node(full_state).value_or_raise
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, IFromSingleChild)
        content = node_type.type.with_child(arg)

        return DefineScratchOutput(scratch_index, Optional(content))

class DefineScratchFromIntIndex(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    idx_scratch_index = 1
    idx_type_index = 2
    idx_index_value = 3

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

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.nested_arg(self.idx_scratch_index).apply()
        type_index = self.nested_arg(self.idx_type_index).apply()
        index_value = self.nested_arg(self.idx_index_value).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(type_index, MetaFullStateIntIndexTypeIndex)
        assert isinstance(index_value, Integer)

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

    idx_scratch_index = 1
    idx_source_index = 2
    idx_int_arg = 3

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

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.nested_arg(self.idx_scratch_index).apply()
        source_index = self.nested_arg(self.idx_source_index).apply()
        int_arg = self.nested_arg(self.idx_int_arg).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(source_index, StateScratchIndex)
        assert isinstance(int_arg, Integer)

        state = full_state.current_state.apply().cast(State)
        source_scratch = source_index.find_in_outer_node(state).value_or_raise
        assert isinstance(source_scratch, Scratch)
        source_scratch.validate()

        content = source_scratch.child.apply().cast(IOptional).value
        assert content is not None

        fn_call = FunctionCall(content, DefaultGroup(int_arg))

        return DefineScratchOutput(scratch_index, Optional(fn_call))

class DefineScratchFromFunctionWithSingleArg(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    idx_scratch_index = 1
    idx_source_index = 2
    idx_single_arg_index = 3

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

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.nested_arg(self.idx_scratch_index).apply()
        source_index = self.nested_arg(self.idx_source_index).apply()
        single_arg_index = self.nested_arg(self.idx_single_arg_index).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(source_index, StateScratchIndex)
        assert isinstance(single_arg_index, StateScratchIndex)

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

        fn_call = FunctionCall(content, DefaultGroup(single_arg))

        return DefineScratchOutput(scratch_index, Optional(fn_call))

class DefineScratchFromFunctionWithArgs(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    idx_scratch_index = 1
    idx_source_index = 2
    idx_args_group_index = 3

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

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.nested_arg(self.idx_scratch_index).apply()
        source_index = self.nested_arg(self.idx_source_index).apply()
        args_group_index = self.nested_arg(self.idx_args_group_index).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(source_index, StateScratchIndex)
        assert isinstance(args_group_index, Optional)

        state = full_state.current_state.apply().cast(State)
        source_scratch = source_index.find_in_outer_node(state).value_or_raise
        assert isinstance(source_scratch, Scratch)
        source_scratch.validate()

        content = source_scratch.child.apply().cast(IOptional).value
        assert content is not None

        args_group_index_value = args_group_index.value

        if args_group_index.value is not None:
            assert isinstance(args_group_index_value, StateArgsGroupIndex)
            args_group = args_group_index_value.find_in_outer_node(state).value_or_raise
            assert isinstance(args_group, PartialArgsGroup)
            args_group.validate()
        else:
            args_group = PartialArgsGroup.create()

        fn_call = FunctionCall(content, args_group.fill_with_void())

        return DefineScratchOutput(scratch_index, Optional(fn_call))

class UpdateScratchFromAnother(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    idx_scratch_index = 1
    idx_scratch_inner_index = 2
    idx_source_index = 3

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

    def _run(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.nested_arg(self.idx_scratch_index).apply()
        scratch_inner_index = self.nested_arg(self.idx_scratch_inner_index).apply()
        source_index = self.nested_arg(self.idx_source_index).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(scratch_inner_index, ScratchNodeIndex)
        assert isinstance(source_index, StateScratchIndex)

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

    idx_index = 1
    idx_new_args_group = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            StateArgsGroupIndex,
            PartialArgsGroup,
        ]))

    def apply(self, full_state: FullState) -> State:
        index = self.nested_arg(self.idx_index).apply()
        new_args_group = self.nested_arg(self.idx_new_args_group).apply()
        assert isinstance(index, StateArgsGroupIndex)
        assert isinstance(new_args_group, PartialArgsGroup)

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

    idx_args_amount = 1
    idx_param_types_index = 2
    idx_args_group_source_index = 3

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

    def _run(self, full_state: FullState) -> CreateArgsGroupOutput:
        args_amount = self.nested_arg(self.idx_args_amount).apply()
        param_types_index = self.nested_arg(self.idx_param_types_index).apply()
        args_group_source_index = self.nested_arg(self.idx_args_group_source_index).apply()
        assert isinstance(args_amount, Integer)
        assert isinstance(param_types_index, Optional)
        assert isinstance(args_group_source_index, Optional)

        state = full_state.current_state.apply().cast(State)

        if args_group_source_index.value is not None:
            args_group_source = args_group_source_index.value.find_in_outer_node(state).value_or_raise
            assert isinstance(args_group_source, PartialArgsGroup)
            args_group_source.validate()
            type_group = args_group_source.param_type_group.apply().cast(ExtendedTypeGroup)
            optional_group = args_group_source.scope_child.apply().cast(OptionalValueGroup)

            if args_amount.as_int != len(optional_group.as_tuple):
                type_group = type_group.new_amount(args_amount.as_int)
                optional_group = optional_group.new_amount(args_amount.as_int)
        else:
            type_group = ExtendedTypeGroup.create()
            optional_group = OptionalValueGroup.from_int(args_amount.as_int)

        if param_types_index.value is not None:
            param_types = param_types_index.value.find_in_outer_node(state).value_or_raise
            assert isinstance(param_types, Scratch)
            param_types.validate()

            type_group_aux = param_types.child.apply().cast(IOptional).value
            assert isinstance(type_group_aux, ExtendedTypeGroup)
            type_group = type_group_aux

            inner_group = type_group.group
            if isinstance(inner_group, CountableTypeGroup):
                assert len(inner_group.as_tuple) == args_amount.as_int

        scope = LaxOpaqueScope.with_content(optional_group)
        new_args_group = PartialArgsGroup(type_group, scope)

        index = StateArgsGroupIndex(
            len(state.args_outer_group.apply().cast(PartialArgsOuterGroup).as_tuple))

        return CreateArgsGroupOutput(index, new_args_group)

###########################################################
################## DEFINE ARGS GROUP ARG ##################
###########################################################

class DefineArgsGroupArgOutput(GeneralAction, IInstantiable):

    idx_group_index = 1
    idx_arg_index = 2
    idx_new_arg = 3

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            StateArgsGroupIndex,
            NodeArgIndex,
            INode,
        ]))

    def apply(self, full_state: FullState) -> State:
        group_index = self.nested_arg(self.idx_group_index).apply()
        arg_index = self.nested_arg(self.idx_arg_index).apply()
        new_arg = self.nested_arg(self.idx_new_arg).apply()
        assert isinstance(group_index, StateArgsGroupIndex)
        assert isinstance(arg_index, NodeArgIndex)
        assert isinstance(new_arg, INode)

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

    idx_args_group_index = 1
    idx_arg_index = 2
    idx_scratch_index = 3

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

    def _run(self, full_state: FullState) -> DefineArgsGroupArgOutput:
        args_group_index = self.nested_arg(self.idx_args_group_index).apply()
        arg_index = self.nested_arg(self.idx_arg_index).apply()
        scratch_index = self.nested_arg(self.idx_scratch_index).apply()
        assert isinstance(args_group_index, StateArgsGroupIndex)
        assert isinstance(arg_index, NodeArgIndex)
        assert isinstance(scratch_index, StateScratchIndex)

        state = full_state.current_state.apply().cast(State)
        args_group_index.find_in_outer_node(state).raise_if_empty()
        new_arg = scratch_index.find_in_outer_node(state).value_or_raise

        return DefineArgsGroupArgOutput(args_group_index, arg_index, new_arg)
