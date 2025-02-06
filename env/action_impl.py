#pylint: disable=too-many-lines
import typing
from abc import ABC
from env.core import (
    INode,
    IInheritableNode,
    NodeArgIndex,
    DefaultGroup,
    Integer,
    Optional,
    Protocol,
    ISingleChild,
    FunctionCall,
    TypeNode,
    IDefault,
    ISingleOptionalChild,
    IFromInt,
    IFromSingleNode,
    IsEmpty,
    CountableTypeGroup,
    IOptional,
    NestedArgIndexGroup,
    IntGroup,
    RunInfo,
    Eq,
    Not,
    IInstantiable,
)
from env.state import (
    State,
    Scratch,
    ScratchGroup,
    StateScratchIndex,
    ScratchNodeIndex,
    StateArgsGroupIndex,
    PartialArgsOuterGroup,
    StateMetaInfo,
    PartialArgsGroup,
    IGoal,
    IGoalAchieved,
    Goal,
    StateDynamicGoalIndex,
    DynamicGoalGroup,
    DynamicGoal,
)
from env.meta_env import MetaInfo
from env.full_state import (
    FullState,
    FullStateIntIndex,
    MetaDefaultTypeIndex,
    MetaFromIntTypeIndex,
    MetaSingleChildTypeIndex,
    MetaFullStateIntIndexTypeIndex,
)
from env.action import (
    IBasicAction,
    BasicAction,
    GeneralAction,
)

###########################################################
################### STATE META ACTIONS ####################
###########################################################

class VerifyGoalOutput(GeneralAction, IInstantiable):

    idx_nested_args_indices = 1
    idx_node = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            Optional[NestedArgIndexGroup].as_type(),
            INode.as_type(),
        ))

    def apply(self, full_state: FullState) -> State:
        nested_args_wrapper = self.inner_arg(
            self.idx_nested_args_indices
        ).apply().cast(Optional[NestedArgIndexGroup])
        node = self.inner_arg(self.idx_node).apply()

        goal = full_state.nested_arg((FullState.idx_meta, MetaInfo.idx_goal)).apply().cast(IGoal)
        nested_args_indices = nested_args_wrapper.value

        if nested_args_indices is not None:
            assert isinstance(nested_args_indices, NestedArgIndexGroup)
            goal = nested_args_indices.apply(goal.as_node).cast(IGoal)

        assert isinstance(goal, Goal)

        state = full_state.current_state.apply().cast(State)
        assert isinstance(node, goal.eval_param_type())

        meta_info = state.meta_info.apply().cast(StateMetaInfo)
        goal_achieved = meta_info.goal_achieved.apply().cast(IGoalAchieved)
        if nested_args_indices is not None:
            assert isinstance(nested_args_indices, NestedArgIndexGroup)
            goal_achieved = nested_args_indices.apply(goal_achieved.as_node).cast(IGoalAchieved)
        Not(goal_achieved).raise_on_false()

        goal.evaluate(state, node).raise_on_false()

        new_meta_info = meta_info.apply_goal_achieved(nested_args_wrapper)
        new_state = state.with_new_args(meta_info=new_meta_info)

        return new_state

class VerifyGoal(
    BasicAction[VerifyGoalOutput],
    IInstantiable,
):

    idx_scratch_index_nested_indices = 1
    idx_type_index = 2
    idx_index_value = 3

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index_nested_indices: Optional[StateScratchIndex] = (
            Optional.create()
            if arg1 == 0
            else Optional(StateScratchIndex(arg1)))
        type_index = MetaFromIntTypeIndex(arg2)
        index_value = Integer(arg3)
        return cls(scratch_index_nested_indices, type_index, index_value)

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            Optional[StateScratchIndex].as_type(),
            MetaFromIntTypeIndex.as_type(),
            Integer.as_type(),
        ))

    def _run_action(self, full_state: FullState) -> VerifyGoalOutput:
        scratch_index_nested_indices = self.inner_arg(
            self.idx_scratch_index_nested_indices
        ).apply().cast(Optional[StateScratchIndex])
        type_index = self.inner_arg(self.idx_type_index).apply()
        index_value = self.inner_arg(self.idx_index_value).apply()
        assert isinstance(scratch_index_nested_indices, Optional)
        assert isinstance(type_index, MetaFromIntTypeIndex)
        assert isinstance(index_value, Integer)

        state = full_state.current_state.apply().cast(State)
        scratch_index = scratch_index_nested_indices.value
        nested = Optional[NestedArgIndexGroup]()
        if scratch_index is not None:
            assert isinstance(scratch_index, StateScratchIndex)
            scratch = scratch_index.find_in_outer_node(state).value_or_raise
            assert isinstance(scratch, Scratch)
            scratch.validate()
            content = scratch.value_or_raise
            assert isinstance(content, NestedArgIndexGroup)
            nested = Optional(content)

        node_type = type_index.find_in_outer_node(full_state).value_or_raise
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, IFromInt)
        content = node_type.type.from_int(index_value.as_int)

        return VerifyGoalOutput(nested, content)

class CreateDynamicGoalOutput(GeneralAction, IInstantiable):

    idx_dynamic_goal_index = 1
    idx_goal_expr = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateDynamicGoalIndex.as_type(),
            IGoal.as_type(),
        ))

    def apply(self, full_state: FullState) -> State:
        index = self.inner_arg(self.idx_dynamic_goal_index).apply().cast(StateDynamicGoalIndex)
        goal_expr = self.inner_arg(self.idx_goal_expr).apply().cast(IGoal)

        state = full_state.current_state.apply().cast(State)
        state_meta = state.meta_info.apply().cast(StateMetaInfo)
        dynamic_goal_group = state_meta.dynamic_goal_group.apply().cast(DynamicGoalGroup)

        items = [item for item in dynamic_goal_group.as_tuple]
        Eq.from_ints(index.as_int, len(items) + 1).raise_on_false()
        items.append(DynamicGoal.from_goal_expr(goal_expr))

        new_dynamic_goal_group = DynamicGoalGroup.from_items(items)
        new_meta_info = state_meta.with_new_args(
            dynamic_goal_group=new_dynamic_goal_group,
        )
        new_state = state.with_new_args(meta_info=new_meta_info)

        return new_state

class CreateDynamicGoal(
    BasicAction[CreateDynamicGoalOutput],
    IInstantiable,
):

    idx_scratch_index_goal = 1

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index_goal = StateScratchIndex(arg1)
        Eq.from_ints(arg2, 0).raise_on_false()
        Eq.from_ints(arg3, 0).raise_on_false()
        return cls(scratch_index_goal)

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(StateScratchIndex.as_type()))

    def _run_action(self, full_state: FullState) -> CreateDynamicGoalOutput:
        scratch_index_goal = self.inner_arg(
            self.idx_scratch_index_goal
        ).apply().cast(StateScratchIndex)

        state = full_state.current_state.apply().cast(State)
        scratch = scratch_index_goal.find_in_outer_node(state).value_or_raise
        assert isinstance(scratch, Scratch)
        scratch.validate()

        goal_expr = scratch.value_or_raise
        assert isinstance(goal_expr, IGoal)

        dynamic_goal_group = state.meta_info.apply().cast(
            StateMetaInfo
        ).dynamic_goal_group.apply().cast(DynamicGoalGroup)
        dynamic_goal_index = StateDynamicGoalIndex(len(dynamic_goal_group.as_tuple) + 1)

        return CreateDynamicGoalOutput(dynamic_goal_index, goal_expr)

class VerifyDynamicGoalOutput(GeneralAction, IInstantiable):

    idx_dynamic_goal = 1
    idx_nested_args_indices = 2
    idx_node = 3

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateDynamicGoalIndex.as_type(),
            Optional[NestedArgIndexGroup].as_type(),
            INode.as_type(),
        ))

    def apply(self, full_state: FullState) -> State:
        dynamic_goal_index = self.inner_arg(
            self.idx_dynamic_goal
        ).apply().cast(StateDynamicGoalIndex)
        nested_args_wrapper = self.inner_arg(
            self.idx_nested_args_indices
        ).apply().cast(Optional[NestedArgIndexGroup])
        node = self.inner_arg(self.idx_node).apply()

        state = full_state.current_state.apply().cast(State)
        dynamic_goal = dynamic_goal_index.find_in_outer_node(state).value_or_raise

        goal = dynamic_goal.goal_expr.apply().cast(IGoal)
        nested_args_indices = nested_args_wrapper.value

        if nested_args_indices is not None:
            assert isinstance(nested_args_indices, NestedArgIndexGroup)
            goal = nested_args_indices.apply(goal.as_node).cast(IGoal)

        assert isinstance(goal, Goal)
        assert isinstance(node, goal.eval_param_type())

        goal_achieved = dynamic_goal.goal_achieved.apply().cast(IGoalAchieved)
        if nested_args_indices is not None:
            assert isinstance(nested_args_indices, NestedArgIndexGroup)
            goal_achieved = nested_args_indices.apply(goal_achieved.as_node).cast(IGoalAchieved)
        Not(goal_achieved).raise_on_false()

        goal.evaluate(state, node).raise_on_false()

        dynamic_goal = dynamic_goal.apply_goal_achieved(nested_args_wrapper)
        new_state = dynamic_goal_index.replace_in_outer_target(
            state,
            dynamic_goal,
        ).value_or_raise

        return new_state

class VerifyDynamicGoal(
    BasicAction[VerifyDynamicGoalOutput],
    IInstantiable,
):

    idx_dynamic_node_index = 1
    idx_scratch_index_nested_indices = 2
    idx_scratch_index_content = 3

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        dynamic_node_index = StateDynamicGoalIndex(arg1)
        scratch_index_nested_indices: Optional[StateScratchIndex] = (
            Optional.create()
            if arg2 == 0
            else Optional(StateScratchIndex(arg2)))
        scratch_content_index = StateScratchIndex(arg3)
        return cls(dynamic_node_index, scratch_index_nested_indices, scratch_content_index)

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateDynamicGoalIndex.as_type(),
            Optional[StateScratchIndex].as_type(),
            StateScratchIndex.as_type(),
        ))

    def _run_action(self, full_state: FullState) -> VerifyDynamicGoalOutput:
        dynamic_node_index = self.inner_arg(
            self.idx_dynamic_node_index
        ).apply().cast(StateDynamicGoalIndex)
        scratch_index_nested_indices = self.inner_arg(
            self.idx_scratch_index_nested_indices
        ).apply().cast(Optional[StateScratchIndex])
        scratch_content_index = self.inner_arg(
            self.idx_scratch_index_content
        ).apply().cast(StateScratchIndex)

        state = full_state.current_state.apply().cast(State)
        nest_scratch_index = scratch_index_nested_indices.value
        nested = Optional[NestedArgIndexGroup]()
        if nest_scratch_index is not None:
            assert isinstance(nest_scratch_index, StateScratchIndex)
            scratch = nest_scratch_index.find_in_outer_node(state).value_or_raise
            assert isinstance(scratch, Scratch)
            scratch.validate()
            content = scratch.value_or_raise
            assert isinstance(content, NestedArgIndexGroup)
            nested = Optional(content)

        scratch = scratch_content_index.find_in_outer_node(state).value_or_raise
        content = scratch.value_or_raise

        return VerifyDynamicGoalOutput(dynamic_node_index, nested, content)

class DeleteDynamicGoalOutput(GeneralAction, IBasicAction[FullState], IInstantiable):

    idx_dynamic_goal_index = 1

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        dynamic_goal_index = StateDynamicGoalIndex(arg1)
        Eq.from_ints(arg2, 0).raise_on_false()
        Eq.from_ints(arg3, 0).raise_on_false()
        return cls(dynamic_goal_index)

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(StateDynamicGoalIndex.as_type()))

    def apply(self, full_state: FullState) -> State:
        dynamic_goal_index = self.inner_arg(self.idx_dynamic_goal_index).apply()
        assert isinstance(dynamic_goal_index, StateDynamicGoalIndex)
        state = full_state.current_state.apply().cast(State)
        new_state = dynamic_goal_index.remove_in_outer_target(state).value_or_raise
        return new_state

###########################################################
################## SCRATCH BASE ACTIONS ###################
###########################################################

class ScratchBaseActionOutput(GeneralAction, ISingleChild[StateScratchIndex], ABC):

    idx_index = 1

    @classmethod
    def with_node(cls, node: StateScratchIndex) -> typing.Self:
        return cls.new(node)

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(StateScratchIndex.as_type()))

    @property
    def child(self):
        return self.inner_arg(self.idx_index).apply()

    def apply(self, full_state: FullState) -> State:
        raise NotImplementedError

class ScratchWithNodeBaseActionOutput(GeneralAction, ABC):

    idx_index = 1
    idx_node = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateScratchIndex.as_type(),
            Scratch.as_type(),
        ))

    @property
    def child(self):
        return self.inner_arg(self.idx_index).apply()

    def apply(self, full_state: FullState) -> State:
        raise NotImplementedError

###########################################################
##################### MANAGE SCRATCH ######################
###########################################################

class CreateScratchOutput(ScratchWithNodeBaseActionOutput, IInstantiable):

    def apply(self, full_state: FullState) -> State:
        index = self.inner_arg(self.idx_index).apply()
        new_node = self.inner_arg(self.idx_node).apply()
        assert isinstance(index, StateScratchIndex)
        assert isinstance(new_node, Scratch)

        state = full_state.current_state.apply().cast(State)
        scratch_group = state.scratch_group.apply().cast(ScratchGroup)
        Eq.from_ints(index.as_int, len(scratch_group.as_tuple) + 1).raise_on_false()
        new_args = list(scratch_group.as_tuple) + [new_node]

        return state.with_new_args(
            scratch_group=scratch_group.func(*new_args),
        )

class CreateScratch(
    BasicAction[CreateScratchOutput],
    IDefault,
    ISingleOptionalChild[StateScratchIndex],
    IInstantiable,
):

    idx_clone_index = 1

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        clone_index: Optional[StateScratchIndex] = (
            Optional.create()
            if arg1 == 0
            else Optional(StateScratchIndex(arg1)))
        Eq.from_ints(arg2, 0).raise_on_false()
        Eq.from_ints(arg3, 0).raise_on_false()
        return cls(clone_index)

    @classmethod
    def create(cls) -> typing.Self:
        return cls(Optional.create())

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            Optional[StateScratchIndex].as_type(),
        ))

    @property
    def child(self):
        return self.inner_arg(self.idx_clone_index).apply()

    def _run_action(self, full_state: FullState) -> CreateScratchOutput:
        clone_index = self.inner_arg(
            self.idx_clone_index
        ).apply().cast(Optional[StateScratchIndex])

        state = full_state.current_state.apply().cast(State)
        scratch_group = state.scratch_group.apply().cast(ScratchGroup)
        index = StateScratchIndex(len(scratch_group.as_tuple) + 1)
        if clone_index.value is None:
            return CreateScratchOutput(index, Scratch.create())
        scratch = clone_index.value.find_in_outer_node(state).value_or_raise
        assert isinstance(scratch, Scratch)
        scratch.validate()

        return CreateScratchOutput(index, scratch)

class DeleteScratchOutput(ScratchBaseActionOutput, IBasicAction[FullState], IInstantiable):

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        index = StateScratchIndex(arg1)
        Eq.from_ints(arg2, 0).raise_on_false()
        Eq.from_ints(arg3, 0).raise_on_false()
        return cls(index)

    def apply(self, full_state: FullState) -> State:
        index = self.inner_arg(self.idx_index).apply()
        assert isinstance(index, StateScratchIndex)
        state = full_state.current_state.apply().cast(State)
        new_state = index.remove_in_outer_target(state).value_or_raise
        return new_state

###########################################################
##################### DEFINE SCRATCH ######################
###########################################################

class DefineScratchOutput(ScratchWithNodeBaseActionOutput, IInstantiable):

    def apply(self, full_state: FullState) -> State:
        index = self.inner_arg(self.idx_index).apply()
        scratch = self.inner_arg(self.idx_node).apply()
        assert isinstance(index, StateScratchIndex)
        assert isinstance(scratch, Scratch)

        state = full_state.current_state.apply().cast(State)
        new_state = index.replace_in_outer_target(state, scratch).value_or_raise

        return new_state

class ClearScratch(BasicAction[DefineScratchOutput], IInstantiable):

    idx_scratch_index = 1

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        Eq.from_ints(arg2, 0).raise_on_false()
        Eq.from_ints(arg3, 0).raise_on_false()
        return cls(scratch_index)

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(StateScratchIndex.as_type()))

    def _run_action(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        return DefineScratchOutput(scratch_index, Scratch.create())

class DefineScratchFromDefault(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    idx_scratch_index = 1
    idx_type_index = 2

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        type_index = MetaDefaultTypeIndex(arg2)
        Eq.from_ints(arg3, 0).raise_on_false()
        return cls(scratch_index, type_index)

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateScratchIndex.as_type(),
            MetaDefaultTypeIndex.as_type(),
        ))

    def _run_action(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        type_index = self.inner_arg(self.idx_type_index).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(type_index, MetaDefaultTypeIndex)

        node_type = type_index.find_in_outer_node(full_state).value_or_raise
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, IDefault)
        content = node_type.type.create()

        return DefineScratchOutput(scratch_index, Scratch(content))

class DefineScratchFromInt(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    idx_scratch_index = 1
    idx_type_index = 2
    idx_index_value = 3

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        type_index = MetaFromIntTypeIndex(arg2)
        index_value = Integer(arg3)
        return cls(scratch_index, type_index, index_value)

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateScratchIndex.as_type(),
            MetaFromIntTypeIndex.as_type(),
            Integer.as_type(),
        ))

    def _run_action(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        type_index = self.inner_arg(self.idx_type_index).apply()
        index_value = self.inner_arg(self.idx_index_value).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(type_index, MetaFromIntTypeIndex)
        assert isinstance(index_value, Integer)

        node_type = type_index.find_in_outer_node(full_state).value_or_raise
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, IFromInt)
        content = node_type.type.from_int(index_value.as_int)

        return DefineScratchOutput(scratch_index, Scratch(content))

class DefineScratchFromSingleArg(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    idx_scratch_index = 1
    idx_type_index = 2
    idx_arg = 3

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        type_index = MetaSingleChildTypeIndex(arg2)
        arg = StateScratchIndex(arg3)
        return cls(scratch_index, type_index, arg)

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateScratchIndex.as_type(),
            MetaSingleChildTypeIndex.as_type(),
            StateScratchIndex.as_type(),
        ))

    def _run_action(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        type_index = self.inner_arg(self.idx_type_index).apply()
        arg_index = self.inner_arg(self.idx_arg).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(type_index, MetaSingleChildTypeIndex)
        assert isinstance(arg_index, StateScratchIndex)

        node_type = type_index.find_in_outer_node(full_state).value_or_raise
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, IFromSingleNode)

        state = full_state.current_state.apply().cast(State)
        scratch = arg_index.find_in_outer_node(state).value_or_raise
        arg = scratch.value_or_raise

        content = node_type.type.with_node(arg)

        return DefineScratchOutput(scratch_index, Scratch(content))

class DefineScratchFromIntIndex(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    idx_scratch_index = 1
    idx_type_index = 2
    idx_index_value = 3

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        type_index = MetaFullStateIntIndexTypeIndex(arg2)
        index_value = Integer(arg3)
        return cls(scratch_index, type_index, index_value)

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateScratchIndex.as_type(),
            MetaFullStateIntIndexTypeIndex.as_type(),
            Integer.as_type(),
        ))

    def _run_action(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        type_index = self.inner_arg(self.idx_type_index).apply()
        index_value = self.inner_arg(self.idx_index_value).apply()
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

        return DefineScratchOutput(scratch_index, Scratch(content))

class DefineScratchFromFunctionWithIntArg(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    idx_scratch_index = 1
    idx_source_index = 2
    idx_int_arg = 3

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        source_index = StateScratchIndex(arg2)
        int_arg = Integer(arg3)
        return cls(scratch_index, source_index, int_arg)

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateScratchIndex.as_type(),
            StateScratchIndex.as_type(),
            Integer.as_type(),
        ))

    def _run_action(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        source_index = self.inner_arg(self.idx_source_index).apply()
        int_arg = self.inner_arg(self.idx_int_arg).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(source_index, StateScratchIndex)
        assert isinstance(int_arg, Integer)

        state = full_state.current_state.apply().cast(State)
        source_scratch = source_index.find_in_outer_node(state).value_or_raise
        assert isinstance(source_scratch, Scratch)
        source_scratch.validate()

        content = source_scratch.value
        assert content is not None

        fn_call: INode | None = None

        if isinstance(content, TypeNode):
            t = content.type
            if issubclass(t, IFromInt):
                fn_call = t.from_int(int_arg.as_int)
            else:
                assert issubclass(t, IInheritableNode)
                fn_call = t.new(int_arg)
        else:
            fn_call = FunctionCall(content, IntGroup(int_arg))

        assert isinstance(fn_call, INode)
        fn_call.as_node.validate()

        return DefineScratchOutput(scratch_index, Scratch(fn_call))

class DefineScratchFromFunctionWithSingleArg(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    idx_scratch_index = 1
    idx_source_index = 2
    idx_single_arg_index = 3

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        source_index = StateScratchIndex(arg2)
        single_arg = StateScratchIndex(arg3)
        return cls(scratch_index, source_index, single_arg)

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateScratchIndex.as_type(),
            StateScratchIndex.as_type(),
            StateScratchIndex.as_type(),
        ))

    def _run_action(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        source_index = self.inner_arg(self.idx_source_index).apply()
        single_arg_index = self.inner_arg(self.idx_single_arg_index).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(source_index, StateScratchIndex)
        assert isinstance(single_arg_index, StateScratchIndex)

        state = full_state.current_state.apply().cast(State)
        source_scratch = source_index.find_in_outer_node(state).value_or_raise
        assert isinstance(source_scratch, Scratch)
        source_scratch.validate()

        content = source_scratch.value_or_raise

        single_arg_outer = single_arg_index.find_in_outer_node(state).value_or_raise
        assert isinstance(single_arg_outer, Scratch)
        single_arg_outer.validate()

        single_arg = single_arg_outer.value_or_raise
        fn_call: INode | None = None

        if isinstance(content, TypeNode):
            t = content.type
            if issubclass(t, ISingleChild):
                fn_call = t.with_node(single_arg)
            else:
                assert issubclass(t, IInheritableNode)
                fn_call = t.new(single_arg)
        else:
            fn_call = FunctionCall(content, DefaultGroup(single_arg))

        assert isinstance(fn_call, INode)
        fn_call.as_node.validate()

        return DefineScratchOutput(scratch_index, Scratch(fn_call))

class DefineScratchFromFunctionWithArgs(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    idx_scratch_index = 1
    idx_source_index = 2
    idx_args_group_index = 3

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        source_index = StateScratchIndex(arg2)
        args_group_index: Optional[StateArgsGroupIndex] = (
            Optional.create()
            if arg3 == 0
            else Optional(StateArgsGroupIndex(arg3)))
        return cls(scratch_index, source_index, args_group_index)

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateScratchIndex.as_type(),
            StateScratchIndex.as_type(),
            Optional[StateArgsGroupIndex].as_type(),
        ))

    def _run_action(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        source_index = self.inner_arg(self.idx_source_index).apply()
        args_group_index = self.inner_arg(self.idx_args_group_index).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(source_index, StateScratchIndex)
        assert isinstance(args_group_index, Optional)

        state = full_state.current_state.apply().cast(State)
        source_scratch = source_index.find_in_outer_node(state).value_or_raise
        assert isinstance(source_scratch, Scratch)
        source_scratch.validate()

        content = source_scratch.value_or_raise

        args_group_index_value = args_group_index.value

        if args_group_index.value is not None:
            assert isinstance(args_group_index_value, StateArgsGroupIndex)
            args_group = args_group_index_value.find_in_outer_node(state).value_or_raise
            assert isinstance(args_group, PartialArgsGroup)
            args_group.validate()
        else:
            args_group = PartialArgsGroup.create()

        filled_args_group = args_group.fill_with_void()

        fn_call: INode | None = None

        if isinstance(content, TypeNode):
            if args_group_index.value is None and issubclass(content.type, IDefault):
                fn_call = content.type.create()
            else:
                t = content.type
                assert issubclass(t, IInheritableNode)
                fn_call = t.new(*filled_args_group.as_tuple)
        else:
            fn_call = FunctionCall(content, filled_args_group)

        assert isinstance(fn_call, INode)
        fn_call.as_node.validate()

        return DefineScratchOutput(scratch_index, Scratch(fn_call))

class DefineScratchFromScratchNode(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    idx_scratch_index = 1
    idx_source_index = 2
    idx_source_inner_index = 3

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        source_index = StateScratchIndex(arg2)
        source_inner_index = ScratchNodeIndex(arg3)
        return cls(scratch_index, source_index, source_inner_index)

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateScratchIndex.as_type(),
            StateScratchIndex.as_type(),
            ScratchNodeIndex.as_type(),
        ))

    def _run_action(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        source_index = self.inner_arg(self.idx_source_index).apply()
        source_inner_index = self.inner_arg(self.idx_source_inner_index).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(source_index, StateScratchIndex)
        assert isinstance(source_inner_index, ScratchNodeIndex)

        state = full_state.current_state.apply().cast(State)
        source_scratch = source_index.find_in_outer_node(state).value_or_raise
        assert isinstance(source_scratch, Scratch)
        source_scratch.validate()

        new_content = source_inner_index.find_in_node(source_scratch).value_or_raise

        return DefineScratchOutput(scratch_index, Scratch(new_content))

class RunScratch(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    idx_scratch_index = 1
    idx_source_index = 2

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        source_index = StateScratchIndex(arg2)
        Eq.from_ints(arg3, 0).raise_on_false()
        return cls(scratch_index, source_index)

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateScratchIndex.as_type(),
            StateScratchIndex.as_type(),
        ))

    def _run_action(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        source_index = self.inner_arg(self.idx_source_index).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(source_index, StateScratchIndex)

        state = full_state.current_state.apply().cast(State)
        scratch = source_index.find_in_outer_node(state).value_or_raise
        assert isinstance(scratch, Scratch)
        old_content = scratch.value_or_raise

        info = RunInfo.create()
        _, content = old_content.as_node.run(info).as_tuple

        _, again = content.as_node.run(info).as_tuple
        Eq(content, again).raise_on_false()

        return DefineScratchOutput(scratch_index, Scratch(content))

###########################################################
##################### UPDATE SCRATCH ######################
###########################################################

class UpdateScratchFromAnother(
    BasicAction[DefineScratchOutput],
    IInstantiable,
):

    idx_scratch_index = 1
    idx_scratch_inner_index = 2
    idx_source_index = 3

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        scratch_inner_index = ScratchNodeIndex(arg2)
        source_index = StateScratchIndex(arg3)
        return cls(scratch_index, scratch_inner_index, source_index)

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateScratchIndex.as_type(),
            ScratchNodeIndex.as_type(),
            StateScratchIndex.as_type(),
        ))

    def _run_action(self, full_state: FullState) -> DefineScratchOutput:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        scratch_inner_index = self.inner_arg(self.idx_scratch_inner_index).apply()
        source_index = self.inner_arg(self.idx_source_index).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(scratch_inner_index, ScratchNodeIndex)
        assert isinstance(source_index, StateScratchIndex)

        state = full_state.current_state.apply().cast(State)
        scratch = scratch_index.find_in_outer_node(state).value_or_raise
        source_scratch = source_index.find_in_outer_node(state).value_or_raise
        inner_content = source_scratch.value_or_raise

        new_content = scratch_inner_index.replace_in_target(
            scratch,
            inner_content,
        ).value_or_raise

        return DefineScratchOutput(scratch_index, Scratch(new_content))

###########################################################
#################### MANAGE ARGS GROUP ####################
###########################################################

class CreateArgsGroupOutput(GeneralAction, IInstantiable):

    idx_index = 1
    idx_new_args_group = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateArgsGroupIndex.as_type(),
            PartialArgsGroup.as_type(),
        ))

    def apply(self, full_state: FullState) -> State:
        index = self.inner_arg(self.idx_index).apply()
        new_args_group = self.inner_arg(self.idx_new_args_group).apply()
        assert isinstance(index, StateArgsGroupIndex)
        assert isinstance(new_args_group, PartialArgsGroup)

        state = full_state.current_state.apply().cast(State)
        args_outer_group = state.args_outer_group.apply().cast(PartialArgsOuterGroup)
        Eq.from_ints(index.as_int, len(args_outer_group.as_tuple) + 1).raise_on_false()
        for arg in new_args_group.as_tuple:
            arg.as_node.validate()

        args_outer_group = state.args_outer_group.apply().cast(PartialArgsOuterGroup)
        new_args = list(args_outer_group.as_tuple) + [new_args_group]

        return state.with_new_args(
            args_outer_group=args_outer_group.func(*new_args),
        )

class CreateArgsGroup(
    BasicAction[CreateArgsGroupOutput],
    IInstantiable,
):

    idx_args_amount = 1
    idx_args_group_source_index = 2

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        args_amount = Integer(arg1)
        args_group_source_index: Optional[StateArgsGroupIndex] = (
            Optional.create()
            if arg2 == 0
            else Optional(StateArgsGroupIndex(arg2)))
        Eq.from_ints(arg3, 0).raise_on_false()
        return cls(args_amount, args_group_source_index)

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            Integer.as_type(),
            Optional[StateArgsGroupIndex].as_type(),
        ))

    def _run_action(self, full_state: FullState) -> CreateArgsGroupOutput:
        args_amount = self.inner_arg(self.idx_args_amount).apply().cast(Integer)
        args_group_source_index = self.inner_arg(
            self.idx_args_group_source_index
        ).apply().cast(Optional[StateArgsGroupIndex])

        state = full_state.current_state.apply().cast(State)

        if args_group_source_index.value is not None:
            source = args_group_source_index.value.find_in_outer_node(state).value_or_raise
            assert isinstance(source, PartialArgsGroup)
            new_args_group = source

            if args_amount.as_int != new_args_group.amount():
                new_args_group = new_args_group.new_amount(args_amount.as_int)
        else:
            new_args_group = PartialArgsGroup.from_int(args_amount.as_int)

        index = StateArgsGroupIndex(
            len(state.args_outer_group.apply().cast(PartialArgsOuterGroup).as_tuple) + 1
        )

        return CreateArgsGroupOutput(index, new_args_group)

class DeleteArgsGroupOutput(GeneralAction, IBasicAction[FullState], IInstantiable):

    idx_args_group_index = 1

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        args_group_index = StateArgsGroupIndex(arg1)
        Eq.from_ints(arg2, 0).raise_on_false()
        Eq.from_ints(arg3, 0).raise_on_false()
        return cls(args_group_index)

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(StateArgsGroupIndex.as_type()))

    def apply(self, full_state: FullState) -> State:
        args_group_index = self.inner_arg(self.idx_args_group_index).apply()
        assert isinstance(args_group_index, StateArgsGroupIndex)
        state = full_state.current_state.apply().cast(State)
        new_state = args_group_index.remove_in_outer_target(state).value_or_raise
        return new_state

class DefineArgsGroupArgOutput(GeneralAction, IInstantiable):

    idx_group_index = 1
    idx_arg_index = 2
    idx_new_arg = 3

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateArgsGroupIndex.as_type(),
            NodeArgIndex.as_type(),
            IOptional.as_type(),
        ))

    def apply(self, full_state: FullState) -> State:
        group_index = self.inner_arg(self.idx_group_index).apply()
        arg_index = self.inner_arg(self.idx_arg_index).apply()
        new_arg = self.inner_arg(self.idx_new_arg).apply()
        assert isinstance(group_index, StateArgsGroupIndex)
        assert isinstance(arg_index, NodeArgIndex)
        assert isinstance(new_arg, IOptional)

        state = full_state.current_state.apply().cast(State)
        new_arg.as_node.validate()
        args_group = group_index.find_in_outer_node(state).value_or_raise
        assert isinstance(args_group, PartialArgsGroup)

        new_args_group = arg_index.replace_in_target(
            args_group,
            Optional.from_optional(new_arg)
        ).value_or_raise
        assert isinstance(new_args_group, PartialArgsGroup)

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
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        args_group_index = StateArgsGroupIndex(arg1)
        arg_index = NodeArgIndex(arg2)
        scratch_index = StateScratchIndex(arg3)
        return cls(args_group_index, arg_index, scratch_index)

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateArgsGroupIndex.as_type(),
            NodeArgIndex.as_type(),
            StateScratchIndex.as_type(),
        ))

    def _run_action(self, full_state: FullState) -> DefineArgsGroupArgOutput:
        args_group_index = self.inner_arg(self.idx_args_group_index).apply()
        arg_index = self.inner_arg(self.idx_arg_index).apply()
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        assert isinstance(args_group_index, StateArgsGroupIndex)
        assert isinstance(arg_index, NodeArgIndex)
        assert isinstance(scratch_index, StateScratchIndex)

        state = full_state.current_state.apply().cast(State)
        args_group_index.find_in_outer_node(state).raise_if_empty()
        scratch = scratch_index.find_in_outer_node(state).value_or_raise
        assert isinstance(scratch, Scratch)

        return DefineArgsGroupArgOutput(args_group_index, arg_index, scratch)
