#pylint: disable=too-many-lines
import typing
from abc import ABC
from env.core import (
    INode,
    IInheritableNode,
    InheritableNode,
    NodeArgIndex,
    NodeArgReverseIndex,
    DefaultGroup,
    Integer,
    BaseIntBoolean,
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
    CompositeType,
    OptionalTypeGroup,
    ScopeDataGroup,
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
    StateMetaHiddenInfo,
)
from env.meta_env import (
    MetaInfo,
    IActionGeneralInfo,
    IActionInfo,
    ActionInfo,
    ActionOutputInfo,
    ActionFullInfo,
    MetaInfoOptions,
    NewCostMultiplier,
)
from env.full_state import (
    FullState,
    FullStateIntIndex,
    HistoryGroupNode,
    HistoryNode,
    MetaDefaultTypeIndex,
    MetaFromIntTypeIndex,
    MetaSingleChildTypeIndex,
    MetaFullStateIntIndexTypeIndex,
)
from env.action import (
    IAction,
    BaseAction,
    IActionOutput,
    IBasicAction,
    BasicAction,
    GeneralAction,
)

###########################################################
###################### META ACTIONS #######################
###########################################################

class IMetaAction(IAction[FullState], ABC):
    pass

class IMetaActionOutput(IActionOutput[FullState], ABC):
    pass

class DynamicActionOutput(InheritableNode, IMetaActionOutput, IInstantiable):

    idx_action = 1
    idx_action_output = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            BaseAction.as_type(),
            IActionOutput.as_type(),
        ))

    def run_output(self, full_state: FullState) -> tuple[State, ActionFullInfo]:
        action = self.inner_arg(self.idx_action).apply().cast(BaseAction)
        output = self.inner_arg(self.idx_action_output).apply().real(IActionOutput[FullState])
        full_output, info = action.inner_run(full_state)
        Eq(full_output.output.apply(), output).raise_on_false()
        state = full_output.new_state.apply().real(State)
        output_info = ActionFullInfo(info.normalize(), ActionOutputInfo.create())
        return state, output_info

class DynamicAction(
    BasicAction[DynamicActionOutput],
    IMetaAction,
    IInstantiable,
):

    idx_scratch_index = 1

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        scratch_index = StateScratchIndex(arg1)
        Eq.from_ints(arg2, 0).raise_on_false()
        Eq.from_ints(arg3, 0).raise_on_false()
        return cls(scratch_index)

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateScratchIndex.as_type(),
        ))

    def _run_action(self, full_state: FullState) -> tuple[DynamicActionOutput, IActionInfo]:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        assert isinstance(scratch_index, StateScratchIndex)

        state = full_state.current_state.apply().real(State)
        scratch = scratch_index.find_in_outer_node(state).value_or_raise
        assert isinstance(scratch, Scratch)

        action = scratch.value_or_raise.real(BaseAction)
        full_output, info = action.inner_run(full_state)
        output = full_output.output.apply().real(IActionOutput[FullState])

        output = DynamicActionOutput(action, output)
        full_info = ActionFullInfo(info.normalize(), ActionInfo.create())

        return output, full_info

class GroupActionOutput(InheritableNode, IMetaActionOutput, IInstantiable):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.rest_protocol(CompositeType(
            DefaultGroup.as_type(),
            CountableTypeGroup(
                BaseAction.as_type(),
                IActionOutput.as_type(),
            ),
        ))

    def run_output(self, full_state: FullState) -> tuple[State, ActionFullInfo]:
        new_state = full_state.current_state.apply().real(State)
        item_infos: list[ActionFullInfo] = []
        for arg in self.args:
            group = arg.real(DefaultGroup)
            item1, item2 = group.as_tuple
            action = item1.real(BaseAction)
            output = item2.real(IActionOutput[FullState])
            full_output, info = action.inner_run(full_state)
            Eq(full_output.output.apply(), output).raise_on_false()
            new_state = full_output.new_state.apply().real(State)
            current = full_state.current.apply().real(HistoryNode)
            full_state = FullState.with_args(
                current=current.with_new_args(state=new_state),
                meta=full_state.meta.apply().real(MetaInfo),
                history=full_state.history.apply().real(HistoryGroupNode))
            item_infos.append(info.normalize())
        full_info = ActionFullInfo(*item_infos, ActionOutputInfo.create())
        return new_state, full_info

class GroupAction(BaseAction[GroupActionOutput], IMetaAction, IInstantiable):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.rest_protocol(BaseAction.as_type())

    def _run_action(self, full_state: FullState) -> tuple[GroupActionOutput, IActionInfo]:
        new_state = full_state.current_state.apply().real(State)
        items: list[DefaultGroup] = []
        item_infos: list[IActionGeneralInfo] = []
        for arg in self.args:
            action = arg.real(BaseAction)
            full_output, info = action.inner_run(full_state)
            output = full_output.output.apply().real(IActionOutput[FullState])
            new_state, out_info = output.run_output(full_state)
            current = full_state.current.apply().real(HistoryNode)
            full_state = FullState.with_args(
                current=current.with_new_args(state=new_state),
                meta=full_state.meta.apply().real(MetaInfo),
                history=full_state.history.apply().real(HistoryGroupNode))
            out_info = (
                out_info.normalize()
                if isinstance(out_info, ActionFullInfo)
                else out_info
            )
            items.append(DefaultGroup(action, output))
            item_infos.append(info.normalize())
            item_infos.append(out_info)
        output = GroupActionOutput(*items)
        full_info = ActionFullInfo(*item_infos, ActionInfo.create())
        return output, full_info

class RestoreHistoryStateOutput(GeneralAction, IBasicAction[FullState], IInstantiable):

    idx_recent_history_index = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            NodeArgReverseIndex.as_type(),
        ))

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        index = NodeArgReverseIndex(arg1)
        Eq.from_ints(arg2, 0).raise_on_false()
        Eq.from_ints(arg3, 0).raise_on_false()
        return cls(index)

    def run_output(self, full_state: FullState) -> tuple[State, ActionOutputInfo]:
        index = self.inner_arg(self.idx_recent_history_index).apply()
        assert isinstance(index, NodeArgReverseIndex)
        history = full_state.history.apply().real(HistoryGroupNode)
        history_item = index.find_in_node(history).value_or_raise.real(HistoryNode)
        new_state = history_item.state.apply().real(State)
        return new_state, ActionOutputInfo.create()

###########################################################
################### STATE META ACTIONS ####################
###########################################################

class VerifyGoalOutput(GeneralAction, IInstantiable):

    idx_nested_args_indices = 1
    idx_node = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(NestedArgIndexGroup.as_type()),
            ),
            INode.as_type(),
        ))

    def run_output(self, full_state: FullState) -> tuple[State, ActionOutputInfo]:
        nested_args_wrapper = self.inner_arg(
            self.idx_nested_args_indices
        ).apply().real(Optional[NestedArgIndexGroup])
        node = self.inner_arg(self.idx_node).apply()

        goal = full_state.nested_arg((FullState.idx_meta, MetaInfo.idx_goal)).apply().real(IGoal)
        nested_args_indices = nested_args_wrapper.value

        if nested_args_indices is not None:
            assert isinstance(nested_args_indices, NestedArgIndexGroup)
            goal = nested_args_indices.apply(goal.as_node).real(IGoal)

        assert isinstance(goal, Goal)

        state = full_state.current_state.apply().real(State)
        assert isinstance(node, goal.eval_param_type())

        meta_info = state.meta_info.apply().real(StateMetaInfo)
        goal_achieved = meta_info.goal_achieved.apply().real(IGoalAchieved)
        if nested_args_indices is not None:
            assert isinstance(nested_args_indices, NestedArgIndexGroup)
            goal_achieved = nested_args_indices.apply(goal_achieved.as_node).real(IGoalAchieved)
        Not(goal_achieved).raise_on_false()

        goal.evaluate(state, node).raise_on_false()

        new_meta_info = meta_info.apply_goal_achieved(nested_args_wrapper)
        new_state = state.with_new_args(meta_info=new_meta_info)

        sub_goal = (
            (nested_args_indices is not None)
            and
            len(nested_args_indices.as_tuple) > 0)

        meta = full_state.meta.apply().real(MetaInfo)
        meta_info_options = meta.options.apply().real(MetaInfoOptions)
        cost_multiplier_main_goal = meta_info_options.cost_multiplier_main_goal.apply().real(
            Integer)
        cost_multiplier_sub_goal = meta_info_options.cost_multiplier_sub_goal.apply().real(
            Integer)
        step_count_to_change_cost = meta_info_options.step_count_to_change_cost.apply().real(
            Integer)
        multiplier = cost_multiplier_sub_goal if sub_goal else cost_multiplier_main_goal
        new_cost_multiplier = NewCostMultiplier.with_args(
            multiplier=multiplier.as_int,
            step_count_to_change=step_count_to_change_cost.as_int)

        return new_state, ActionOutputInfo.with_args(new_cost_multiplier=new_cost_multiplier)

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
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(StateScratchIndex.as_type()),
            ),
            MetaFromIntTypeIndex.as_type(),
            Integer.as_type(),
        ))

    def _run_action(self, full_state: FullState) -> tuple[VerifyGoalOutput, IActionInfo]:
        scratch_index_nested_indices = self.inner_arg(
            self.idx_scratch_index_nested_indices
        ).apply().real(Optional[StateScratchIndex])
        type_index = self.inner_arg(self.idx_type_index).apply()
        index_value = self.inner_arg(self.idx_index_value).apply()
        assert isinstance(scratch_index_nested_indices, Optional)
        assert isinstance(type_index, MetaFromIntTypeIndex)
        assert isinstance(index_value, Integer)

        state = full_state.current_state.apply().real(State)
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

        return VerifyGoalOutput(nested, content), ActionInfo.create()

class CreateDynamicGoalOutput(GeneralAction, IInstantiable):

    idx_dynamic_goal_index = 1
    idx_goal_expr = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateDynamicGoalIndex.as_type(),
            IGoal.as_type(),
        ))

    def run_output(self, full_state: FullState) -> tuple[State, ActionOutputInfo]:
        index = self.inner_arg(self.idx_dynamic_goal_index).apply().real(StateDynamicGoalIndex)
        goal_expr = self.inner_arg(self.idx_goal_expr).apply().real(IGoal)

        state = full_state.current_state.apply().real(State)
        state_meta = state.meta_info.apply().real(StateMetaInfo)
        dynamic_goal_group = state_meta.dynamic_goal_group.apply().real(DynamicGoalGroup)

        items = [item for item in dynamic_goal_group.as_tuple]
        Eq.from_ints(index.as_int, len(items) + 1).raise_on_false()
        items.append(DynamicGoal.from_goal_expr(goal_expr))

        new_dynamic_goal_group = DynamicGoalGroup.from_items(items)
        new_meta_info = state_meta.with_new_args(
            dynamic_goal_group=new_dynamic_goal_group,
        )
        new_state = state.with_new_args(meta_info=new_meta_info)

        return new_state, ActionOutputInfo.create()

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

    def _run_action(self, full_state: FullState) -> tuple[CreateDynamicGoalOutput, IActionInfo]:
        scratch_index_goal = self.inner_arg(
            self.idx_scratch_index_goal
        ).apply().real(StateScratchIndex)

        state = full_state.current_state.apply().real(State)
        scratch = scratch_index_goal.find_in_outer_node(state).value_or_raise
        assert isinstance(scratch, Scratch)
        scratch.validate()

        goal_expr = scratch.value_or_raise
        assert isinstance(goal_expr, IGoal)

        dynamic_goal_group = state.meta_info.apply().real(
            StateMetaInfo
        ).dynamic_goal_group.apply().real(DynamicGoalGroup)
        dynamic_goal_index = StateDynamicGoalIndex(len(dynamic_goal_group.as_tuple) + 1)

        return CreateDynamicGoalOutput(dynamic_goal_index, goal_expr), ActionInfo.create()

class VerifyDynamicGoalOutput(GeneralAction, IInstantiable):

    idx_dynamic_goal = 1
    idx_nested_args_indices = 2
    idx_node = 3

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateDynamicGoalIndex.as_type(),
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(NestedArgIndexGroup.as_type()),
            ),
            INode.as_type(),
        ))

    def run_output(self, full_state: FullState) -> tuple[State, ActionOutputInfo]:
        dynamic_goal_index = self.inner_arg(
            self.idx_dynamic_goal
        ).apply().real(StateDynamicGoalIndex)
        nested_args_wrapper = self.inner_arg(
            self.idx_nested_args_indices
        ).apply().real(Optional[NestedArgIndexGroup])
        node = self.inner_arg(self.idx_node).apply()

        state = full_state.current_state.apply().real(State)
        dynamic_goal = dynamic_goal_index.find_in_outer_node(state).value_or_raise

        goal = dynamic_goal.goal_expr.apply().real(IGoal)
        nested_args_indices = nested_args_wrapper.value

        if nested_args_indices is not None:
            assert isinstance(nested_args_indices, NestedArgIndexGroup)
            goal = nested_args_indices.apply(goal.as_node).real(IGoal)

        assert isinstance(goal, Goal)
        assert isinstance(node, goal.eval_param_type())

        goal_achieved = dynamic_goal.goal_achieved.apply().real(IGoalAchieved)
        if nested_args_indices is not None:
            assert isinstance(nested_args_indices, NestedArgIndexGroup)
            goal_achieved = nested_args_indices.apply(goal_achieved.as_node).real(IGoalAchieved)
        Not(goal_achieved).raise_on_false()

        goal.evaluate(state, node).raise_on_false()

        dynamic_goal = dynamic_goal.apply_goal_achieved(nested_args_wrapper)
        new_state = dynamic_goal_index.replace_in_outer_target(
            state,
            dynamic_goal,
        ).value_or_raise

        meta = full_state.meta.apply().real(MetaInfo)
        meta_info_options = meta.options.apply().real(MetaInfoOptions)
        multiplier = meta_info_options.cost_multiplier_custom_goal.apply().real(
            Integer)
        step_count_to_change_cost = meta_info_options.step_count_to_change_cost.apply().real(
            Integer)
        new_cost_multiplier = NewCostMultiplier.with_args(
            multiplier=multiplier.as_int,
            step_count_to_change=step_count_to_change_cost.as_int)

        return new_state, ActionOutputInfo.with_args(new_cost_multiplier=new_cost_multiplier)

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
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(StateScratchIndex.as_type()),
            ),
            StateScratchIndex.as_type(),
        ))

    def _run_action(self, full_state: FullState) -> tuple[VerifyDynamicGoalOutput, IActionInfo]:
        dynamic_node_index = self.inner_arg(
            self.idx_dynamic_node_index
        ).apply().real(StateDynamicGoalIndex)
        scratch_index_nested_indices = self.inner_arg(
            self.idx_scratch_index_nested_indices
        ).apply().real(Optional[StateScratchIndex])
        scratch_content_index = self.inner_arg(
            self.idx_scratch_index_content
        ).apply().real(StateScratchIndex)

        state = full_state.current_state.apply().real(State)
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

        return VerifyDynamicGoalOutput(dynamic_node_index, nested, content), ActionInfo.create()

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

    def run_output(self, full_state: FullState) -> tuple[State, ActionOutputInfo]:
        dynamic_goal_index = self.inner_arg(self.idx_dynamic_goal_index).apply()
        assert isinstance(dynamic_goal_index, StateDynamicGoalIndex)
        state = full_state.current_state.apply().real(State)
        new_state = dynamic_goal_index.remove_in_outer_target(state).value_or_raise
        return new_state, ActionOutputInfo.create()

class DefineStateHiddenInfoOutput(GeneralAction, IInstantiable):

    idx_hidden_info = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateMetaHiddenInfo.as_type(),
        ))

    def run_output(self, full_state: FullState) -> tuple[State, ActionOutputInfo]:
        hidden_info = self.inner_arg(
            self.idx_hidden_info
        ).apply().real(StateMetaHiddenInfo)

        state = full_state.current_state.apply().real(State)
        meta_info = state.meta_info.apply().real(StateMetaInfo)

        new_meta_info = meta_info.with_new_args(hidden_info=hidden_info)
        new_state = state.with_new_args(meta_info=new_meta_info)

        return new_state, ActionOutputInfo.create()

class ResetStateHiddenInfo(BasicAction[DefineStateHiddenInfoOutput], IInstantiable):

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        Eq.from_ints(arg1, 0).raise_on_false()
        Eq.from_ints(arg2, 0).raise_on_false()
        Eq.from_ints(arg3, 0).raise_on_false()
        return cls()

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup())

    def _run_action(self, full_state: FullState) -> tuple[
        DefineStateHiddenInfoOutput,
        IActionInfo,
    ]:
        hidden_info = StateMetaHiddenInfo.create()
        return DefineStateHiddenInfoOutput(hidden_info), ActionInfo.create()

class DefineStateHiddenInfo(BasicAction[DefineStateHiddenInfoOutput], IInstantiable):

    idx_hidden_index = 1
    idx_type_index = 2
    idx_index_value = 3

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        hidden_index = NodeArgIndex(arg1)
        type_index = MetaFromIntTypeIndex(arg2)
        index_value = Integer(arg3)
        return cls(hidden_index, type_index, index_value)

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            NodeArgIndex.as_type(),
            MetaFromIntTypeIndex.as_type(),
            Integer.as_type(),
        ))

    def _run_action(self, full_state: FullState) -> tuple[
        DefineStateHiddenInfoOutput,
        IActionInfo,
    ]:
        hidden_index = self.inner_arg(self.idx_hidden_index).apply()
        type_index = self.inner_arg(self.idx_type_index).apply()
        index_value = self.inner_arg(self.idx_index_value).apply()
        assert isinstance(hidden_index, NodeArgIndex)
        assert isinstance(type_index, MetaFromIntTypeIndex)
        assert isinstance(index_value, Integer)

        node_type = type_index.find_in_outer_node(full_state).value_or_raise
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, IFromInt)
        hidden_value: INode = node_type.type.from_int(index_value.as_int)
        if not issubclass(node_type.type, BaseIntBoolean):
            hidden_value = Optional(hidden_value)

        state = full_state.current_state.apply().real(State)
        meta_info = state.meta_info.apply().real(StateMetaInfo)
        hidden_info = meta_info.hidden_info.apply().real(StateMetaHiddenInfo)
        hidden_info = hidden_index.replace_in_target(hidden_info, hidden_value).value_or_raise

        return DefineStateHiddenInfoOutput(hidden_info), ActionInfo.create()

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

    def run_output(self, full_state: FullState) -> tuple[State, ActionOutputInfo]:
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

    def run_output(self, full_state: FullState) -> tuple[State, ActionOutputInfo]:
        raise NotImplementedError

###########################################################
##################### MANAGE SCRATCH ######################
###########################################################

class CreateScratchOutput(ScratchWithNodeBaseActionOutput, IInstantiable):

    def run_output(self, full_state: FullState) -> tuple[State, ActionOutputInfo]:
        index = self.inner_arg(self.idx_index).apply()
        new_node = self.inner_arg(self.idx_node).apply()
        assert isinstance(index, StateScratchIndex)
        assert isinstance(new_node, Scratch)

        state = full_state.current_state.apply().real(State)
        scratch_group = state.scratch_group.apply().real(ScratchGroup)
        Eq.from_ints(index.as_int, len(scratch_group.as_tuple) + 1).raise_on_false()
        new_args = list(scratch_group.as_tuple) + [new_node]

        new_state = state.with_new_args(
            scratch_group=scratch_group.func(*new_args),
        )
        return new_state, ActionOutputInfo.create()

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
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(StateScratchIndex.as_type()),
            ),
        ))

    @property
    def child(self):
        return self.inner_arg(self.idx_clone_index).apply()

    def _run_action(self, full_state: FullState) -> tuple[CreateScratchOutput, IActionInfo]:
        clone_index = self.inner_arg(
            self.idx_clone_index
        ).apply().real(Optional[StateScratchIndex])

        state = full_state.current_state.apply().real(State)
        scratch_group = state.scratch_group.apply().real(ScratchGroup)
        index = StateScratchIndex(len(scratch_group.as_tuple) + 1)
        if clone_index.value is None:
            return CreateScratchOutput(index, Scratch.create()), ActionInfo.create()
        scratch = clone_index.value.find_in_outer_node(state).value_or_raise
        assert isinstance(scratch, Scratch)
        scratch.validate()

        return CreateScratchOutput(index, scratch), ActionInfo.create()

class DeleteScratchOutput(ScratchBaseActionOutput, IBasicAction[FullState], IInstantiable):

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        index = StateScratchIndex(arg1)
        Eq.from_ints(arg2, 0).raise_on_false()
        Eq.from_ints(arg3, 0).raise_on_false()
        return cls(index)

    def run_output(self, full_state: FullState) -> tuple[State, ActionOutputInfo]:
        index = self.inner_arg(self.idx_index).apply()
        assert isinstance(index, StateScratchIndex)
        state = full_state.current_state.apply().real(State)
        new_state = index.remove_in_outer_target(state).value_or_raise
        return new_state, ActionOutputInfo.create()

###########################################################
##################### DEFINE SCRATCH ######################
###########################################################

class DefineScratchOutput(ScratchWithNodeBaseActionOutput, IInstantiable):

    def run_output(self, full_state: FullState) -> tuple[State, ActionOutputInfo]:
        index = self.inner_arg(self.idx_index).apply()
        scratch = self.inner_arg(self.idx_node).apply()
        assert isinstance(index, StateScratchIndex)
        assert isinstance(scratch, Scratch)

        state = full_state.current_state.apply().real(State)
        new_state = index.replace_in_outer_target(state, scratch).value_or_raise

        return new_state, ActionOutputInfo.create()

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

    def _run_action(self, full_state: FullState) -> tuple[DefineScratchOutput, IActionInfo]:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        output = DefineScratchOutput(scratch_index, Scratch.create())
        return output, ActionInfo.create()

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

    def _run_action(self, full_state: FullState) -> tuple[DefineScratchOutput, IActionInfo]:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        type_index = self.inner_arg(self.idx_type_index).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(type_index, MetaDefaultTypeIndex)

        node_type = type_index.find_in_outer_node(full_state).value_or_raise
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, IDefault)
        content = node_type.type.create()

        output = DefineScratchOutput(scratch_index, Scratch(content))
        return output, ActionInfo.create()

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

    def _run_action(self, full_state: FullState) -> tuple[DefineScratchOutput, IActionInfo]:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        type_index = self.inner_arg(self.idx_type_index).apply()
        index_value = self.inner_arg(self.idx_index_value).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(type_index, MetaFromIntTypeIndex)
        assert isinstance(index_value, Integer)

        node_type = type_index.find_in_outer_node(full_state).value_or_raise
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, IFromInt)
        content = node_type.type.from_int(index_value.as_int)

        output = DefineScratchOutput(scratch_index, Scratch(content))
        return output, ActionInfo.create()

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

    def _run_action(self, full_state: FullState) -> tuple[DefineScratchOutput, IActionInfo]:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        type_index = self.inner_arg(self.idx_type_index).apply()
        arg_index = self.inner_arg(self.idx_arg).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(type_index, MetaSingleChildTypeIndex)
        assert isinstance(arg_index, StateScratchIndex)

        node_type = type_index.find_in_outer_node(full_state).value_or_raise
        assert isinstance(node_type, TypeNode) and issubclass(node_type.type, IFromSingleNode)

        state = full_state.current_state.apply().real(State)
        scratch = arg_index.find_in_outer_node(state).value_or_raise
        arg = scratch.value_or_raise

        content = node_type.type.with_node(arg)

        output = DefineScratchOutput(scratch_index, Scratch(content))
        return output, ActionInfo.create()

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

    def _run_action(self, full_state: FullState) -> tuple[DefineScratchOutput, IActionInfo]:
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

        output = DefineScratchOutput(scratch_index, Scratch(content))
        return output, ActionInfo.create()

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

    def _run_action(self, full_state: FullState) -> tuple[DefineScratchOutput, IActionInfo]:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        source_index = self.inner_arg(self.idx_source_index).apply()
        int_arg = self.inner_arg(self.idx_int_arg).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(source_index, StateScratchIndex)
        assert isinstance(int_arg, Integer)

        state = full_state.current_state.apply().real(State)
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

        output = DefineScratchOutput(scratch_index, Scratch(fn_call))
        return output, ActionInfo.create()

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

    def _run_action(self, full_state: FullState) -> tuple[DefineScratchOutput, IActionInfo]:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        source_index = self.inner_arg(self.idx_source_index).apply()
        single_arg_index = self.inner_arg(self.idx_single_arg_index).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(source_index, StateScratchIndex)
        assert isinstance(single_arg_index, StateScratchIndex)

        state = full_state.current_state.apply().real(State)
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

        output = DefineScratchOutput(scratch_index, Scratch(fn_call))
        return output, ActionInfo.create()

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
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(StateArgsGroupIndex.as_type()),
            ),
        ))

    def _run_action(self, full_state: FullState) -> tuple[DefineScratchOutput, IActionInfo]:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        source_index = self.inner_arg(self.idx_source_index).apply()
        args_group_index = self.inner_arg(self.idx_args_group_index).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(source_index, StateScratchIndex)
        assert isinstance(args_group_index, Optional)

        state = full_state.current_state.apply().real(State)
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

        output = DefineScratchOutput(scratch_index, Scratch(fn_call))
        return output, ActionInfo.create()

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

    def _run_action(self, full_state: FullState) -> tuple[DefineScratchOutput, IActionInfo]:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        source_index = self.inner_arg(self.idx_source_index).apply()
        source_inner_index = self.inner_arg(self.idx_source_inner_index).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(source_index, StateScratchIndex)
        assert isinstance(source_inner_index, ScratchNodeIndex)

        state = full_state.current_state.apply().real(State)
        source_scratch = source_index.find_in_outer_node(state).value_or_raise
        assert isinstance(source_scratch, Scratch)
        source_scratch.validate()

        new_content = source_inner_index.find_in_node(source_scratch).value_or_raise

        output = DefineScratchOutput(scratch_index, Scratch(new_content))
        return output, ActionInfo.create()

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

    def _run_action(self, full_state: FullState) -> tuple[DefineScratchOutput, IActionInfo]:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        scratch_inner_index = self.inner_arg(self.idx_scratch_inner_index).apply()
        source_index = self.inner_arg(self.idx_source_index).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(scratch_inner_index, ScratchNodeIndex)
        assert isinstance(source_index, StateScratchIndex)

        state = full_state.current_state.apply().real(State)
        scratch = scratch_index.find_in_outer_node(state).value_or_raise
        source_scratch = source_index.find_in_outer_node(state).value_or_raise
        inner_content = source_scratch.value_or_raise

        new_content = scratch_inner_index.replace_in_target(
            scratch,
            inner_content,
        ).value_or_raise

        output = DefineScratchOutput(scratch_index, Scratch(new_content))
        return output, ActionInfo.create()

###########################################################
####################### RUN SCRATCH #######################
###########################################################

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

    def _run_action(self, full_state: FullState) -> tuple[DefineScratchOutput, IActionInfo]:
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        source_index = self.inner_arg(self.idx_source_index).apply()
        assert isinstance(scratch_index, StateScratchIndex)
        assert isinstance(source_index, StateScratchIndex)

        state = full_state.current_state.apply().real(State)
        scratch = source_index.find_in_outer_node(state).value_or_raise
        assert isinstance(scratch, Scratch)
        old_content = scratch.value_or_raise

        info = RunInfo.with_args(
            scope_data_group=ScopeDataGroup(),
            return_after_scope=Optional(),
        )
        _, content = old_content.as_node.run(info).as_tuple

        _, again = content.as_node.run(info).as_tuple
        Eq(content, again).raise_on_false()

        output = DefineScratchOutput(scratch_index, Scratch(content))
        return output, ActionInfo.create()

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

    def run_output(self, full_state: FullState) -> tuple[State, ActionOutputInfo]:
        index = self.inner_arg(self.idx_index).apply()
        new_args_group = self.inner_arg(self.idx_new_args_group).apply()
        assert isinstance(index, StateArgsGroupIndex)
        assert isinstance(new_args_group, PartialArgsGroup)

        state = full_state.current_state.apply().real(State)
        args_outer_group = state.args_outer_group.apply().real(PartialArgsOuterGroup)
        Eq.from_ints(index.as_int, len(args_outer_group.as_tuple) + 1).raise_on_false()
        for arg in new_args_group.as_tuple:
            arg.as_node.validate()

        args_outer_group = state.args_outer_group.apply().real(PartialArgsOuterGroup)
        new_args = list(args_outer_group.as_tuple) + [new_args_group]

        new_state = state.with_new_args(
            args_outer_group=args_outer_group.func(*new_args),
        )
        return new_state, ActionOutputInfo.create()

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
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(StateArgsGroupIndex.as_type()),
            ),
        ))

    def _run_action(self, full_state: FullState) -> tuple[CreateArgsGroupOutput, IActionInfo]:
        args_amount = self.inner_arg(self.idx_args_amount).apply().real(Integer)
        args_group_source_index = self.inner_arg(
            self.idx_args_group_source_index
        ).apply().real(Optional[StateArgsGroupIndex])

        state = full_state.current_state.apply().real(State)

        if args_group_source_index.value is not None:
            source = args_group_source_index.value.find_in_outer_node(state).value_or_raise
            assert isinstance(source, PartialArgsGroup)
            new_args_group = source

            if args_amount.as_int != new_args_group.amount():
                new_args_group = new_args_group.new_amount(args_amount.as_int)
        else:
            new_args_group = PartialArgsGroup.from_int(args_amount.as_int)

        index = StateArgsGroupIndex(
            len(state.args_outer_group.apply().real(PartialArgsOuterGroup).as_tuple) + 1
        )

        output = CreateArgsGroupOutput(index, new_args_group)
        return output, ActionInfo.create()

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

    def run_output(self, full_state: FullState) -> tuple[State, ActionOutputInfo]:
        args_group_index = self.inner_arg(self.idx_args_group_index).apply()
        assert isinstance(args_group_index, StateArgsGroupIndex)
        state = full_state.current_state.apply().real(State)
        new_state = args_group_index.remove_in_outer_target(state).value_or_raise
        return new_state, ActionOutputInfo.create()

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

    def run_output(self, full_state: FullState) -> tuple[State, ActionOutputInfo]:
        group_index = self.inner_arg(self.idx_group_index).apply()
        arg_index = self.inner_arg(self.idx_arg_index).apply()
        new_arg = self.inner_arg(self.idx_new_arg).apply()
        assert isinstance(group_index, StateArgsGroupIndex)
        assert isinstance(arg_index, NodeArgIndex)
        assert isinstance(new_arg, IOptional)

        state = full_state.current_state.apply().real(State)
        new_arg.as_node.validate()
        args_group = group_index.find_in_outer_node(state).value_or_raise
        assert isinstance(args_group, PartialArgsGroup)

        new_args_group = arg_index.replace_in_target(
            args_group,
            Optional.from_optional(new_arg)
        ).value_or_raise
        assert isinstance(new_args_group, PartialArgsGroup)

        new_state = group_index.replace_in_outer_target(state, new_args_group).value_or_raise

        return new_state, ActionOutputInfo.create()

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

    def _run_action(self, full_state: FullState) -> tuple[DefineArgsGroupArgOutput, IActionInfo]:
        args_group_index = self.inner_arg(self.idx_args_group_index).apply()
        arg_index = self.inner_arg(self.idx_arg_index).apply()
        scratch_index = self.inner_arg(self.idx_scratch_index).apply()
        assert isinstance(args_group_index, StateArgsGroupIndex)
        assert isinstance(arg_index, NodeArgIndex)
        assert isinstance(scratch_index, StateScratchIndex)

        state = full_state.current_state.apply().real(State)
        args_group_index.find_in_outer_node(state).raise_if_empty()
        scratch = scratch_index.find_in_outer_node(state).value_or_raise
        assert isinstance(scratch, Scratch)

        output = DefineArgsGroupArgOutput(args_group_index, arg_index, scratch)
        return output, ActionInfo.create()
