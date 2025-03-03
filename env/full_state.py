# pylint: disable=too-many-lines
import typing
from abc import ABC
from env.core import (
    INode,
    IOpaqueScope,
    InheritableNode,
    IDefault,
    BaseGroup,
    IFromInt,
    IInt,
    Integer,
    INodeIndex,
    NodeMainBaseIndex,
    NodeArgBaseIndex,
    IFromSingleNode,
    IGroup,
    IFunction,
    IBoolean,
    TypeNode,
    IOptional,
    Optional,
    Protocol,
    CountableTypeGroup,
    IExceptionInfo,
    IWrapper,
    CompositeType,
    OptionalTypeGroup,
    TmpInnerArg,
    TmpNestedArg,
    IInstantiable,
)
from env.state import (
    State,
    Scratch,
    PartialArgsGroup,
    IGoalAchieved,
    IGoal,
    StateDefinition,
)
from env.meta_env import (
    MetaInfo,
    IFullState,
    IFullStateIndex,
    FullStateIntBaseIndex,
    IAction,
    IBasicAction,
    IRawAction,
    IActionOutput,
    SubtypeOuterGroup,
    DetailedType,
    GeneralTypeGroup,
    MetaInfoOptions,
)

T = typing.TypeVar('T', bound=INode)

###########################################################
####################### ACTION DATA #######################
###########################################################

class BaseActionData(InheritableNode, ABC):

    idx_raw_action = 1
    idx_action = 2
    idx_output = 3
    idx_exception = 4

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IRawAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IActionOutput.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IExceptionInfo.as_type()),
            ),
        ))

    @property
    def raw_action(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_raw_action)

    @property
    def action(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_action)

    @property
    def output(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_output)

    @property
    def exception(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_exception)

    @classmethod
    def from_args(
        cls,
        raw_action: Optional[IRawAction],
        action: Optional[IAction],
        output: Optional[IActionOutput],
        exception: Optional[IExceptionInfo],
    ) -> typing.Self:
        return cls(raw_action, action, output, exception)

    def _strict_validate(self):
        alias_info = self._thin_strict_validate()

        raw_action = self.raw_action.apply().cast(Optional[IRawAction]).value
        action = self.action.apply().cast(Optional[IAction]).value
        output = self.output.apply().cast(Optional[IActionOutput]).value
        exception = self.exception.apply().cast(Optional[IExceptionInfo]).value

        if raw_action is not None:
            raw_action_typed = raw_action.real(IRawAction).as_node
            if action is not None:
                raw_action_typed.strict_validate()
            else:
                raw_action_typed.validate()

        if action is not None:
            action_typed = action.real(IAction).as_node
            if output is not None:
                action_typed.strict_validate()
            else:
                action_typed.validate()
        if output is not None:
            output_typed = output.real(IActionOutput).as_node
            if exception is not None:
                output_typed.strict_validate()
            else:
                output_typed.validate()

        if exception is not None:
            exception.real(IExceptionInfo).as_node.validate()

        return alias_info

class SuccessActionData(BaseActionData, IInstantiable):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IRawAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IActionOutput.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(),
            ),
        ))

class RawActionErrorActionData(BaseActionData, IInstantiable):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IRawAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IExceptionInfo.as_type()),
            ),
        ))

class ActionTypeErrorActionData(BaseActionData, IInstantiable):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IRawAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IExceptionInfo.as_type()),
            ),
        ))

class ActionErrorActionData(BaseActionData, IInstantiable):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IRawAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IExceptionInfo.as_type()),
            ),
        ))

class ActionOutputErrorActionData(BaseActionData, IInstantiable):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IRawAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IActionOutput.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IExceptionInfo.as_type()),
            ),
        ))

###########################################################
####################### META ITEMS ########################
###########################################################

class RunProcessingCost(InheritableNode, IInstantiable):

    idx_actions = 1
    idx_steps = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            Integer.as_type(),
            Integer.as_type(),
        ))

    @property
    def actions(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_actions)

    @property
    def steps(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_steps)

    @classmethod
    def with_args(
        cls,
        actions: int,
        steps: int,
    ) -> typing.Self:
        return cls(
            Integer(actions),
            Integer(steps),
        )

class RunMemoryCost(InheritableNode, IInstantiable):

    idx_full_state_memory = 1
    idx_visible_state_memory = 2
    idx_main_state_memory = 3
    idx_run_memory = 4

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            Integer.as_type(),
            Integer.as_type(),
            Integer.as_type(),
            Integer.as_type(),
        ))

    @property
    def full_state_memory(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_full_state_memory)

    @property
    def visible_state_memory(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_visible_state_memory)

    @property
    def main_state_memory(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_main_state_memory)

    @property
    def run_memory(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_run_memory)

    @classmethod
    def with_args(
        cls,
        full_state_memory: int,
        visible_state_memory: int,
        main_state_memory: int,
        run_memory: int,
    ) -> typing.Self:
        return cls(
            Integer(full_state_memory),
            Integer(visible_state_memory),
            Integer(main_state_memory),
            Integer(run_memory),
        )

class RunCost(InheritableNode, IInstantiable):

    idx_processing_cost = 1
    idx_memory_cost = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            RunProcessingCost.as_type(),
            RunMemoryCost.as_type(),
        ))

    @property
    def processing_cost(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_processing_cost)

    @property
    def memory_cost(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_memory_cost)

    @classmethod
    def with_args(
        cls,
        processing_cost: RunProcessingCost,
        memory_cost: RunMemoryCost,
    ) -> typing.Self:
        return cls(
            processing_cost,
            memory_cost,
        )

class CostMultiplier(InheritableNode, IInstantiable):

    idx_current_multiplier = 1
    idx_default_multiplier = 2
    idx_applied_initial_multiplier = 3
    idx_steps = 4
    idx_step_count_to_change = 5

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            Integer.as_type(),
            Integer.as_type(),
            Integer.as_type(),
            Integer.as_type(),
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(Integer.as_type()),
            ),
        ))

    @property
    def current_multiplier(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_current_multiplier)

    @property
    def default_multiplier(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_default_multiplier)

    @property
    def applied_initial_multiplier(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_applied_initial_multiplier)

    @property
    def steps(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_steps)

    @property
    def step_count_to_change(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_step_count_to_change)

    @classmethod
    def with_args(
        cls,
        current_multiplier: int,
        default_multiplier: int,
        applied_initial_multiplier: int,
        steps: int,
        step_count_to_change: int | None = None,
    ) -> typing.Self:
        return cls(
            Integer(current_multiplier),
            Integer(default_multiplier),
            Integer(applied_initial_multiplier),
            Integer(steps),
            Optional.with_int(step_count_to_change),
        )

class NewCostMultiplier(InheritableNode, IInstantiable):

    idx_multiplier = 1
    idx_step_count_to_change = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            Integer.as_type(),
            Integer.as_type(),
        ))

    @property
    def multiplier(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_multiplier)

    @property
    def step_count_to_change(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_step_count_to_change)

    @classmethod
    def with_args(
        cls,
        multiplier: int,
        step_count_to_change: int | None = None,
    ) -> typing.Self:
        return cls(
            Integer(multiplier),
            Optional.with_int(step_count_to_change),
        )

class MetaData(InheritableNode, IDefault, IInstantiable):

    idx_remaining_steps = 1
    idx_run_cost = 2
    idx_cost_multiplier = 3
    idx_new_cost_multiplier = 4

    @classmethod
    def create(cls) -> typing.Self:
        return cls.with_args()

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(Integer.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(RunCost.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(CostMultiplier.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(NewCostMultiplier.as_type()),
            ),
        ))

    @property
    def remaining_steps(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_remaining_steps)

    @property
    def run_cost(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_run_cost)

    @property
    def cost_multiplier(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_cost_multiplier)

    @property
    def new_cost_multiplier(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_new_cost_multiplier)

    @classmethod
    def with_args(
        cls,
        remaining_steps: int | None = None,
        run_cost: RunCost | None = None,
        cost_multiplier: CostMultiplier | None = None,
        new_cost_multiplier: NewCostMultiplier | None = None,
    ) -> typing.Self:
        return cls.with_actual_args(
            remaining_steps=Optional.with_int(remaining_steps),
            run_cost=Optional.with_value(run_cost),
            cost_multiplier=Optional.with_value(cost_multiplier),
            new_cost_multiplier=Optional.with_value(new_cost_multiplier),
        )

    @classmethod
    def with_actual_args(
        cls,
        remaining_steps: Optional[Integer],
        run_cost: Optional[RunCost],
        cost_multiplier: Optional[CostMultiplier],
        new_cost_multiplier: Optional[NewCostMultiplier],
    ) -> typing.Self:
        return cls(
            remaining_steps,
            run_cost,
            cost_multiplier,
            new_cost_multiplier,
        )

    def with_new_args(
        self,
        remaining_steps: int | None = None,
        run_cost: Optional[RunCost] | None = None,
        cost_multiplier: Optional[CostMultiplier] | None = None,
        new_cost_multiplier: Optional[NewCostMultiplier] | None = None,
    ) -> typing.Self:
        return self.with_actual_args(
            remaining_steps=(
                Optional.with_int(remaining_steps)
                if remaining_steps is not None
                else self.remaining_steps.apply().real(Optional[Integer])
            ),
            run_cost=(
                run_cost
                if run_cost is not None
                else self.run_cost.apply().real(Optional[RunCost])
            ),
            cost_multiplier=(
                cost_multiplier
                if cost_multiplier is not None
                else self.cost_multiplier.apply().real(Optional[CostMultiplier])
            ),
            new_cost_multiplier=(
                new_cost_multiplier
                if new_cost_multiplier is not None
                else self.new_cost_multiplier.apply().real(Optional[NewCostMultiplier])
            ),
        )

###########################################################
################# FULL STATE DEFINITIONS ##################
###########################################################

class HistoryNode(InheritableNode, IDefault, IWrapper, IInstantiable):

    idx_state = 1
    idx_meta_data = 2
    idx_action_data = 3

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            State.as_type(),
            MetaData.as_type(),
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(BaseActionData.as_type()),
            ),
        ))

    @classmethod
    def create(cls):
        return cls(State.create(), Optional.create(), Optional.create())

    @classmethod
    def create_with_goal_and_options(
        cls,
        goal: IGoal,
        remaining_steps: int | None = None,
    ) -> typing.Self:
        goal_achieved = IGoalAchieved.from_goal_expr(goal)
        return cls(
            State.create_with_goal(goal_achieved),
            MetaData.with_args(remaining_steps=remaining_steps),
            Optional.create(),
        )

    @classmethod
    def with_args(
        cls,
        state: State,
        meta_data: MetaData,
        action_data: Optional[BaseActionData] | None = None,
    ):
        action_data = action_data if action_data is not None else Optional.create()
        return cls(state, meta_data, action_data)

    def with_new_args(
        self,
        state: State | None = None,
        meta_data: MetaData | None = None,
        action_data: Optional[BaseActionData] | None = None,
    ) -> typing.Self:
        state = (
            state
            if state is not None
            else self.state.apply().cast(State))
        meta_data = (
            meta_data
            if meta_data is not None
            else self.meta_data.apply().cast(MetaData))
        action_data = (
            action_data
            if action_data is not None
            else self.action_data.apply().cast(Optional[BaseActionData]))
        return self.with_args(
            state=state,
            meta_data=meta_data,
            action_data=action_data)

    @property
    def state(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_state)

    @property
    def meta_data(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_meta_data)

    @property
    def action_data(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_action_data)

class HistoryGroupNode(BaseGroup[HistoryNode], IInstantiable):

    @classmethod
    def item_type(cls) -> type[HistoryNode]:
        return HistoryNode

###########################################################
####################### FULL STATE ########################
###########################################################

class FullState(
    InheritableNode,
    IFullState,
    IOpaqueScope,
    IFromSingleNode[MetaInfo],
    IWrapper,
    IInstantiable,
):

    idx_meta = 1
    idx_current = 2
    idx_history = 3

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            MetaInfo.as_type(),
            HistoryNode.as_type(),
            HistoryGroupNode.as_type(),
        ))

    @classmethod
    def with_node(cls, node: MetaInfo) -> typing.Self:
        return cls.with_args(meta=node)

    @classmethod
    def with_args(
        cls,
        meta: MetaInfo,
        current: HistoryNode | None = None,
        history: HistoryGroupNode | None = None,
    ) -> typing.Self:
        goal = meta.goal.apply().cast(IGoal)
        options = meta.options.apply().cast(MetaInfoOptions)
        max_steps_opt = options.max_steps.apply().cast(Optional[IInt])
        max_steps = max_steps_opt.value.as_int if max_steps_opt.value is not None else None
        current = current if current is not None else HistoryNode.create_with_goal_and_options(
            goal=goal,
            remaining_steps=max_steps,
        )
        history = history if history is not None else HistoryGroupNode()
        return cls.new(meta, current, history)

    @property
    def meta(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_meta)

    @property
    def current(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_current)

    @property
    def history(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_history)

    @property
    def current_state(self) -> TmpNestedArg:
        return self.nested_arg(
            (self.idx_current, HistoryNode.idx_state)
        )

    @property
    def last_action_data(self) -> IOptional[BaseActionData]:
        history_group = self.history.apply().cast(HistoryGroupNode)
        history = history_group.as_tuple
        if len(history) == 0:
            return Optional.create()
        last_item = history_group.as_tuple[-1]
        assert isinstance(last_item, HistoryNode)
        action_data_opt = last_item.action_data.apply().cast(IOptional[BaseActionData])
        return action_data_opt

    def goal_achieved(self) -> bool:
        state = self.current_state.apply().cast(State)
        return state.goal_achieved()

    def node_types(self) -> tuple[type[INode], ...]:
        meta = self.meta.apply().cast(MetaInfo)
        all_types_wrappers = meta.all_types.apply().cast(GeneralTypeGroup).as_tuple
        all_types = tuple(wrapper.type for wrapper in all_types_wrappers)
        return all_types

    def history_amount(self) -> int:
        return len(self.history.apply().cast(HistoryGroupNode).as_tuple)

    def at_history(self, index: int) -> tuple[typing.Self, IOptional[BaseActionData]]:
        history = self.history.apply().cast(HistoryGroupNode).as_tuple
        assert index > 0
        assert index <= len(history)
        item = history[index-1]
        current = HistoryNode.with_args(
            state=item.state.apply().cast(State),
            meta_data=item.meta_data.apply().cast(MetaData),
        )
        action_data_opt = item.action_data.apply().cast(IOptional[BaseActionData])
        new_history_group = HistoryGroupNode.from_items(
            history[:index-1]
        )
        new_full_state = self.with_args(
            meta=self.meta.apply().cast(MetaInfo),
            current=current,
            history=new_history_group,
        )
        return new_full_state, action_data_opt


###########################################################
###################### MAIN INDICES #######################
###########################################################

class FullStateIndex(IFullStateIndex[FullState, T], typing.Generic[T], ABC):

    @classmethod
    def outer_type(cls):
        return FullState

class FullStateIntIndex(FullStateIntBaseIndex[FullState, T], typing.Generic[T], ABC):

    @classmethod
    def outer_type(cls):
        return FullState

class FullStateMainIndex(NodeMainBaseIndex, IFullStateIndex[FullState, INode], IInstantiable):

    @classmethod
    def outer_type(cls):
        return FullState

    def find_in_outer_node(self, node: FullState):
        return self.find_in_node(node)

    def replace_in_outer_target(self, target: FullState, new_node: INode):
        return self.replace_in_target(target, new_node)

class FullStateArgIndex(NodeArgBaseIndex, IFullStateIndex[FullState, INode], IInstantiable):

    @classmethod
    def outer_type(cls):
        return FullState

    @classmethod
    def item_type(cls):
        return INode

    def find_in_outer_node(self, node: FullState):
        return self.find_in_node(node)

    def replace_in_outer_target(self, target: FullState, new_node: INode):
        return self.replace_in_target(target, new_node)

class FullStateGroupBaseIndex(FullStateIntIndex[T], ABC):

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        raise NotImplementedError

    def find_in_outer_node(self, node: FullState):
        return self.find_arg(self.group(node).apply())

    def replace_in_outer_target(self, target: FullState, new_node: T):
        raise NotImplementedError

class FullStateReadonlyGroupBaseIndex(FullStateGroupBaseIndex[T], ABC):

    def replace_in_outer_target(self, target: FullState, new_node: T):
        return Optional.create()

class FullStateGroupTypeBaseIndex(FullStateReadonlyGroupBaseIndex[TypeNode[T]], ABC):

    @classmethod
    def item_type(cls) -> type[TypeNode[T]]:
        return TypeNode

    @classmethod
    def inner_item_type(cls) -> type[T]:
        raise NotImplementedError

###########################################################
###################### META INDICES #######################
###########################################################

class MetaAllTypesTypeIndex(
    FullStateGroupTypeBaseIndex[INode],
    IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return INode

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_all_types,
        ))

class MetaTypesDetailsTypeIndex(FullStateReadonlyGroupBaseIndex[DetailedType], IInstantiable):

    @classmethod
    def item_type(cls):
        return DetailedType

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_all_types_details,
        ))

class MetaAllowedBasicActionsTypeIndex(
    FullStateGroupTypeBaseIndex[IBasicAction[FullState]],
    IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IBasicAction

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_allowed_basic_actions,
        ))

class MetaAllowedActionsTypeIndex(
    FullStateGroupTypeBaseIndex[IAction[FullState]],
    IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IAction

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_allowed_actions,
        ))

class MetaDefaultTypeIndex(FullStateGroupTypeBaseIndex[IDefault], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IDefault

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_default_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaFromIntTypeIndex(FullStateGroupTypeBaseIndex[IFromInt], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IFromInt

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_from_int_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaIntTypeIndex(FullStateGroupTypeBaseIndex[IInt], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IInt

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_int_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaNodeIndexTypeIndex(FullStateGroupTypeBaseIndex[INodeIndex], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return INodeIndex

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_node_index_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaFullStateIndexTypeIndex(FullStateGroupTypeBaseIndex[FullStateIndex], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return FullStateIndex

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_full_state_index_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaFullStateIntIndexTypeIndex(FullStateGroupTypeBaseIndex[FullStateIntIndex], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return FullStateIntIndex

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_full_state_int_index_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaSingleChildTypeIndex(FullStateGroupTypeBaseIndex[IFromSingleNode], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IFromSingleNode

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_single_child_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaGroupTypeIndex(FullStateGroupTypeBaseIndex[IGroup], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IGroup

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_group_outer_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaFunctionTypeIndex(FullStateGroupTypeBaseIndex[IFunction], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IFunction

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_function_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaBooleanTypeIndex(FullStateGroupTypeBaseIndex[IBoolean], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IBoolean

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_boolean_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

O = typing.TypeVar('O', bound=IActionOutput)

class MetaAllActionsTypeIndex(
    FullStateGroupTypeBaseIndex[IAction[FullState]],
    IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IAction

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_all_actions,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaBasicActionsTypeIndex(
    FullStateGroupTypeBaseIndex[IBasicAction[FullState]],
    IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IBasicAction

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_basic_actions,
            SubtypeOuterGroup.idx_subtypes,
        ))

###########################################################
################## CURRENT STATE INDICES ##################
###########################################################

class CurrentStateScratchIndex(FullStateReadonlyGroupBaseIndex[Scratch], IInstantiable):

    @classmethod
    def item_type(cls):
        return Scratch

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_current,
            HistoryNode.idx_state,
            State.idx_scratch_group,
        ))

class CurrentStateArgsOuterGroupIndex(
    FullStateReadonlyGroupBaseIndex[PartialArgsGroup],
    IInstantiable,
):

    @classmethod
    def item_type(cls):
        return PartialArgsGroup

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_current,
            HistoryNode.idx_state,
            State.idx_args_outer_group,
        ))

class CurrentStateDefinitionIndex(FullStateReadonlyGroupBaseIndex[StateDefinition], IInstantiable):

    @classmethod
    def item_type(cls):
        return StateDefinition

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_current,
            HistoryNode.idx_state,
            State.idx_definition_group,
        ))
