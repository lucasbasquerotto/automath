import typing
from abc import ABC
from env.core import (
    INode,
    IDefault,
    BaseNode,
    TypeNode,
    IFunction,
    IOptional,
    CompositeType,
    IntBoolean,
    OptionalBase,
    OptionalTypeGroup,
    Integer,
    Optional,
    AliasInfo,
    Type,
    InheritableNode,
    BaseGroup,
    NodeIntBaseIndex,
    NodeMainIndex,
    NodeArgIndex,
    IOpaqueScope,
    BaseInt,
    Protocol,
    BaseOptionalValueGroup,
    ITypedIndex,
    ITypedIntIndex,
    CountableTypeGroup,
    FunctionExpr,
    NestedArgIndexGroup,
    Eq,
    And,
    IBoolean,
    BaseIntBoolean,
    IWrapper,
    TmpInnerArg,
    IInstantiable,
)

T = typing.TypeVar('T', bound=INode)
K = typing.TypeVar('K', bound=INode)

###########################################################
#################### STATE DEFINITIONS ####################
###########################################################

class IGoal(INode, ABC):
    pass

class Goal(InheritableNode, IGoal, typing.Generic[T, K], ABC):

    idx_goal_inner_expr = 1
    idx_eval_param_type = 2

    @classmethod
    def goal_type(cls) -> type[T]:
        raise NotImplementedError

    @classmethod
    def eval_param_type(cls) -> type[K]:
        raise NotImplementedError

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            cls.goal_type().as_type(),
            Type(cls.eval_param_type().as_type()),
        ))

    @property
    def goal_inner_expr(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_goal_inner_expr)

    @property
    def eval_param_type_raw(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_eval_param_type)

    def evaluate(self, state: 'State', eval_param: K) -> IBoolean:
        raise NotImplementedError

    @classmethod
    def with_goal(cls, goal_expr: T) -> typing.Self:
        return cls(goal_expr, cls.eval_param_type().as_type())

class GoalGroup(BaseGroup[IGoal], IGoal, IInstantiable):

    @classmethod
    def item_type(cls) -> TypeNode:
        return IGoal.as_type()

class IGoalAchieved(IBoolean, ABC):

    @classmethod
    def from_goal_expr(cls, goal: IGoal) -> 'IGoalAchieved':
        if isinstance(goal, Goal):
            return GoalAchieved.create()
        if isinstance(goal, GoalGroup):
            return GoalAchievedGroup(*[
                cls.from_goal_expr(sub_goal)
                for sub_goal in goal.as_tuple
            ])
        raise NotImplementedError(type(goal))

    def define_achieved(
        self,
        nested_args_wrapper: Optional[NestedArgIndexGroup],
    ) -> typing.Self:
        nested_args_indices = nested_args_wrapper.value
        groups: list[tuple[int, GoalAchievedGroup]] = []
        goal_achieved: IGoalAchieved = self

        if nested_args_indices is not None:
            idxs = [idx.as_int for idx in nested_args_indices.as_tuple]
            for idx in idxs:
                assert isinstance(goal_achieved, GoalAchievedGroup), f'{idx} in {idxs}'
                groups.append((idx, goal_achieved))
                goal_achieved = goal_achieved.as_node.inner_arg(idx).apply().cast(IGoalAchieved)

        assert isinstance(goal_achieved, GoalAchieved)
        assert goal_achieved.as_bool is False
        goal_achieved = GoalAchieved.achieved()

        for idx, group in groups[::-1]:
            replaced = NodeArgIndex(idx).replace_in_target(group, goal_achieved)
            goal_achieved = replaced.value_or_raise

        assert isinstance(goal_achieved, type(self))
        return typing.cast(typing.Self, goal_achieved)

class GoalAchieved(BaseIntBoolean, IGoalAchieved, IInstantiable):

    @classmethod
    def achieved(cls) -> typing.Self:
        return cls(1)

class GoalAchievedGroup(BaseGroup[IGoalAchieved], IGoalAchieved, IInstantiable):

    @classmethod
    def item_type(cls) -> TypeNode:
        return IGoalAchieved.as_type()

    @property
    def as_bool(self) -> bool:
        return And(*self.args).as_bool

class DynamicGoal(InheritableNode, IDefault, IInstantiable):

    idx_goal_expr = 1
    idx_goal_achieved = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            IGoal.as_type(),
            IGoalAchieved.as_type(),
        ))

    @classmethod
    def from_goal_expr(cls, goal: IGoal) -> typing.Self:
        return cls(goal, IGoalAchieved.from_goal_expr(goal))

    @property
    def goal_expr(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_goal_expr)

    @property
    def goal_achieved(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_goal_achieved)

    def apply_goal_achieved(
        self,
        nested_args_wrapper: Optional[NestedArgIndexGroup],
    ) -> typing.Self:
        goal_achieved = self.goal_achieved.apply().cast(IGoalAchieved)
        goal_achieved = goal_achieved.define_achieved(nested_args_wrapper)
        result = NodeArgIndex(self.idx_goal_achieved).replace_in_target(self, goal_achieved)
        return result.value_or_raise

class DynamicGoalGroup(BaseGroup[DynamicGoal], IInstantiable):

    @classmethod
    def item_type(cls) -> TypeNode:
        return DynamicGoal.as_type()

class StateMetaHiddenInfo(InheritableNode, IDefault, IInstantiable):

    idx_meta_hidden = 1
    idx_history_amount_to_show = 2
    idx_history_state_hidden = 3
    idx_history_meta_hidden = 4
    idx_history_raw_action_hidden = 5
    idx_history_action_hidden = 6
    idx_history_action_output_hidden = 7
    idx_history_action_exception_hidden = 8

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            IntBoolean.as_type(),
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(Integer.as_type())
            ),
            IntBoolean.as_type(),
            IntBoolean.as_type(),
            IntBoolean.as_type(),
            IntBoolean.as_type(),
            IntBoolean.as_type(),
            IntBoolean.as_type(),
        ))

    @property
    def meta_hidden(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_meta_hidden)

    @property
    def history_amount_to_show(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_history_amount_to_show)

    @property
    def history_state_hidden(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_history_state_hidden)

    @property
    def history_meta_hidden(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_history_meta_hidden)

    @property
    def history_raw_action_hidden(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_history_raw_action_hidden)

    @property
    def history_action_hidden(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_history_action_hidden)

    @property
    def history_action_output_hidden(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_history_action_output_hidden)

    @property
    def history_action_exception_hidden(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_history_action_exception_hidden)

    @classmethod
    def create(cls) -> typing.Self:
        return cls.with_args()

    @classmethod
    def with_args(
        cls,
        meta_hidden: IntBoolean | None = None,
        history_amount_to_show: Optional[Integer] | None = None,
        history_state_hidden: IntBoolean | None = None,
        history_meta_hidden: IntBoolean | None = None,
        history_raw_action_hidden: IntBoolean | None = None,
        history_action_hidden: IntBoolean | None = None,
        history_action_output_hidden: IntBoolean | None = None,
        history_action_exception_hidden: IntBoolean | None = None,
    ) -> typing.Self:
        return cls(
            meta_hidden or IBoolean.false(),
            history_amount_to_show or Optional.create(),
            history_state_hidden or IBoolean.false(),
            history_meta_hidden or IBoolean.false(),
            history_raw_action_hidden or IBoolean.false(),
            history_action_hidden or IBoolean.false(),
            history_action_output_hidden or IBoolean.false(),
            history_action_exception_hidden or IBoolean.false(),
        )


class StateMetaInfo(InheritableNode, IDefault, IInstantiable):

    idx_goal_achieved = 1
    idx_dynamic_goal_group = 2
    idx_hidden_info = 3

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            IGoalAchieved.as_type(),
            DynamicGoalGroup.as_type(),
            StateMetaHiddenInfo.as_type(),
        ))

    @property
    def goal_achieved(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_goal_achieved)

    @property
    def dynamic_goal_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_dynamic_goal_group)

    @property
    def hidden_info(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_hidden_info)

    @classmethod
    def create(cls) -> typing.Self:
        return cls.with_args()

    @classmethod
    def with_goal_achieved(cls, goal_achieved: IGoalAchieved) -> typing.Self:
        return cls.with_args(goal_achieved=goal_achieved)

    @classmethod
    def with_goal_expr(cls, goal: IGoal) -> typing.Self:
        return cls.with_goal_achieved(IGoalAchieved.from_goal_expr(goal))

    @classmethod
    def with_args(
        cls,
        goal_achieved: IGoalAchieved | None = None,
        dynamic_goal_group: DynamicGoalGroup | None = None,
        hidden_info: StateMetaHiddenInfo | None = None,
    ) -> typing.Self:
        return cls(
            goal_achieved or GoalAchieved.create(),
            dynamic_goal_group or DynamicGoalGroup(),
            hidden_info or StateMetaHiddenInfo.create(),
        )

    def with_new_args(
        self,
        goal_achieved: IGoalAchieved | None = None,
        dynamic_goal_group: DynamicGoalGroup | None = None,
        hidden_info: StateMetaHiddenInfo | None = None,
    ) -> typing.Self:
        goal_achieved = goal_achieved or self.goal_achieved.apply().cast(IGoalAchieved)
        dynamic_goal_group = (
            dynamic_goal_group
            or self.dynamic_goal_group.apply().cast(DynamicGoalGroup))
        hidden_info = hidden_info or self.hidden_info.apply().cast(StateMetaHiddenInfo)
        return self.with_args(
            goal_achieved=goal_achieved,
            dynamic_goal_group=dynamic_goal_group,
            hidden_info=hidden_info,
        )

    def apply_goal_achieved(
        self,
        nested_args_wrapper: Optional[NestedArgIndexGroup],
    ) -> typing.Self:
        goal_achieved = self.goal_achieved.apply().cast(IGoalAchieved)
        goal_achieved = goal_achieved.define_achieved(nested_args_wrapper)
        result = NodeArgIndex(self.idx_goal_achieved).replace_in_target(self, goal_achieved)
        return result.value_or_raise

    def is_goal_achieved(self) -> bool:
        return self.goal_achieved.apply().cast(IGoalAchieved).as_bool

class IContext(IWrapper, ABC):
    pass

class Scratch(OptionalBase[INode], IContext, IOpaqueScope, IInstantiable):

    def _strict_validate(self):
        self.validate()
        return AliasInfo.create()

class ScratchGroup(BaseGroup[Scratch], IInstantiable):

    @classmethod
    def item_type(cls) -> TypeNode:
        return Scratch.as_type()

    @classmethod
    def from_raw_items(cls, items: typing.Sequence[INode | None]) -> typing.Self:
        return cls.from_items([Scratch.with_value(s) for s in items])

    def to_raw_items(self) -> tuple[INode | None, ...]:
        return tuple(s.value for s in self.as_tuple)

class PartialArgsGroup(BaseOptionalValueGroup[INode], IOpaqueScope, IInstantiable):

    def _strict_validate(self):
        return self._thin_strict_validate()

class PartialArgsOuterGroup(BaseGroup[PartialArgsGroup], IInstantiable):

    @classmethod
    def item_type(cls) -> TypeNode:
        return PartialArgsGroup.as_type()

class IDefinitionKey(INode, ABC):
    pass

D = typing.TypeVar('D', bound=IDefinitionKey)

class FunctionId(BaseInt, IDefinitionKey, IFunction, IInstantiable):
    pass

class StateDefinition(InheritableNode, typing.Generic[D, T], ABC):

    idx_definition_key = 1
    idx_definition_expr = 2

    @property
    def definition_key(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_definition_key)

    @property
    def definition_expr(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_definition_expr)

class FunctionDefinition(
    StateDefinition[FunctionId, FunctionExpr],
    IInstantiable,
):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            FunctionId.as_type(),
            FunctionExpr.as_type(),
        ))

class StateDefinitionGroup(BaseGroup[StateDefinition], IInstantiable):

    @classmethod
    def item_type(cls) -> TypeNode:
        return StateDefinition.as_type()

###########################################################
########################## STATE ##########################
###########################################################

class State(InheritableNode, IOpaqueScope, IDefault, IWrapper, IInstantiable):

    idx_meta_info = 1
    idx_scratch_group = 2
    idx_args_outer_group = 3
    idx_definition_group = 4

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateMetaInfo.as_type(),
            ScratchGroup.as_type(),
            PartialArgsOuterGroup.as_type(),
            StateDefinitionGroup.as_type(),
        ))

    @classmethod
    def create(cls) -> typing.Self:
        return cls(
            StateMetaInfo.create(),
            ScratchGroup(),
            PartialArgsOuterGroup(),
            StateDefinitionGroup(),
        )

    @classmethod
    def create_with_goal(cls, goal_achieved: IGoalAchieved) -> typing.Self:
        return cls(
            StateMetaInfo.with_goal_achieved(goal_achieved),
            ScratchGroup(),
            PartialArgsOuterGroup(),
            StateDefinitionGroup(),
        )

    @property
    def meta_info(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_meta_info)

    @property
    def scratch_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_scratch_group)

    @property
    def args_outer_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_args_outer_group)

    @property
    def definition_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_definition_group)

    def goal_achieved(self) -> bool:
        meta_info = self.meta_info.apply().cast(StateMetaInfo)
        goal_achieved = meta_info.goal_achieved.apply().cast(IGoalAchieved)
        result = goal_achieved.as_bool is True
        return result

    def with_new_args(
        self,
        meta_info: StateMetaInfo | None = None,
        scratch_group: ScratchGroup | None = None,
        args_outer_group: PartialArgsOuterGroup | None = None,
        definition_group: StateDefinitionGroup | None = None,
    ) -> typing.Self:
        return self.func(
            meta_info or self.meta_info.apply(),
            scratch_group or self.scratch_group.apply(),
            args_outer_group or self.args_outer_group.apply(),
            definition_group or self.definition_group.apply(),
        )

    @classmethod
    def with_args(
        cls,
        meta_info: StateMetaInfo | None = None,
        scratch_group: ScratchGroup | None = None,
        args_outer_group: PartialArgsOuterGroup | None = None,
        definition_group: StateDefinitionGroup | None = None,
    ) -> typing.Self:
        return cls(
            meta_info or StateMetaInfo.create(),
            scratch_group or ScratchGroup(),
            args_outer_group or PartialArgsOuterGroup(),
            definition_group or StateDefinitionGroup())

    @classmethod
    def from_raw(
        cls,
        meta_info: StateMetaInfo | None = None,
        scratches: typing.Sequence[INode | None] | None = None,
        args_groups: typing.Sequence[PartialArgsGroup] | None = None,
        definitions: typing.Sequence[StateDefinition] | None = None,
    ) -> typing.Self:
        scratches = scratches or tuple()
        args_groups = args_groups or tuple()
        definitions = definitions or tuple()
        return cls.with_args(
            meta_info=meta_info,
            scratch_group=ScratchGroup.from_raw_items(scratches),
            args_outer_group=PartialArgsOuterGroup.from_items(args_groups),
            definition_group=StateDefinitionGroup.from_items(definitions))

###########################################################
######################### INDICES #########################
###########################################################

class IStateIndex(ITypedIndex[State, T], typing.Generic[T], ABC):

    @classmethod
    def outer_type(cls) -> type[State]:
        return State

    @classmethod
    def item_type(cls) -> TypeNode:
        raise NotImplementedError

    def find_in_outer_node(self, node: State) -> IOptional[T]:
        raise NotImplementedError

    def replace_in_outer_target(self, target: State, new_node: T) -> IOptional[State]:
        raise NotImplementedError

    def remove_in_outer_target(self, target: State) -> IOptional[State]:
        raise NotImplementedError

class StateIntIndex(BaseInt, IStateIndex[T], ITypedIntIndex[State, T], typing.Generic[T], ABC):
    pass

class StateDynamicGoalIndex(StateIntIndex[DynamicGoal], IInstantiable):

    @classmethod
    def item_type(cls) -> TypeNode:
        return DynamicGoal.as_type()

    @classmethod
    def _state_dynamic_goal_group(cls, state: State) -> DynamicGoalGroup:
        return state.meta_info.apply().cast(
            StateMetaInfo
        ).dynamic_goal_group.apply().cast(DynamicGoalGroup)

    @classmethod
    def _update_state(
        cls,
        target: State,
        group_opt: IOptional[DynamicGoalGroup],
    ) -> IOptional[State]:
        group = group_opt.value
        if group is None:
            return Optional.create()
        assert isinstance(group, DynamicGoalGroup)
        return Optional(target.with_new_args(
            meta_info=target.meta_info.apply().cast(StateMetaInfo).with_new_args(
                dynamic_goal_group=group,
            ),
        ))

    def find_in_outer_node(self, node: State) -> IOptional[DynamicGoal]:
        return self.find_arg(self._state_dynamic_goal_group(node))

    def replace_in_outer_target(self, target: State, new_node: DynamicGoal) -> IOptional[State]:
        group_opt = self.replace_arg(self._state_dynamic_goal_group(target), new_node)
        return self._update_state(target, group_opt)

    def remove_in_outer_target(self, target: State) -> IOptional[State]:
        group_opt = self.remove_arg(self._state_dynamic_goal_group(target))
        return self._update_state(target, group_opt)

class StateScratchIndex(StateIntIndex[Scratch], IInstantiable):

    @classmethod
    def item_type(cls) -> TypeNode:
        return Scratch.as_type()

    @classmethod
    def _update_state(cls, target: State, group_opt: IOptional[BaseNode]) -> IOptional[State]:
        group = group_opt.value
        if group is None:
            return Optional.create()
        assert isinstance(group, ScratchGroup)
        return Optional(target.with_new_args(
            scratch_group=group,
        ))

    def find_in_outer_node(self, node: State) -> IOptional[Scratch]:
        return self.find_arg(node.scratch_group.apply())

    def replace_in_outer_target(self, target: State, new_node: Scratch) -> IOptional[State]:
        group_opt = self.replace_arg(target.scratch_group.apply(), new_node)
        return self._update_state(target, group_opt)

    def remove_in_outer_target(self, target: State) -> IOptional[State]:
        group_opt = self.remove_arg(target.scratch_group.apply())
        return self._update_state(target, group_opt)

class ScratchNodeIndex(NodeIntBaseIndex, IInstantiable):

    def find_in_node(self, node: INode):
        assert isinstance(node, Scratch)
        content = node.value
        if content is None:
            return Optional.create()
        return NodeMainIndex(self.as_int).find_in_node(content)

    def replace_in_target(self, target_node: INode, new_node: INode):
        assert isinstance(target_node, Scratch)
        assert isinstance(new_node, INode)
        if self.as_int == 1:
            return Optional(new_node)
        old_content = target_node.value_or_raise
        content = NodeMainIndex(self.as_int).replace_in_target(old_content, new_node)
        return content

    def remove_in_target(self, target_node: INode):
        assert isinstance(target_node, Scratch)
        if self.as_int == 1:
            return Optional.create()
        old_content = target_node.value_or_raise
        content = NodeMainIndex(self.as_int).remove_in_target(old_content)
        return content

class StateArgsGroupIndex(StateIntIndex[PartialArgsGroup], IInstantiable):

    @classmethod
    def item_type(cls) -> TypeNode:
        return PartialArgsGroup.as_type()

    @classmethod
    def _update_state(cls, target: State, group_opt: IOptional[BaseNode]) -> IOptional[State]:
        args_outer_group = group_opt.value
        if args_outer_group is None:
            return Optional.create()
        assert isinstance(args_outer_group, PartialArgsOuterGroup)
        return Optional(target.with_new_args(args_outer_group=args_outer_group))

    def find_in_outer_node(self, node: State) -> IOptional[PartialArgsGroup]:
        return self.find_arg(node.args_outer_group.apply())

    def replace_in_outer_target(
        self,
        target: State,
        new_node: PartialArgsGroup,
    ) -> IOptional[State]:
        group_opt = self.replace_arg(target.args_outer_group.apply(), new_node)
        return self._update_state(target, group_opt)

    def remove_in_outer_target(self, target: State) -> IOptional[State]:
        group_opt = self.remove_arg(target.args_outer_group.apply())
        return self._update_state(target, group_opt)

class StateArgsGroupArgIndex(InheritableNode, IStateIndex[INode], IInstantiable):

    idx_group_index = 1
    idx_arg_index = 2

    @classmethod
    def item_type(cls) -> TypeNode:
        return INode.as_type()

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StateArgsGroupIndex.as_type(),
            NodeArgIndex.as_type(),
        ))

    @property
    def group_index(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_group_index)

    @property
    def arg_index(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_arg_index)

    def find_in_outer_node(self, node: State):
        group_index = self.group_index.apply().cast(StateArgsGroupIndex)
        arg_index = self.arg_index.apply().cast(NodeArgIndex)

        group = group_index.find_in_outer_node(node).value
        if group is None:
            return Optional.create()

        return arg_index.find_in_node(group)

    def replace_in_outer_target(self, target: State, new_node: INode):
        group_index = self.group_index.apply().cast(StateArgsGroupIndex)
        arg_index = self.arg_index.apply().cast(NodeArgIndex)

        group = group_index.find_in_outer_node(target).value
        if group is None:
            return Optional.create()
        new_group = arg_index.replace_in_target(
            target_node=group,
            new_node=new_node,
        ).value
        if new_group is None:
            return Optional.create()
        result = group_index.replace_in_outer_target(target, new_group)

        return Optional(result)

    def remove_in_outer_target(self, target: State):
        group_index = self.group_index.apply().cast(StateArgsGroupIndex)
        arg_index = self.arg_index.apply().cast(NodeArgIndex)

        group = group_index.find_in_outer_node(target).value
        if group is None:
            return Optional.create()
        new_group = arg_index.remove_in_target(group).value
        if new_group is None:
            return Optional.create()
        result = group_index.replace_in_outer_target(target, new_group)

        return Optional(result)

class StateDefinitionIndex(StateIntIndex[StateDefinition], IInstantiable):

    @classmethod
    def item_type(cls) -> TypeNode:
        return StateDefinition.as_type()

    def find_in_outer_node(self, node: State) -> IOptional[StateDefinition]:
        return self.find_arg(node.definition_group.apply())

    def replace_in_outer_target(
        self,
        target: State,
        new_node: StateDefinition,
    ) -> IOptional[State]:
        definitions = list(target.definition_group.apply().cast(StateDefinitionGroup).as_tuple)
        for i, definition in enumerate(definitions):
            if self.as_int == (i+1):
                Eq(
                    new_node.definition_key.apply(),
                    definition.definition_key.apply(),
                ).raise_on_false()
                definitions[i] = new_node
                return Optional(target.with_new_args(
                    definition_group=StateDefinitionGroup.from_items(definitions),
                ))
        return Optional.create()

    def remove_in_outer_target(self, target: State) -> IOptional[State]:
        definitions = list(target.definition_group.apply().cast(StateDefinitionGroup).as_tuple)
        for i, _ in enumerate(definitions):
            if self.as_int == (i+1):
                definitions.pop(i)
                return Optional(target.with_new_args(
                    definition_group=StateDefinitionGroup.from_items(definitions),
                ))
        return Optional.create()
