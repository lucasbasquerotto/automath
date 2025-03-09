import typing
from abc import ABC
import functools
from env.core import (
    IDefault,
    IFromInt,
    IInt,
    INodeIndex,
    IFromSingleNode,
    IGroup,
    IFunction,
    IBoolean,
    InheritableNode,
    INode,
    BaseGroup,
    TypeNode,
    Type,
    Optional,
    BaseInt,
    Integer,
    IOptional,
    ITypedIndex,
    ITypedIntIndex,
    Protocol,
    CountableTypeGroup,
    IntersectionType,
    BaseNode,
    IWrapper,
    IntGroup,
    CompositeType,
    OptionalTypeGroup,
    TmpInnerArg,
    IInstantiable)
from env.state import State, IGoal
from env.env_utils import load_all_superclasses

T = typing.TypeVar('T', bound=INode)
K = typing.TypeVar('K', bound=INode)

###########################################################
####################### META ITEMS ########################
###########################################################

class GeneralTypeGroup(BaseGroup[TypeNode[T]], IInstantiable, typing.Generic[T]):

    @classmethod
    def item_type(cls):
        return TypeNode

    @classmethod
    def from_types(cls, types: typing.Sequence[type[T]]) -> typing.Self:
        return cls.from_items([TypeNode(t) for t in types])

class DetailedType(
    InheritableNode,
    IInstantiable,
    typing.Generic[T],
):

    idx_node_type = 1
    idx_superclasses = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            TypeNode.as_type(),
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(Protocol.as_type()),
            ),
            GeneralTypeGroup.as_type(),
        ))

    @property
    def child(self) -> TypeNode[T]:
        return self.inner_arg(self.idx_node_type).apply().cast(TypeNode[T])

    @classmethod
    def with_child(cls, child: TypeNode[T], all_types: typing.Sequence[TypeNode[T]]) -> typing.Self:
        return cls(
            child,
            (
                Optional(child.type.protocol())
                if issubclass(child.type, IInstantiable) and child.type != IInstantiable
                else Optional()
            ),
            GeneralTypeGroup.from_types(sorted(
                [t for t in load_all_superclasses(child.type)],
                key=lambda t: all_types.index(t.as_type()),
            ))
        )

class DetailedTypeGroup(BaseGroup[DetailedType[T]], IInstantiable, typing.Generic[T]):

    @classmethod
    def item_type(cls):
        return DetailedType

    @classmethod
    def from_types(cls, types: typing.Sequence[TypeNode[T]]) -> typing.Self:
        return cls.from_items([DetailedType.with_child(node_type, types) for node_type in types])

    def to_type_group(self) -> GeneralTypeGroup[T]:
        return GeneralTypeGroup.from_items([item.child for item in self.as_tuple])

class SubtypeOuterGroup(InheritableNode, IWrapper, IInstantiable, typing.Generic[T]):

    idx_common_type = 1
    idx_subtypes = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            IntersectionType.as_type(),
            GeneralTypeGroup.as_type(),
        ))

    @classmethod
    def from_all_types(cls, common_type: TypeNode[T], all_types: GeneralTypeGroup):
        assert isinstance(common_type, TypeNode)
        assert isinstance(all_types, GeneralTypeGroup)
        full_type = IntersectionType(
            Type(common_type),
            Type(IInstantiable.as_type()),
            Type(BaseNode.as_type()),
        )
        subtypes = GeneralTypeGroup.from_items(
            [
                item
                for item in all_types.as_tuple
                if full_type.static_valid(item)
            ]
        )
        assert subtypes == GeneralTypeGroup.from_items(
            [
                item
                for item in all_types.as_tuple
                if (
                    issubclass(item.type, common_type.type)
                    and
                    issubclass(item.type, IInstantiable)
                    and
                    issubclass(item.type, BaseNode)
                )
            ]
        )
        return cls(common_type, subtypes)

    @property
    def common_type(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_common_type)

    @property
    def subtypes(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_subtypes)

    def _validate(self):
        super()._validate()
        common_type = self.common_type.apply()
        subtypes = self.subtypes.apply()
        if isinstance(common_type, TypeNode) and isinstance(subtypes, GeneralTypeGroup):
            assert all(issubclass(item.type, common_type.type) for item in subtypes.as_tuple)

class MetaInfoOptions(InheritableNode, IDefault, IInstantiable):

    idx_max_history_state_size = 1
    idx_max_steps = 2
    idx_step_count_to_change_cost = 3
    idx_cost_multiplier_default = 4
    idx_cost_multiplier_custom_goal = 5
    idx_cost_multiplier_sub_goal = 6
    idx_cost_multiplier_main_goal = 7
    idx_cost_multiplier_action = 8
    idx_cost_multiplier_step = 9
    idx_cost_full_state_memory = 10
    idx_cost_visible_state_memory = 11
    idx_cost_main_state_memory = 12
    idx_cost_run_memory = 13

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
                OptionalTypeGroup(Integer.as_type()),
            ),
            Integer.as_type(),
            Integer.as_type(),
            Integer.as_type(),
            Integer.as_type(),
            Integer.as_type(),
            Integer.as_type(),
            Integer.as_type(),
            Integer.as_type(),
            Integer.as_type(),
            Integer.as_type(),
            Integer.as_type(),
        ))

    @property
    def max_history_state_size(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_max_history_state_size)

    @property
    def max_steps(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_max_steps)

    @property
    def step_count_to_change_cost(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_step_count_to_change_cost)

    @property
    def cost_multiplier_default(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_cost_multiplier_default)

    @property
    def cost_multiplier_custom_goal(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_cost_multiplier_custom_goal)

    @property
    def cost_multiplier_sub_goal(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_cost_multiplier_sub_goal)

    @property
    def cost_multiplier_main_goal(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_cost_multiplier_main_goal)

    @property
    def cost_multiplier_action(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_cost_multiplier_action)

    @property
    def cost_multiplier_step(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_cost_multiplier_step)

    @property
    def cost_full_state_memory(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_cost_full_state_memory)

    @property
    def cost_visible_state_memory(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_cost_visible_state_memory)

    @property
    def cost_main_state_memory(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_cost_main_state_memory)

    @property
    def cost_run_memory(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_cost_run_memory)

    @classmethod
    def with_args(
        cls,
        max_history_state_size: int | None = None,
        max_steps: int | None = None,
        step_count_to_change_cost: int | None = None,
        cost_multiplier_default: int | None = None,
        cost_multiplier_custom_goal: int | None = None,
        cost_multiplier_sub_goal: int | None = None,
        cost_multiplier_main_goal: int | None = None,
        cost_multiplier_action: int | None = None,
        cost_multiplier_step: int | None = None,
        cost_full_state_memory: int | None = None,
        cost_visible_state_memory: int | None = None,
        cost_main_state_memory: int | None = None,
        cost_run_memory: int | None = None,
    ) -> typing.Self:
        return cls(
            Optional.with_int(max_history_state_size),
            Optional.with_int(max_steps),
            Integer(
                step_count_to_change_cost
                if step_count_to_change_cost is not None
                else 5
            ),
            Integer(
                cost_multiplier_default
                if cost_multiplier_default is not None
                else 20
            ),
            Integer(
                cost_multiplier_custom_goal
                if cost_multiplier_custom_goal is not None
                else 10
            ),
            Integer(
                cost_multiplier_sub_goal
                if cost_multiplier_sub_goal is not None
                else 3
            ),
            Integer(
                cost_multiplier_main_goal
                if cost_multiplier_main_goal is not None
                else 1
            ),
            Integer(
                cost_multiplier_action
                if cost_multiplier_action is not None
                else 10
            ),
            Integer(
                cost_multiplier_step
                if cost_multiplier_step is not None
                else 1
            ),
            Integer(
                cost_full_state_memory
                if cost_full_state_memory is not None
                else 1
            ),
            Integer(
                cost_visible_state_memory
                if cost_visible_state_memory is not None
                else 10
            ),
            Integer(
                cost_main_state_memory
                if cost_main_state_memory is not None
                else 100
            ),
            Integer(
                cost_run_memory
                if cost_run_memory is not None
                else 1
            ),
        )

###########################################################
################## FULL STATE BASE NODES ##################
###########################################################

class IFullState(INode, ABC):
    pass

S = typing.TypeVar('S', bound=IFullState)

class IFullStateIndex(ITypedIndex[S, T], typing.Generic[S, T], ABC):

    @classmethod
    def outer_type(cls) -> type[S]:
        raise NotImplementedError

    @classmethod
    def item_type(cls) -> type[T]:
        raise NotImplementedError

class FullStateIntBaseIndex(
    BaseInt,
    IFullStateIndex[S, T],
    ITypedIntIndex[IFullState, T],
    typing.Generic[S, T],
    ABC,
):
    pass

###########################################################
####################### ACTION INFO #######################
###########################################################

class IActionGeneralInfo(INode, ABC):
    pass

class IActionInfo(IActionGeneralInfo, ABC):
    pass

class ActionInfo(InheritableNode, IActionInfo, IDefault, IInstantiable):
    #TODO implement
    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup())

class IActionOutputInfo(IActionGeneralInfo, ABC):
    pass

class ActionOutputInfo(InheritableNode, IActionOutputInfo, IDefault, IInstantiable):
    #TODO implement
    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup())

###########################################################
#################### ACTION BASE NODES ####################
###########################################################

class IActionOutput(INode, typing.Generic[S], ABC):

    def run_output(self, full_state: S) -> tuple[State, IActionOutputInfo]:
        raise NotImplementedError

class IAction(INode, typing.Generic[S], ABC):

    def run_action(self, full_state: S) -> S:
        raise NotImplementedError

class IBasicAction(IAction[S], typing.Generic[S], ABC):

    def to_raw_args(self) -> IntGroup:
        args = self.as_node.args
        raw_args: list[int] = []
        for arg in args:
            if isinstance(arg, IOptional):
                if arg.is_empty().as_bool:
                    raw_args.append(0)
                else:
                    value: INode = arg.value_or_raise
                    assert isinstance(value, IInt)
                    raw_args.append(value.as_int)
            else:
                assert isinstance(arg, IInt)
                raw_args.append(arg.as_int)
        for _ in range(len(args), 3):
            raw_args.append(0)
        return IntGroup.from_ints(raw_args)

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        raw_args = [arg1, arg2, arg3]
        group = cls.protocol().arg_group.apply()
        assert isinstance(group, CountableTypeGroup), type(group)
        types = group.as_tuple
        assert len(types) <= 3

        result = cls._from_raw(arg1, arg2, arg3)
        assert result.to_raw_args() == IntGroup.from_ints(raw_args)

        node = result.as_node
        assert len(node.args) == len(types)
        for i, raw_arg in enumerate(raw_args):
            if i >= len(types):
                assert raw_arg == 0
            else:
                type_node = types[i]
                arg = node.args[i]
                if isinstance(type_node, CompositeType):
                    t1 = type_node.type.apply().real(TypeNode)
                    t_args = type_node.type_args.apply().real(OptionalTypeGroup)
                    t_arg = t_args.type_node.apply().real(TypeNode)
                    assert issubclass(t_arg.type, IInt)
                    type_node = t1
                assert isinstance(type_node, TypeNode)
                t = type_node.type
                if issubclass(t, IOptional):
                    assert isinstance(arg, IOptional)
                    if raw_arg == 0:
                        assert arg.is_empty().as_bool
                    else:
                        value: INode = arg.value_or_raise
                        assert isinstance(value, IInt)
                        assert value == value.from_int(raw_arg)
                else:
                    assert issubclass(t, IInt)
                    assert isinstance(arg, IInt)
                    assert arg == t.from_int(raw_arg)
        return result

    @classmethod
    def _from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        raise NotImplementedError

class IRawAction(IAction[S], typing.Generic[S], ABC):

    def to_action(self, full_state: S) -> IBasicAction[S]:
        raise NotImplementedError

###########################################################
######################## META INFO ########################
###########################################################

class MetaMainArgs:

    def __init__(
        self,
        all_types_group: GeneralTypeGroup,
        allowed_basic_actions_group: GeneralTypeGroup,
        all_types_details: DetailedTypeGroup,
        default_group: SubtypeOuterGroup[IDefault],
        from_int_group: SubtypeOuterGroup[IFromInt],
        int_group: SubtypeOuterGroup[IInt],
        node_index_group: SubtypeOuterGroup[INodeIndex],
        full_state_index_group: SubtypeOuterGroup[IFullStateIndex],
        full_state_int_index_group: SubtypeOuterGroup[FullStateIntBaseIndex],
        single_child_group: SubtypeOuterGroup[IFromSingleNode],
        group_outer_group: SubtypeOuterGroup[IGroup],
        function_group: SubtypeOuterGroup[IFunction],
        boolean_group: SubtypeOuterGroup[IBoolean],
        instantiable_group: SubtypeOuterGroup[IInstantiable],
        basic_actions: SubtypeOuterGroup[IBasicAction],
        all_actions: SubtypeOuterGroup[IAction],
    ):
        self.all_types_group = all_types_group
        self.allowed_basic_actions_group = allowed_basic_actions_group
        self.all_types_details = all_types_details
        self.default_group = default_group
        self.from_int_group = from_int_group
        self.int_group = int_group
        self.node_index_group = node_index_group
        self.full_state_index_group = full_state_index_group
        self.full_state_int_index_group = full_state_int_index_group
        self.single_child_group = single_child_group
        self.group_outer_group = group_outer_group
        self.function_group = function_group
        self.boolean_group = boolean_group
        self.instantiable_group = instantiable_group
        self.basic_actions = basic_actions
        self.all_actions = all_actions

@functools.cache
def _meta_main_args(all_types: tuple[TypeNode, ...]) -> MetaMainArgs:
    for t in all_types:
        if issubclass(t.type, IInstantiable):
            for st in t.type.__bases__:
                if st != IInstantiable:
                    assert not issubclass(st, IInstantiable), \
                        f"Instantiable class {t.type} has subclass {st}"

            if t.type != IInstantiable:
                protocol = t.type.protocol()
                protocol.valid_protocol()

    all_types_group = GeneralTypeGroup.from_items(all_types)
    allowed_basic_actions_group = GeneralTypeGroup.from_items(
        [
            t for t in all_types
            if issubclass(t.type, IBasicAction) and issubclass(t.type, IInstantiable)
        ]
    )
    return MetaMainArgs(
        all_types_group=all_types_group,
        allowed_basic_actions_group=allowed_basic_actions_group,
        all_types_details=DetailedTypeGroup.from_types(
            all_types),
        default_group=SubtypeOuterGroup.from_all_types(
            TypeNode(IDefault), all_types_group),
        from_int_group=SubtypeOuterGroup.from_all_types(
            TypeNode(IFromInt), all_types_group),
        int_group=SubtypeOuterGroup.from_all_types(
            TypeNode(IInt), all_types_group),
        node_index_group=SubtypeOuterGroup.from_all_types(
            TypeNode(INodeIndex), all_types_group),
        full_state_index_group=SubtypeOuterGroup.from_all_types(
            TypeNode(IFullStateIndex), all_types_group),
        full_state_int_index_group=SubtypeOuterGroup.from_all_types(
            TypeNode(FullStateIntBaseIndex), all_types_group),
        single_child_group=SubtypeOuterGroup.from_all_types(
            TypeNode(IFromSingleNode), all_types_group),
        group_outer_group=SubtypeOuterGroup.from_all_types(
            TypeNode(IGroup), all_types_group),
        function_group=SubtypeOuterGroup.from_all_types(
            TypeNode(IFunction), all_types_group),
        boolean_group=SubtypeOuterGroup.from_all_types(
            TypeNode(IBoolean), all_types_group),
        instantiable_group=SubtypeOuterGroup.from_all_types(
            TypeNode(IInstantiable), all_types_group),
        basic_actions=SubtypeOuterGroup.from_all_types(
            TypeNode(IBasicAction), all_types_group),
        all_actions=SubtypeOuterGroup.from_all_types(
            TypeNode(IAction), all_types_group),
    )
class MetaInfo(InheritableNode, IWrapper, IInstantiable):
    idx_goal = 1
    idx_options = 2
    idx_all_types = 3
    idx_allowed_basic_actions = 4
    idx_allowed_actions = 5
    idx_all_types_details = 6
    idx_default_group = 7
    idx_from_int_group = 8
    idx_int_group = 9
    idx_node_index_group = 10
    idx_full_state_index_group = 11
    idx_full_state_int_index_group = 12
    idx_single_child_group = 13
    idx_group_outer_group = 14
    idx_function_group = 15
    idx_boolean_group = 16
    idx_instantiable_group = 17
    idx_basic_actions = 18
    idx_all_actions = 19

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            IGoal.as_type(),
            MetaInfoOptions.as_type(),
            GeneralTypeGroup.as_type(),
            CompositeType(
                GeneralTypeGroup.as_type(),
                CountableTypeGroup(IBasicAction.as_type()),
            ),
            CompositeType(
                GeneralTypeGroup.as_type(),
                CountableTypeGroup(IAction.as_type()),
            ),
            DetailedTypeGroup.as_type(),
            CompositeType(
                SubtypeOuterGroup.as_type(),
                CountableTypeGroup(IDefault.as_type()),
            ),
            CompositeType(
                SubtypeOuterGroup.as_type(),
                CountableTypeGroup(IFromInt.as_type()),
            ),
            CompositeType(
                SubtypeOuterGroup.as_type(),
                CountableTypeGroup(IInt.as_type()),
            ),
            CompositeType(
                SubtypeOuterGroup.as_type(),
                CountableTypeGroup(INodeIndex.as_type()),
            ),
            CompositeType(
                SubtypeOuterGroup.as_type(),
                CountableTypeGroup(IFullStateIndex.as_type()),
            ),
            CompositeType(
                SubtypeOuterGroup.as_type(),
                CountableTypeGroup(FullStateIntBaseIndex.as_type()),
            ),
            CompositeType(
                SubtypeOuterGroup.as_type(),
                CountableTypeGroup(IFromSingleNode.as_type()),
            ),
            CompositeType(
                SubtypeOuterGroup.as_type(),
                CountableTypeGroup(IGroup.as_type()),
            ),
            CompositeType(
                SubtypeOuterGroup.as_type(),
                CountableTypeGroup(IFunction.as_type()),
            ),
            CompositeType(
                SubtypeOuterGroup.as_type(),
                CountableTypeGroup(IBoolean.as_type()),
            ),
            CompositeType(
                SubtypeOuterGroup.as_type(),
                CountableTypeGroup(IInstantiable.as_type()),
            ),
            CompositeType(
                SubtypeOuterGroup.as_type(),
                CountableTypeGroup(IBasicAction.as_type()),
            ),
            CompositeType(
                SubtypeOuterGroup.as_type(),
                CountableTypeGroup(IAction.as_type()),
            ),
        ))

    @classmethod
    @functools.cache
    def with_defaults(
        cls,
        goal: IGoal,
        all_types: tuple[TypeNode, ...],
        allowed_actions: tuple[TypeNode[IAction], ...] | None = None,
        max_history_state_size: int | None = None,
        max_steps: int | None = None,
    ) -> typing.Self:
        main_args = _meta_main_args(all_types)
        allowed_actions_list = [
            t
            for t in all_types
            if (
                issubclass(t.type, IAction)
                and issubclass(t.type, IInstantiable)
                and (allowed_actions is None or t in allowed_actions)
            )
        ]
        allowed_actions_group = GeneralTypeGroup.from_items(allowed_actions_list)

        return cls(
            goal,
            MetaInfoOptions.with_args(
                max_history_state_size=max_history_state_size,
                max_steps=max_steps,
            ),
            main_args.all_types_group,
            main_args.allowed_basic_actions_group,
            allowed_actions_group,
            main_args.all_types_details,
            main_args.default_group,
            main_args.from_int_group,
            main_args.int_group,
            main_args.node_index_group,
            main_args.full_state_index_group,
            main_args.full_state_int_index_group,
            main_args.single_child_group,
            main_args.group_outer_group,
            main_args.function_group,
            main_args.boolean_group,
            main_args.instantiable_group,
            main_args.basic_actions,
            main_args.all_actions,
        )

    @property
    def goal(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_goal)

    @property
    def options(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_options)

    @property
    def all_types(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_all_types)

    @property
    def allowed_basic_actions(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_allowed_basic_actions)

    @property
    def allowed_actions(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_allowed_actions)

    @property
    def all_types_details(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_all_types_details)

    @property
    def default_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_default_group)

    @property
    def from_int_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_from_int_group)

    @property
    def int_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_int_group)

    @property
    def node_index_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_node_index_group)

    @property
    def full_state_index_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_full_state_index_group)

    @property
    def full_state_int_index_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_full_state_int_index_group)

    @property
    def single_child_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_single_child_group)

    @property
    def group_outer_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_group_outer_group)

    @property
    def function_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_function_group)

    @property
    def boolean_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_boolean_group)

    @property
    def instantiable_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_instantiable_group)

    @property
    def basic_actions(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_basic_actions)

    @property
    def all_actions(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_all_actions)
