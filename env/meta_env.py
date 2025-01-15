import typing
from abc import ABC
from env.core import (
    IDefault,
    IFromInt,
    IInt,
    INodeIndex,
    IFromSingleChild,
    IGroup,
    IFunction,
    IBoolean,
    InheritableNode,
    INode,
    BaseGroup,
    Integer,
    TypeNode,
    Optional,
    BaseInt,
    IOptional,
    ITypedIndex,
    ITypedIntIndex,
    ExtendedTypeGroup,
    CountableTypeGroup,
    IntersectionType,
    BaseNode,
    IWrapper,
    TmpNestedArg,
    IInstantiable)
from env.state import State, IGoal
from env.env_utils import load_all_superclasses

T = typing.TypeVar('T', bound=INode)
K = typing.TypeVar('K', bound=INode)

###########################################################
####################### META ITEMS ########################
###########################################################

class MetaData(InheritableNode, IInstantiable):

    idx_remaining_steps = 1

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            Optional[Integer],
        ]))

    @property
    def remaining_steps(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_remaining_steps)

    @classmethod
    def with_args(
        cls,
        remaining_steps: int | None = None,
    ) -> typing.Self:
        return cls(
            Optional.with_int(remaining_steps),
        )

    def with_new_args(
        self,
        remaining_steps: int | None = None,
    ) -> typing.Self:
        return self.func(
            (
                Optional.with_int(remaining_steps)
                if remaining_steps is not None
                else self.remaining_steps.apply()
            ),
        )

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
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            TypeNode[T],
            Optional[ExtendedTypeGroup],
            GeneralTypeGroup[INode],
        ]))

    @property
    def child(self) -> TypeNode[T]:
        return self.nested_arg(self.idx_node_type).apply().cast(TypeNode[T])

    @classmethod
    def with_child(cls, child: TypeNode[T], all_types: typing.Sequence[TypeNode[T]]) -> typing.Self:
        return cls(
            child,
            (
                Optional(child.type.arg_type_group())
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
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IntersectionType,
            GeneralTypeGroup,
        ]))

    @classmethod
    def from_all_types(cls, common_type: TypeNode[T], all_types: GeneralTypeGroup):
        assert isinstance(common_type, TypeNode)
        assert isinstance(all_types, GeneralTypeGroup)
        types = IntersectionType(common_type, IInstantiable.as_type(), BaseNode.as_type())
        subtypes = GeneralTypeGroup.from_items(
            [
                item
                for item in all_types.as_tuple
                if types.accepts(item.type.as_type())
            ]
        )
        return cls(common_type, subtypes)

    @property
    def common_type(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_common_type)

    @property
    def subtypes(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_subtypes)

    def validate(self):
        super().validate()
        common_type = self.common_type.apply()
        subtypes = self.subtypes.apply()
        if isinstance(common_type, TypeNode) and isinstance(subtypes, GeneralTypeGroup):
            assert all(issubclass(item.type, common_type.type) for item in subtypes.as_tuple)

class MetaInfoOptions(InheritableNode, IDefault, IInstantiable):

    idx_max_history_state_size = 1
    idx_max_steps = 2

    @classmethod
    def create(cls) -> typing.Self:
        return cls(Optional.create(), Optional.create())

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IOptional[IInt],
            IOptional[IInt],
        ]))

    @property
    def max_history_state_size(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_max_history_state_size)

    @property
    def max_steps(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_max_steps)

    @classmethod
    def with_args(
        cls,
        max_history_state_size: int | None = None,
        max_steps: int | None = None,
    ) -> typing.Self:
        return cls(
            Optional.with_int(max_history_state_size),
            Optional.with_int(max_steps),
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

class IFullStateIntIndex(
    BaseInt,
    IFullStateIndex[S, T],
    ITypedIntIndex[IFullState, T],
    typing.Generic[S, T],
    ABC,
):
    pass

###########################################################
#################### ACTION BASE NODES ####################
###########################################################

class IActionOutput(INode, typing.Generic[S], ABC):

    def apply(self, full_state: S) -> State:
        raise NotImplementedError

class IAction(INode, typing.Generic[S], ABC):

    def run_action(self, full_state: S) -> S:
        raise NotImplementedError

class IBasicAction(IAction[S], typing.Generic[S], ABC):

    @classmethod
    def from_raw(cls, arg1: int, arg2: int, arg3: int) -> typing.Self:
        raise NotImplementedError


###########################################################
######################## META INFO ########################
###########################################################

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
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IGoal,
            MetaInfoOptions,
            GeneralTypeGroup[INode],
            GeneralTypeGroup[IBasicAction],
            GeneralTypeGroup[IAction],
            DetailedTypeGroup,
            SubtypeOuterGroup[IDefault],
            SubtypeOuterGroup[IFromInt],
            SubtypeOuterGroup[IInt],
            SubtypeOuterGroup[INodeIndex],
            SubtypeOuterGroup[IFullStateIndex],
            SubtypeOuterGroup[IFullStateIntIndex],
            SubtypeOuterGroup[IFromSingleChild],
            SubtypeOuterGroup[IGroup],
            SubtypeOuterGroup[IFunction],
            SubtypeOuterGroup[IBoolean],
            SubtypeOuterGroup[IInstantiable],
            SubtypeOuterGroup[IBasicAction],
            SubtypeOuterGroup[IAction],
        ]))

    @classmethod
    def with_defaults(
        cls,
        goal: IGoal,
        all_types: typing.Sequence[TypeNode],
        allowed_actions: typing.Sequence[TypeNode[IAction]] | None = None,
        max_history_state_size: int | None = None,
        max_steps: int | None = None,
    ) -> typing.Self:
        for t in all_types:
            if issubclass(t.type, IInstantiable):
                for st in t.type.__bases__:
                    if st != IInstantiable:
                        assert not issubclass(st, IInstantiable), \
                            f"Instantiable class {t.type} has subclass {st}"

        all_types_group = GeneralTypeGroup.from_items(all_types)
        allowed_actions = [
            t
            for t in all_types
            if (
                issubclass(t.type, IAction)
                and issubclass(t.type, IInstantiable)
                and (allowed_actions is None or t in allowed_actions)
            )
        ]
        allowed_actions_group = GeneralTypeGroup.from_items(allowed_actions)
        allowed_basic_actions_group = GeneralTypeGroup.from_items(
            [
                t for t in all_types
                if issubclass(t.type, IBasicAction) and issubclass(t.type, IInstantiable)
            ]
        )
        return cls(
            goal,
            MetaInfoOptions.with_args(
                max_history_state_size=max_history_state_size,
                max_steps=max_steps,
            ),
            all_types_group,
            allowed_basic_actions_group,
            allowed_actions_group,
            DetailedTypeGroup.from_types(all_types),
            SubtypeOuterGroup.from_all_types(TypeNode(IDefault), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IFromInt), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IInt), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(INodeIndex), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IFullStateIndex), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IFullStateIntIndex), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IFromSingleChild), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IGroup), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IFunction), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IBoolean), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IInstantiable), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IBasicAction), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IAction), all_types_group),
        )

    @property
    def goal(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_goal)

    @property
    def options(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_options)

    @property
    def all_types(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_all_types)

    @property
    def allowed_basic_actions(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_allowed_basic_actions)

    @property
    def allowed_actions(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_allowed_actions)

    @property
    def all_types_details(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_all_types_details)

    @property
    def default_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_default_group)

    @property
    def from_int_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_from_int_group)

    @property
    def int_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_int_group)

    @property
    def node_index_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_node_index_group)

    @property
    def full_state_index_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_full_state_index_group)

    @property
    def full_state_int_index_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_full_state_int_index_group)

    @property
    def single_child_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_single_child_group)

    @property
    def group_outer_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_group_outer_group)

    @property
    def function_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_function_group)

    @property
    def boolean_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_boolean_group)

    @property
    def instantiable_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_instantiable_group)

    @property
    def basic_actions(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_basic_actions)

    @property
    def all_actions(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_all_actions)
