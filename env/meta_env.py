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
    TypeNode,
    Optional,
    BaseInt,
    IOptional,
    ITypedIndex,
    ITypedIntIndex,
    ExtendedTypeGroup,
    CountableTypeGroup,
    TmpNestedArg,
    IInstantiable)
from env.state import State

T = typing.TypeVar('T', bound=INode)

###########################################################
####################### META ITEMS ########################
###########################################################

class GoalNode(InheritableNode, ABC):

    def evaluate(self, state: State) -> bool:
        raise NotImplementedError

class IMetaData(INode, ABC):
    pass

class GeneralTypeGroup(BaseGroup[TypeNode[T]], typing.Generic[T], IInstantiable):

    @classmethod
    def item_type(cls):
        return TypeNode

class DetailedType(
    InheritableNode,
    IFromSingleChild[TypeNode[T]],
    typing.Generic[T],
    IInstantiable,
):
    idx_node_type = 0

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            TypeNode[T],
        ]))

    @property
    def child(self) -> TypeNode[T]:
        node_type = self.args[self.idx_node_type]
        return typing.cast(TypeNode[T], node_type)

    @classmethod
    def with_child(cls, child: TypeNode[T]) -> typing.Self:
        return cls(child)

class DetailedTypeGroup(BaseGroup[DetailedType[T]], typing.Generic[T], IInstantiable):

    @classmethod
    def item_type(cls):
        return DetailedType

    @classmethod
    def from_types(cls, types: typing.Sequence[TypeNode[T]]) -> typing.Self:
        return cls.from_items([DetailedType.with_child(node_type) for node_type in types])

    def to_type_group(self) -> GeneralTypeGroup[T]:
        return GeneralTypeGroup.from_items([item.child for item in self.as_tuple])

class SubtypeOuterGroup(InheritableNode, typing.Generic[T], IInstantiable):
    idx_common_type = 0
    idx_subtypes = 1

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            TypeNode[T],
            GeneralTypeGroup[T],
        ]))

    def validate(self):
        super().validate()
        common_type = self.args[self.idx_common_type]
        subtypes = self.args[self.idx_subtypes]
        if isinstance(common_type, TypeNode) and isinstance(subtypes, GeneralTypeGroup):
            assert all(issubclass(item.type, common_type.type) for item in subtypes.as_tuple)

    @classmethod
    def from_all_types(cls, common_type: TypeNode[T], all_types: GeneralTypeGroup):
        assert isinstance(common_type, TypeNode)
        assert isinstance(all_types, GeneralTypeGroup)
        subtypes = GeneralTypeGroup.from_items(
            [item for item in all_types.as_tuple if issubclass(item.type, common_type.type)]
        )
        cls(common_type, subtypes)

class MetaInfoOptions(InheritableNode, IDefault, IInstantiable):
    idx_max_history_state_size = 0

    @classmethod
    def create(cls) -> typing.Self:
        return cls(Optional.create())

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IOptional[IInt],
        ]))

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

class MetaInfo(InheritableNode, IInstantiable):
    idx_goal = 0
    idx_options = 1
    idx_all_types = 2
    idx_all_types_details = 3
    idx_default_group = 4
    idx_from_int_group = 5
    idx_int_group = 6
    idx_node_index_group = 7
    idx_full_state_index_group = 8
    idx_full_state_int_index_group = 9
    idx_single_child_group = 10
    idx_group_outer_group = 11
    idx_function_group = 12
    idx_boolean_group = 13
    idx_allowed_actions = 14
    idx_basic_actions = 15

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            GoalNode,
            MetaInfoOptions,
            GeneralTypeGroup[INode],
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
            SubtypeOuterGroup[IAction],
            SubtypeOuterGroup[IBasicAction],
        ]))

    @classmethod
    def with_defaults(cls, goal: GoalNode, all_types: typing.Sequence[TypeNode]) -> typing.Self:
        all_types_group = GeneralTypeGroup.from_items(all_types)
        return cls(
            goal,
            MetaInfoOptions.create(),
            all_types_group,
            DetailedTypeGroup.from_types(all_types),
            SubtypeOuterGroup.from_all_types(TypeNode(INode), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IFromInt), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IInt), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(INodeIndex), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IFullStateIndex), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IFullStateIntIndex), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IFromSingleChild), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IGroup), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IFunction), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IBoolean), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IAction), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IBasicAction), all_types_group),
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
