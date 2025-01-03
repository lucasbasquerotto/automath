import typing
from abc import ABC
from environment.core import (
    BASIC_NODE_TYPES,
    IDefault,
    IFromInt,
    IInt,
    INodeIndex,
    ISingleChild,
    IGroup,
    IFunction,
    IBoolean,
    InheritableNode,
    INode,
    BaseGroup,
    TypeNode,
    Optional,
    IOptional)
from environment.state import State
from environment.action import BaseAction

T = typing.TypeVar('T', bound=INode)

class GoalNode(InheritableNode):
    def evaluate(self, state: State) -> bool:
        raise NotImplementedError

class IMetaData(INode, ABC):
    pass

class GeneralTypeGroup(BaseGroup[TypeNode[T]], typing.Generic[T]):

    @classmethod
    def item_type(cls):
        return TypeNode

class DetailedType(InheritableNode, ISingleChild[TypeNode[T]], typing.Generic[T]):
    def __init__(self, node_type: TypeNode[T]):
        super().__init__(node_type)

    @property
    def child(self) -> TypeNode[T]:
        node_type = self.args[0]
        return typing.cast(TypeNode[T], node_type)

    @classmethod
    def with_child(cls, child: TypeNode[T]) -> typing.Self:
        return cls(child)

class DetailedTypeGroup(BaseGroup[DetailedType[T]], typing.Generic[T]):

    @classmethod
    def item_type(cls):
        return DetailedType

    @classmethod
    def from_types(cls, types: typing.Sequence[TypeNode[T]]) -> typing.Self:
        return cls.from_items([DetailedType.with_child(node_type) for node_type in types])

    def to_type_group(self) -> GeneralTypeGroup[T]:
        return GeneralTypeGroup.from_items([item.child for item in self.as_tuple])

class SubtypeOuterGroup(InheritableNode, typing.Generic[T]):

    def __init__(self, common_type: TypeNode[T], subtypes: GeneralTypeGroup[T]):
        super().__init__(common_type, subtypes)

    @property
    def common_type(self) -> TypeNode[T]:
        common_type = self.args[0]
        return typing.cast(TypeNode[T], common_type)

    @property
    def subtypes(self) -> GeneralTypeGroup[T]:
        subtypes = self.args[1]
        return typing.cast(GeneralTypeGroup[T], subtypes)

    def validate(self):
        super().validate()
        common_type = self.common_type
        subtypes = self.subtypes
        assert all(issubclass(item.type, common_type.type) for item in subtypes.as_tuple)

    @classmethod
    def from_all_types(cls, common_type: TypeNode[T], all_types: GeneralTypeGroup):
        assert isinstance(common_type, TypeNode)
        assert isinstance(all_types, GeneralTypeGroup)
        subtypes = GeneralTypeGroup.from_items(
            [item for item in all_types.as_tuple if issubclass(item.type, common_type.type)]
        )
        cls(common_type, subtypes)

class MetaInfoOptions(InheritableNode, IDefault):
    def __init__(
        self,
        max_history_state_size: IOptional[IInt],
    ):
        super().__init__(max_history_state_size)

    @property
    def max_history_state_size(self) -> IOptional[IInt]:
        max_history_state_size = self.args[0]
        return typing.cast(IOptional[IInt], max_history_state_size)

    @classmethod
    def create(cls) -> typing.Self:
        return cls(Optional.create())

class MetaInfo(InheritableNode):
    def __init__(
        self,
        goal: GoalNode,
        options: MetaInfoOptions,
        all_types: GeneralTypeGroup[INode],
        all_types_details: DetailedTypeGroup,
        default_group: SubtypeOuterGroup[IDefault],
        from_int_group: SubtypeOuterGroup[IFromInt],
        int_group: SubtypeOuterGroup[IInt],
        node_index_group: SubtypeOuterGroup[INodeIndex],
        single_child_group: SubtypeOuterGroup[ISingleChild],
        group_outer_group: SubtypeOuterGroup[IGroup],
        function_group: SubtypeOuterGroup[IFunction],
        boolean_group: SubtypeOuterGroup[IBoolean],
        allowed_actions: SubtypeOuterGroup[BaseAction],
    ):
        super().__init__(
            goal,
            options,
            all_types,
            all_types_details,
            default_group,
            from_int_group,
            int_group,
            node_index_group,
            single_child_group,
            group_outer_group,
            function_group,
            boolean_group,
            allowed_actions,
        )

    @property
    def goal(self) -> GoalNode:
        goal = self.args[0]
        return typing.cast(GoalNode, goal)

    @property
    def options(self) -> MetaInfoOptions:
        options = self.args[1]
        return typing.cast(MetaInfoOptions, options)

    @property
    def all_types(self) -> DetailedTypeGroup:
        all_types = self.args[1]
        return typing.cast(DetailedTypeGroup, all_types)

    @property
    def all_types_details(self) -> DetailedTypeGroup:
        all_types_details = self.args[2]
        return typing.cast(DetailedTypeGroup, all_types_details)

    @property
    def default_group(self) -> SubtypeOuterGroup[IDefault]:
        default_group = self.args[3]
        return typing.cast(SubtypeOuterGroup[IDefault], default_group)

    @property
    def from_int_group(self) -> SubtypeOuterGroup[IFromInt]:
        from_int_group = self.args[4]
        return typing.cast(SubtypeOuterGroup[IFromInt], from_int_group)

    @property
    def int_group(self) -> SubtypeOuterGroup[IInt]:
        int_group = self.args[5]
        return typing.cast(SubtypeOuterGroup[IInt], int_group)

    @property
    def node_index_group(self) -> SubtypeOuterGroup[INodeIndex]:
        node_index_group = self.args[6]
        return typing.cast(SubtypeOuterGroup[INodeIndex], node_index_group)

    @property
    def single_child_group(self) -> SubtypeOuterGroup[ISingleChild]:
        single_child_group = self.args[7]
        return typing.cast(SubtypeOuterGroup[ISingleChild], single_child_group)

    @property
    def group_outer_group(self) -> SubtypeOuterGroup[IGroup]:
        group_outer_group = self.args[8]
        return typing.cast(SubtypeOuterGroup[IGroup], group_outer_group)

    @property
    def function_group(self) -> SubtypeOuterGroup[IFunction]:
        function_group = self.args[9]
        return typing.cast(SubtypeOuterGroup[IFunction], function_group)

    @property
    def boolean_group(self) -> SubtypeOuterGroup[IBoolean]:
        boolean_group = self.args[10]
        return typing.cast(SubtypeOuterGroup[IBoolean], boolean_group)

    @property
    def allowed_actions(self) -> SubtypeOuterGroup[BaseAction]:
        allowed_actions = self.args[11]
        return typing.cast(SubtypeOuterGroup[BaseAction], allowed_actions)

    @classmethod
    def with_defaults(cls, goal: GoalNode) -> typing.Self:
        all_types = [TypeNode(t) for t in BASIC_NODE_TYPES]
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
            SubtypeOuterGroup.from_all_types(TypeNode(ISingleChild), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IGroup), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IFunction), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(IBoolean), all_types_group),
            SubtypeOuterGroup.from_all_types(TypeNode(BaseAction), all_types_group),
        )
