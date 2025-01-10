# pylint: disable=too-many-lines
from __future__ import annotations
from abc import ABC
import typing
import sympy

###########################################################
##################### MAIN INTERFACES #####################
###########################################################

class INode(ABC):

    @classmethod
    def as_type(cls) -> TypeNode[typing.Self]:
        return TypeNode(cls)

    @classmethod
    def new(cls, *args: int | INode | typing.Type[INode]) -> typing.Self:
        raise NotImplementedError(cls)

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        raise NotImplementedError(cls)

    @property
    def as_node(self) -> BaseNode:
        raise NotImplementedError(self.__class__)

    @property
    def as_instance(self) -> IInstantiable:
        raise NotImplementedError(self.__class__)

    @property
    def func(self) -> typing.Type[typing.Self]:
        return type(self)

class ISpecialValue(ABC):

    @property
    def node_value(self) -> INode:
        raise NotImplementedError(self.__class__)

class IDefault(INode, ABC):

    @classmethod
    def create(cls) -> typing.Self:
        return cls.new()

    @property
    def as_node(self) -> BaseNode:
        raise NotImplementedError(self.__class__)

class IFromInt(INode, ABC):

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        raise NotImplementedError(cls)

class IInt(IFromInt, ABC):

    @property
    def as_int(self) -> int:
        raise NotImplementedError(self.__class__)

class INodeIndex(INode, ABC):

    def find_in_node(self, node: INode) -> IOptional[INode]:
        raise NotImplementedError(self.__class__)

    def replace_in_target(
        self,
        target_node: INode,
        new_node: INode,
    ) -> IOptional[INode]:
        raise NotImplementedError

    def remove_in_target(self, target_node: INode) -> IOptional[INode]:
        raise NotImplementedError(self.__class__)

T = typing.TypeVar('T', bound=INode)
O = typing.TypeVar('O', bound=INode)
K = typing.TypeVar('K', bound=INode)
INT = typing.TypeVar('INT', bound=IInt)

class IFromSingleChild(INode, typing.Generic[T], ABC):

    @classmethod
    def with_child(cls, child: T) -> typing.Self:
        return cls.new(child)

class ISingleChild(IFromSingleChild, typing.Generic[T], ABC):

    @property
    def child(self) -> T:
        raise NotImplementedError(self.__class__)

class IOptional(IDefault, IFromSingleChild[T], typing.Generic[T], ABC):

    @property
    def value(self) -> T | None:
        raise NotImplementedError()

    def value_or_else(self, default_value: T) -> T:
        value = self.value
        if value is None:
            value = default_value
        return value

    @property
    def value_or_raise(self) -> T:
        value = IsEmpty(self).value_or_raise
        return typing.cast(T, value)

    def raise_if_empty(self):
        IsEmpty(self).raise_if_empty()

class ISingleOptionalChild(ISingleChild[IOptional[T]], typing.Generic[T], ABC):

    @classmethod
    def with_optional(cls, child: T | None) -> typing.Self:
        return cls.with_child(Optional(child) if child is not None else Optional())

    @property
    def child(self) -> IOptional[T]:
        raise NotImplementedError(self.__class__)

class IGroup(INode, typing.Generic[T], ABC):

    @classmethod
    def item_type(cls) -> type[T]:
        raise NotImplementedError(cls)

    @property
    def as_tuple(self) -> tuple[T, ...]:
        raise NotImplementedError(self.__class__)

    @classmethod
    def from_items(cls, items: typing.Sequence[T]) -> typing.Self:
        return cls(*items)

class IFunction(INode, ABC):

    def with_arg_group(self, group: BaseGroup) -> INode:
        raise NotImplementedError(self.__class__)

class IBoolean(INode):

    @property
    def as_bool(self) -> bool | None:
        raise NotImplementedError(self.__class__)

    def raise_on_false(self):
        if self.as_bool is False:
            raise self.to_exception()

    def to_exception(self):
        return InvalidNodeException(BooleanExceptionInfo(self))


class ITypedIndex(INodeIndex, typing.Generic[O, T], ABC):

    @classmethod
    def outer_type(cls) -> type[O]:
        raise NotImplementedError(cls)

    @classmethod
    def item_type(cls) -> type[T]:
        raise NotImplementedError(cls)

    def find_in_node(self, node: INode):
        assert isinstance(node, self.outer_type())
        return self.find_in_outer_node(node)

    def replace_in_target(self, target_node: INode, new_node: INode):
        assert isinstance(target_node, self.outer_type())
        assert isinstance(new_node, self.item_type())
        return self.replace_in_outer_target(target_node, new_node)

    def remove_in_target(self, target_node: INode):
        assert isinstance(target_node, self.outer_type())
        return self.remove_in_outer_target(target_node)

    def find_in_outer_node(self, node: O) -> IOptional[T]:
        raise NotImplementedError(self.__class__)

    def replace_in_outer_target(self, target: O, new_node: T) -> IOptional[O]:
        raise NotImplementedError(self.__class__)

    def remove_in_outer_target(self, target: O) -> IOptional[O]:
        raise NotImplementedError(self.__class__)

class ITypedIntIndex(IInt, ITypedIndex[O, T], typing.Generic[O, T], ABC):

    @classmethod
    def outer_type(cls) -> type[O]:
        raise NotImplementedError

    def find_arg(self, node: INode) -> IOptional[T]:
        result = NodeArgIndex(self.as_int).find_in_node(node)
        if result.value is not None:
            assert isinstance(result.value, self.item_type()), \
                f'{type(result.value)} != {self.item_type()}'
        return typing.cast(IOptional[T], result)

    def replace_arg(self, target: K, new_node: T) -> IOptional[K]:
        return NodeArgIndex(self.as_int).replace_in_target(target, new_node)

    def remove_arg(self, target: K) -> IOptional[K]:
        return NodeArgIndex(self.as_int).remove_in_target(target)

class IInstantiable(INode, ABC):

    @classmethod
    def new(cls, *args: int | INode | typing.Type[INode]) -> typing.Self:
        return cls(*args)

    @property
    def as_instance(self) -> typing.Self:
        return self

###########################################################
############### TEMPORARY AUXILIAR CLASSES ################
###########################################################

class TmpNestedArg:

    def __init__(self, node: BaseNode, idx: int):
        assert isinstance(node, BaseNode)
        assert isinstance(idx, int)
        self.node = node
        self.idx = idx

    def apply(self) -> BaseNode:
        node = self.node
        idx = self.idx
        node_aux = node.args[idx-1]
        assert isinstance(node_aux, BaseNode)
        node = node_aux
        return node

class TmpNestedArgs:

    def __init__(self, node: BaseNode, idxs: tuple[int, ...]):
        assert isinstance(node, BaseNode)
        assert all(isinstance(idx, int) for idx in idxs)
        self.node = node
        self.idxs = idxs

    def apply(self) -> BaseNode:
        node = self.node
        idxs = self.idxs
        for idx in idxs:
            node_aux = node.args[idx-1]
            assert isinstance(node_aux, BaseNode)
            node = node_aux
        return node

###########################################################
######################## BASE NODE ########################
###########################################################

class BaseNode(INode, ABC):

    def __init__(self, *args: int | INode | typing.Type[INode]):
        if not isinstance(self, IInstantiable):
            raise NotImplementedError(self.__class__)
        self._args = args
        self._cached_length: int | None = None
        self._cached_hash: int | None = None

    @property
    def args(self) -> tuple[int | INode | typing.Type[INode], ...]:
        return self._args

    @property
    def as_node(self) -> BaseNode:
        return self

    def __eq__(self, other) -> bool:
        if not isinstance(other, INode):
            return False
        if not self.func == other.as_node.func:
            return False
        if not len(self.args) == len(other.as_node.args):
            return False
        for i, arg in enumerate(self.args):
            if arg != other.as_node.args[i]:
                return False
        return True

    def __getitem__(self, index: INodeIndex):
        return index.find_in_node(self)

    def replace_at(self, index: INodeIndex, new_node: INode):
        return index.replace_in_target(self, new_node)

    def __len__(self) -> int:
        if self._cached_length is not None:
            return self._cached_length
        length = 1 + sum(
            len(arg.as_node)
            for arg in self.args
            if isinstance(arg, INode)
        )
        self._cached_length = length
        return length

    def __hash__(self) -> int:
        if self._cached_hash is not None:
            return self._cached_hash
        hash_value = hash((self.func, self.args))
        self._cached_hash = hash_value
        return hash_value

    def find_until(self, node_type: type[T], until_type: type[INode] | None) -> set[T]:
        return {
            node
            for node in self.args
            if isinstance(node, node_type)
        }.union({
            node
            for parent_node in self.args
            if (
                isinstance(parent_node, INode)
                and
                (until_type is None or not isinstance(parent_node, until_type))
            )
            for node in parent_node.as_node.find_until(node_type, until_type)
        })

    def find(self, node_type: type[T]) -> set[T]:
        return self.find_until(node_type, None)

    def replace_until(
        self,
        before: INode,
        after: INode,
        until_type: type[INode] | None,
    ) -> INode:
        if self == before:
            return after
        return self.func(*[
            (
                arg.as_node.replace_until(before, after, until_type)
                if (
                    isinstance(arg, INode)
                    and
                    (until_type is None or not isinstance(arg, until_type))
                )
                else arg
            )
            for arg in self.args
        ])

    def replace(self, before: INode, after: INode) -> INode:
        return self.replace_until(before, after, None)

    def subs(self, mapping: dict[INode, INode]) -> INode:
        node: INode = self
        for key, value in mapping.items():
            node = node.as_node.replace(key, value)
        return node

    def has(self, node: INode) -> bool:
        return node in self.as_node.find(node.__class__)

    def has_until(self, node: INode, until_type: type[INode] | None) -> bool:
        return node in self.as_node.find_until(node.__class__, until_type)

    def eval(self) -> sympy.Basic:
        name = self.func.__name__
        if len(self.args) == 0:
            return sympy.Symbol(name)

        def eval_arg(arg: int | INode | typing.Type[INode]):
            if isinstance(arg, INode):
                return arg.as_node.eval()
            if isinstance(arg, int):
                return sympy.Integer(arg)
            if isinstance(arg, type) and issubclass(arg, INode):
                return sympy.Symbol(arg.__name__)
            raise ValueError(f"Invalid argument: {arg}")

        fn = sympy.Function(name)
        args: list[sympy.Basic] = [eval_arg(arg) for arg in self.args]
        # pylint: disable=not-callable
        return fn(*args)

    def nested_arg(self, idx: int) -> TmpNestedArg:
        return TmpNestedArg(self, idx)

    def nested_args(self, idxs: tuple[int, ...]) -> TmpNestedArgs:
        return TmpNestedArgs(self, idxs)

    def validate(self):
        for arg in self.args:
            if isinstance(arg, INode):
                arg.as_node.validate()

    def cast(self, t: typing.Type[T]) -> T:
        origin = typing.get_origin(t)
        t = origin if origin is not None else t
        assert isinstance(self, t)
        return typing.cast(T, self)

###########################################################
######################## TYPE NODE ########################
###########################################################

class IType(INode, ABC):

    def accepts(self, inner_type: IType) -> bool | None:
        raise NotImplementedError

    def accepted_by(self, outer_type: IType) -> bool | None:
        raise NotImplementedError

    def valid(self, instance: INode) -> bool | None:
        raise NotImplementedError

    def verify(self, instance: INode):
        if self.valid(instance) is False:
            raise InvalidNodeException(TypeExceptionInfo(self, instance))

class TypeNode(BaseNode, IType, IFunction, ISpecialValue, typing.Generic[T], IInstantiable):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup())

    def __init__(self, t: type[T]):
        origin = typing.get_origin(t)
        t = origin if origin is not None else t
        assert isinstance(t, type) and issubclass(t, INode), t
        super().__init__(t)

    @property
    def type(self) -> typing.Type[INode]:
        t = self.args[0]
        return typing.cast(typing.Type[INode], t)

    @property
    def node_value(self) -> INode:
        return self

    def with_arg_group(self, group: BaseGroup) -> INode:
        return self.type.new(*group.as_tuple)

    def accepted_by(self, outer_type: IType) -> bool | None:
        if isinstance(outer_type, TypeNode):
            return issubclass(self.type, outer_type.type)
        return outer_type.accepts(self)

    def accepts(self, inner_type: IType) -> bool | None:
        if isinstance(inner_type, TypeNode):
            return issubclass(inner_type.type, self.type)
        return inner_type.accepted_by(self)

    def valid(self, instance: INode) -> bool | None:
        if not issubclass(self.type, Placeholder):
            if isinstance(instance, Placeholder):
                p_type = instance.type_node.apply().cast(IType)
                return p_type.accepted_by(self)
        return isinstance(instance, self.type)

    def __eq__(self, other) -> bool:
        if not isinstance(other, TypeNode):
            return False
        return self.type == other.type

###########################################################
######################## INT NODE #########################
###########################################################

class BaseInt(BaseNode, IInt, ISpecialValue, ABC):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup())

    def __init__(self, value: int):
        assert isinstance(value, int)
        super().__init__(value)

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls(value)

    @property
    def as_int(self) -> int:
        value = self.args[0]
        assert isinstance(value, int)
        return value

    @property
    def node_value(self) -> INode:
        return self

class Integer(BaseInt, IInstantiable):
    pass

###########################################################
######################## MAIN NODE ########################
###########################################################

class InheritableNode(BaseNode, ABC):

    def __init__(self, *args: INode):
        assert all(isinstance(arg, INode) for arg in args)
        super().__init__(*args)

    @property
    def args(self) -> tuple[INode, ...]:
        return typing.cast(tuple[INode, ...], self._args)

    def __getitem__(self, index: INodeIndex) -> INode | None:
        if isinstance(index, NodeArgIndex):
            args = self.args
            if index.as_int > 0 and index.as_int <= len(args):
                return args[index.as_int - 1]
            return None
        return super().__getitem__(index)

    def replace_at(self, index: INodeIndex, new_node: INode) -> INode | None:
        assert isinstance(index, INodeIndex)
        assert isinstance(new_node, INode)
        if isinstance(index, NodeArgIndex):
            value = index.as_int
            args = self.args
            if 0 <= value <= len(args):
                return self.func(*[
                    (new_node if i == value - 1 else arg)
                    for i, arg in enumerate(args)
                ])
            return None
        return super().replace_at(index, new_node)

    def validate(self):
        super().validate()
        args = self.args
        type_group = self.as_instance.__class__.arg_type_group().group
        if isinstance(type_group, OptionalTypeGroup):
            assert len(args) <= 1
        elif isinstance(type_group, CountableTypeGroup):
            assert len(args) == len(type_group.args)

    def run(self):
        args = self.args
        self.as_instance.arg_type_group().validate_values(DefaultGroup(*args))

###########################################################
###################### SPECIAL NODES ######################
###########################################################

class Void(InheritableNode, IDefault, IInstantiable):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup())

class UnknownType(InheritableNode, IType, IDefault, IInstantiable):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup())

    def accepted_by(self, outer_type: IType) -> bool | None:
        return None

    def accepts(self, inner_type: IType) -> bool | None:
        return None

    def valid(self, instance: INode) -> bool | None:
        return None

class OptionalBase(InheritableNode, IOptional[T], typing.Generic[T], ABC):

    idx_value = 1

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(OptionalTypeGroup(TypeNode(INode)))

    @property
    def value(self) -> T | None:
        if len(self.args) == 0:
            return None
        value = self.nested_arg(self.idx_value).apply()
        return typing.cast(T, value)

    @classmethod
    def from_optional(cls, o: IOptional[T]) -> typing.Self:
        if isinstance(o, cls):
            return o
        return cls(o.value) if o.value is not None else cls()

class Optional(OptionalBase[T], IInstantiable, typing.Generic[T]):
    pass

###########################################################
######################## WRAPPERS #########################
###########################################################

class BooleanWrapper(
    InheritableNode,
    ISingleChild[IBoolean],
    IBoolean,
    ABC,
):

    idx_value = 1

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IBoolean,
        ]))

    @classmethod
    def with_child(cls, child: IBoolean) -> typing.Self:
        return cls(child)

    @property
    def raw_child(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_value)

    @property
    def child(self) -> IBoolean:
        return self.raw_child.apply().cast(IBoolean)

    @property
    def as_bool(self) -> bool | None:
        return self.child.as_bool

class SingleOptionalChildWrapper(
    InheritableNode,
    ISingleOptionalChild[T],
    typing.Generic[T],
    ABC,
):

    idx_value = 1

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IOptional[T],
        ]))

    @property
    def raw_child(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_value)

    @property
    def child(self) -> IOptional[T]:
        return self.raw_child.apply().cast(IOptional[T])

###########################################################
########################## SCOPE ##########################
###########################################################

class ScopeBaseId(BaseInt, IDefault, ABC):

    @classmethod
    def create(cls) -> typing.Self:
        return cls(1)

class ScopeId(ScopeBaseId, IInstantiable):
    pass

class TemporaryScopeId(ScopeBaseId, IInstantiable):
    pass

class Scope(InheritableNode, ABC):

    @property
    def id(self) -> ScopeBaseId:
        raise NotImplementedError

    def has_dependency(self) -> bool:
        raise NotImplementedError

    def replace_id(self, new_id: ScopeBaseId) -> typing.Self:
        node = self.replace_until(self.id, new_id, OpaqueScope)
        assert isinstance(node, self.__class__)
        return node

class SimpleBaseScope(Scope, typing.Generic[T], ABC):

    idx_id = 1
    idx_child = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            ScopeBaseId,
            INode,
        ]))

    @property
    def raw_id(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_id)

    @property
    def id(self) -> ScopeBaseId:
        return self.raw_id.apply().cast(ScopeBaseId)

    @property
    def child(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_child)

    def has_dependency(self) -> bool:
        return self.child.apply().has_until(self.id, OpaqueScope)

class SimpleScope(SimpleBaseScope[T], IInstantiable, typing.Generic[T]):
    pass

class OpaqueScope(SimpleBaseScope[T], typing.Generic[T], ABC):

    @classmethod
    def with_content(cls, child: T) -> typing.Self:
        return cls(ScopeId(1), child)

    @classmethod
    def normalize_from(cls, node: INode, next_id: int) -> INode:
        if isinstance(node, OpaqueScope):
            return node.normalize()
        child_scope_id = (next_id+1) if isinstance(node, Scope) else next_id
        new_args = [
            cls.normalize_from(child, child_scope_id)
            if isinstance(child, INode)
            else child
            for child in node.as_node.args
        ]
        node = node.as_node.func(*new_args)
        if isinstance(node, Scope):
            node = node.replace_id(TemporaryScopeId(next_id))
        return node

    def normalize(self) -> typing.Self:
        next_id = 1
        child_args = [
            self.normalize_from(child, next_id+1)
            if isinstance(child, INode)
            else child
            for child in self.child.apply().args
        ]
        child = self.child.apply().func(*child_args)
        node = self.func(TemporaryScopeId(next_id), child)
        tmp_ids = node.find_until(TemporaryScopeId, OpaqueScope)
        for tmp_id in tmp_ids:
            node_aux = node.replace_until(tmp_id, ScopeId(tmp_id.as_int), OpaqueScope)
            assert isinstance(node_aux, self.__class__)
            node = node_aux
        return node

class LaxOpaqueScope(OpaqueScope[T], IInstantiable, typing.Generic[T]):
    pass

class StrictOpaqueScope(OpaqueScope[T], IInstantiable, typing.Generic[T]):

    def validate(self):
        assert self == self.normalize()
        super().validate()

class Placeholder(InheritableNode, IFromInt, typing.Generic[T], ABC):

    idx_parent_scope = 1
    idx_index = 2
    idx_type_node = 3

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            ScopeBaseId,
            BaseInt,
            IType,
        ]))

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls(
            ScopeId.create(),
            Integer(value),
            UnknownType(),
        )

    @property
    def parent_scope(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_parent_scope)

    @property
    def index(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_index)

    @property
    def type_node(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_type_node)

class Param(Placeholder[T], IInstantiable, typing.Generic[T]):
    pass

class Var(Placeholder[T], IInstantiable, typing.Generic[T]):
    pass

###########################################################
######################## NODE IDXS ########################
###########################################################

class NodeIntBaseIndex(BaseInt, INodeIndex, IInt, ABC):
    pass

class NodeMainBaseIndex(NodeIntBaseIndex, ABC):

    @classmethod
    def _inner_getitem(cls, node: INode, index: int) -> tuple[INode | None, int]:
        assert index > 0
        index -= 1
        if index == 0:
            return node, index
        for arg in node.as_node.args:
            if isinstance(arg, INode):
                arg_item, index = cls._inner_getitem(arg, index)
                if index == 0:
                    return arg_item, index
        return None, index

    @classmethod
    def _replace(
        cls,
        target_node: INode,
        new_node: INode,
        index: int,
    ) -> tuple[INode | None, int]:
        assert index > 0
        index -= 1
        if index == 0:
            return new_node, index
        for i, arg in enumerate(target_node.as_node.args):
            if isinstance(arg, INode):
                new_arg, index = cls._replace(arg, new_node, index)
                if index == 0:
                    if new_arg is None:
                        return None, index
                    return target_node.func(*[
                        (new_arg if i == j else old_arg)
                        for j, old_arg in enumerate(target_node.as_node.args)
                    ]), index
        return None, index

    @classmethod
    def _ancestors(cls, node: INode, index: int) -> tuple[list[INode], int]:
        assert index > 0
        index -= 1
        if index == 0:
            return [node], index
        for arg in node.as_node.args:
            if isinstance(arg, INode):
                ancestors, index = cls._ancestors(node, index)
                if index == 0:
                    if len(ancestors) == 0:
                        return [], index
                    return [node] + ancestors, index
        return [], index

    def find_in_node(self, node: INode):
        item, _ = self._inner_getitem(node, self.as_int)
        return Optional(item) if item is not None else Optional()

    def replace_in_target(self, target_node: INode, new_node: INode):
        new_target, _ = self._replace(target_node, new_node, self.as_int)
        return Optional(new_target) if new_target is not None else Optional()

    def ancestors(self, node: INode) -> tuple[INode, ...]:
        ancestors, _ = self._ancestors(node, self.as_int)
        return tuple(ancestors)

class NodeMainIndex(NodeMainBaseIndex, IInstantiable):
    pass

class NodeArgBaseIndex(NodeIntBaseIndex, ABC):

    def find_in_node(self, node: INode) -> IOptional[INode]:
        index = self.as_int
        if not isinstance(node, InheritableNode):
            return Optional()
        args = node.args
        if 0 < index <= len(args):
            return Optional(args[index - 1])
        return Optional()

    def replace_in_target(self, target_node: T, new_node: INode) -> IOptional[T]:
        if not isinstance(target_node, InheritableNode):
            return Optional()
        index = self.as_int
        args = target_node.args
        if 0 < index <= len(args):
            new_target = target_node.func(*[
                (new_node if i == index - 1 else arg)
                for i, arg in enumerate(args)
            ])
            assert isinstance(new_target, type(target_node))
            return Optional(typing.cast(T, new_target))
        return Optional()

    def remove_in_target(self, target_node: T) -> Optional[T]:
        if not isinstance(target_node, InheritableNode):
            return Optional()
        index = self.as_int
        args = target_node.args
        if 0 < index <= len(args):
            new_args = [
                arg
                for i, arg in enumerate(args)
                if i != index - 1
            ]
            new_target = target_node.func(*new_args)
            assert isinstance(new_target, type(target_node))
            return Optional(typing.cast(T, new_target))
        return Optional()

class NodeArgIndex(NodeArgBaseIndex, IInstantiable):
    pass

###########################################################
####################### ITEMS GROUP #######################
###########################################################

class BaseGroup(InheritableNode, IGroup[T], typing.Generic[T], ABC):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(RestTypeGroup(TypeNode(cls.item_type())))

    @property
    def args(self) -> tuple[T, ...]:
        return typing.cast(tuple[T, ...], self._args)

    @property
    def as_tuple(self) -> tuple[T, ...]:
        return self.args

    @classmethod
    def from_items(cls, items: typing.Sequence[T]) -> typing.Self:
        return cls(*items)

class DefaultGroup(BaseGroup[INode], IInstantiable):

    @classmethod
    def item_type(cls):
        return INode

class IntGroup(BaseGroup[IInt], IInstantiable):

    @classmethod
    def item_type(cls):
        return IInt

class OptionalValueGroup(BaseGroup[IOptional[T]], IFromInt, IInstantiable, typing.Generic[T]):

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls(*[Optional() for _ in range(value)])

    @classmethod
    def item_type(cls):
        return IOptional[T]

    @classmethod
    def from_optional_items(cls, items: typing.Sequence[T | None]) -> typing.Self:
        return cls(*[Optional(item) if item is not None else Optional() for item in items])

    def strict(self) -> DefaultGroup:
        values = [item.value for item in self.args if item.value is not None]
        assert len(values) == len(self.args)
        return DefaultGroup(*values)

    def fill_with_void(self) -> DefaultGroup:
        def get_value(v: T | None) -> INode:
            if v is None:
                return Void()
            return v
        values = [get_value(item.value) for item in self.args]
        return DefaultGroup(*values)

    def new_amount(self, amount: int) -> typing.Self:
        items = self.as_tuple
        if amount < len(items):
            return self.from_items(list(items)[:amount])
        elif amount > len(items):
            diff = amount - len(items)
            new_items = list(items) + [Optional()] * diff
            return self.from_items(new_items)
        return self

###########################################################
####################### TYPE GROUPS #######################
###########################################################

class IBaseTypeGroup(INode, ABC):
    pass

class CountableTypeGroup(BaseGroup[IType], IBaseTypeGroup, IInstantiable, typing.Generic[T]):

    @classmethod
    def item_type(cls) -> type[IType]:
        return IType

    @classmethod
    def from_types(cls, types: typing.Sequence[type[INode]]) -> typing.Self:
        return cls(*[TypeNode(t) for t in types])

class SingleValueTypeGroup(
    InheritableNode,
    IBaseTypeGroup,
    ISingleChild[IType],
    ABC,
):

    idx_type_node = 1

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IType,
        ]))

    @classmethod
    def with_child(cls, child: IType):
        return cls(child)

    @property
    def type_node(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_type_node)

    @property
    def child(self) -> IType:
        return self.type_node.apply().cast(IType)


class RestTypeGroup(SingleValueTypeGroup, IInstantiable):
    pass

class OptionalTypeGroup(SingleValueTypeGroup, IInstantiable):
    pass

class ExtendedTypeGroup(
    InheritableNode,
    IDefault,
    IFromSingleChild[IOptional[IInt]],
    IInstantiable,
):

    idx_group = 1

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IBaseTypeGroup,
        ]))

    @property
    def group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_group)

    def validate_values(self, values: BaseGroup):
        group = self.group.apply().cast(IBaseTypeGroup)
        args = values.args

        if isinstance(group, CountableTypeGroup):
            assert len(args) == len(group.args)
            for i, arg in enumerate(args):
                t_arg = group.args[i]
                t_arg.verify(arg)
                assert isinstance(arg, INode)
        elif isinstance(group, SingleValueTypeGroup):
            t_arg = group.type_node.apply().cast(IType)
            for arg in args:
                t_arg.verify(arg)
                assert isinstance(arg, INode)
        else:
            raise ValueError(f"Invalid group type: {group}")

    @classmethod
    def create(cls) -> typing.Self:
        return cls.rest()

    @classmethod
    def with_child(cls, child: IOptional[INT]) -> typing.Self:
        value = child.value.as_int if child.value is not None else None
        if value is None:
            return cls.rest()
        return cls(
            CountableTypeGroup.from_items([UnknownType()] * value))

    @classmethod
    def rest(cls, type_node: IType = UnknownType()) -> typing.Self:
        return cls(RestTypeGroup(type_node))

    def new_amount(self, amount: int) -> typing.Self:
        group = self.group.apply().cast(IBaseTypeGroup)
        if not isinstance(group, CountableTypeGroup):
            return self.with_child(Optional(Integer(amount)))
        items = group.as_tuple
        if amount == len(items):
            return self
        return self.func(
            CountableTypeGroup.from_items([
                (items[i] if i < len(items) else UnknownType())
                for i in range(amount)
            ]))

###########################################################
########################## TYPES ##########################
###########################################################

class UnionType(InheritableNode, IType, IInstantiable):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup.rest(IType.as_type())

    def accepted_by(self, outer_type: IType) -> bool | None:
        return IntersectionType(*self.args).accepts(outer_type)

    def accepts(self, inner_type: IType) -> bool | None:
        args = [arg for arg in self.args if isinstance(arg, IType)]
        assert len(args) == len(self.args)
        items = [t.accepts(inner_type) for t in args]
        if any([item is True for item in items]):
            return True
        if all([item is False for item in items]):
            return False
        return None

    def valid(self, instance: INode) -> bool | None:
        args = [arg for arg in self.args if isinstance(arg, IType)]
        assert len(args) == len(self.args)
        items = [t.valid(instance) for t in args]
        if any([item is True for item in items]):
            return True
        if all([item is False for item in items]):
            return False
        return None

class IntersectionType(InheritableNode, IType, IInstantiable):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup.rest(IType.as_type())

    def accepted_by(self, outer_type: IType) -> bool | None:
        return UnionType(*self.args).accepts(outer_type)

    def accepts(self, inner_type: IType) -> bool | None:
        args = [arg for arg in self.args if isinstance(arg, IType)]
        assert len(args) == len(self.args)
        items = [t.accepts(inner_type) for t in args]
        if any([item is False for item in items]):
            return False
        if all([item is True for item in items]):
            return True
        return None

    def valid(self, instance: INode) -> bool | None:
        args = [arg for arg in self.args if isinstance(arg, IType)]
        assert len(args) == len(self.args)
        items = [t.valid(instance) for t in args]
        if any([item is False for item in items]):
            return False
        if all([item is True for item in items]):
            return True
        return None

###########################################################
###################### FUNCTION NODE ######################
###########################################################

class FunctionExprBase(
    InheritableNode,
    IFunction,
    IFromSingleChild[T],
    typing.Generic[T],
    ABC,
):

    idx_param_type_group = 1
    idx_scope = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            ExtendedTypeGroup,
            SimpleBaseScope[T],
        ]))

    @classmethod
    def with_child(cls, child: T) -> typing.Self:
        return cls.new(ExtendedTypeGroup.rest(), child)

    @property
    def param_type_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_param_type_group)

    @property
    def scope(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_scope)

    @property
    def scope_id(self) -> TmpNestedArgs:
        return self.nested_args((self.idx_scope, SimpleBaseScope.idx_id))

    @property
    def expr(self) -> TmpNestedArgs:
        return self.nested_args((self.idx_scope, SimpleBaseScope.idx_child))

    def owned_params(self) -> typing.Sequence[Param]:
        params = [
            param
            for param in self.expr.apply().find_until(Param, OpaqueScope)
            if param.parent_scope.apply().cast(IInt).as_int == self.scope_id]
        params = sorted(params, key=lambda param: param.index.apply().cast(IInt).as_int)
        return params

    def with_args(self, *args: INode) -> INode:
        return self.with_arg_group(DefaultGroup(*args))

    def with_arg_group(self, group: BaseGroup) -> INode:
        self.validate()
        type_group = self.param_type_group.apply().cast(ExtendedTypeGroup)
        args = group.as_tuple
        type_group.validate_values(group)
        params = self.owned_params()
        scope = self.scope.apply().cast(SimpleBaseScope)
        for param in params:
            index = param.index.apply().cast(IInt).as_int
            assert index > 0
            assert index <= len(args)
            arg = args[index-1]
            param_type = param.type_node.apply().cast(IType)
            param_type.verify(arg)
            scope_aux = scope.replace_until(param, arg, OpaqueScope)
            assert isinstance(scope_aux, SimpleBaseScope)
            scope = scope_aux
        assert not scope.has_dependency()
        return scope.child.apply()

    def validate(self):
        params = self.owned_params()
        group = self.nested_args(
            (self.idx_param_type_group, ExtendedTypeGroup.idx_group)
        ).apply().cast(IBaseTypeGroup)
        scope = self.scope
        for param in params:
            index = param.index.apply().cast(IInt).as_int
            assert index > 0
            if isinstance(group, CountableTypeGroup):
                assert index <= len(group.args)
                g_type_node = group.args[index-1]
                g_type_node.verify(param)
            elif isinstance(group, SingleValueTypeGroup):
                g_type_node = group.type_node.apply().cast(IType)
                g_type_node.verify(param)
            else:
                raise ValueError(f"Invalid group type: {group}")
        if isinstance(scope, OpaqueScope):
            all_inner_functions = scope.find_until(FunctionExpr, OpaqueScope)
            all_functions = all_inner_functions.union(self)
            all_functions_scope_ids = {f.scope.id for f in all_functions}
            all_inner_params = scope.find_until(Param, OpaqueScope)
            all_inner_params_scope_ids = {
                p.parent_scope.apply().cast(IInt).as_int
                for p in all_inner_params
            }
            assert all_inner_params_scope_ids.issubset(all_functions_scope_ids)
        super().validate()

class FunctionExpr(
    FunctionExprBase[T],
    IInstantiable,
    typing.Generic[T],
):
    pass

class FunctionCall(InheritableNode, IInstantiable):

    idx_function = 1
    idx_arg_group = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IFunction,
            BaseGroup,
        ]))

    @property
    def function(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_function)

    @property
    def arg_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_arg_group)

    @classmethod
    def define(cls, fn: IFunction, args: BaseGroup) -> INode:
        if isinstance(fn, TypeNode):
            return fn.type.new(*args.as_tuple)
        return cls(fn, args)

###########################################################
######################## EXCEPTION ########################
###########################################################

class IExceptionInfo(INode, ABC):

    def as_exception(self):
        return InvalidNodeException(self)

class BooleanExceptionInfo(
    BooleanWrapper,
    IExceptionInfo,
    IInstantiable,
):
    pass

class TypeExceptionInfo(
    InheritableNode,
    IExceptionInfo,
    IInstantiable,
):

    idx_type = 1
    idx_node = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IType,
            INode,
        ]))

    @property
    def type(self):
        return self.nested_arg(self.idx_type)

    @property
    def node(self):
        return self.nested_arg(self.idx_node)

class InvalidNodeException(Exception):

    def __init__(self, info: IExceptionInfo):
        super().__init__(info)

    @property
    def info(self) -> IExceptionInfo:
        info = self.args[0]
        return typing.cast(IExceptionInfo, info)

###########################################################
######################## EXCEPTION ########################
###########################################################

class ExceptionInfoWrapper(
    InheritableNode,
    ISingleChild[IExceptionInfo],
    IExceptionInfo,
    IInstantiable,
):

    idx_info = 1

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IExceptionInfo,
        ]))

    @classmethod
    def with_child(cls, child: IExceptionInfo) -> typing.Self:
        return cls(child)

    @property
    def info(self):
        return self.nested_arg(self.idx_info)

    @property
    def child(self) -> IExceptionInfo:
        return self.info.apply().cast(IExceptionInfo)

###########################################################
################### CORE BOOLEAN NODES ####################
###########################################################

class Not(BooleanWrapper, IInstantiable):

    @property
    def as_bool(self) -> bool | None:
        child = self.raw_child.apply()
        if not isinstance(child, IBoolean):
            return None
        if child.as_bool is None:
            return None
        return not child.as_bool

class IsEmpty(SingleOptionalChildWrapper[INode], IBoolean, IInstantiable):

    def _raise_if_empty(self) -> INode:
        optional = self.nested_arg(self.idx_value).apply()
        if not isinstance(optional, IOptional):
            raise self.to_exception()
        value = optional.value
        if value is None:
            raise self.to_exception()
        return value

    @property
    def value_or_raise(self) -> INode:
        return self._raise_if_empty()

    def raise_if_empty(self):
        self._raise_if_empty()

    @property
    def as_bool(self) -> bool | None:
        value = self.nested_arg(self.idx_value).apply()
        if not isinstance(value, IOptional):
            return None
        return value.value is None

class IsInstance(InheritableNode, IBoolean, typing.Generic[T], IInstantiable):

    idx_instance = 1
    idx_type = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types((
            INode,
            TypeNode[T],
        )))

    @property
    def instance(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_instance)

    @property
    def type(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_type)

    @property
    def as_bool(self) -> bool | None:
        instance = self.instance.apply()
        assert isinstance(instance, INode)
        t = self.type.apply()
        assert isinstance(t, TypeNode)
        valid = t.valid(instance)
        return valid

    @property
    def as_type_or_raise(self) -> T:
        if not self.as_bool:
            raise self.to_exception()
        return typing.cast(T, self.type.apply())

    @classmethod
    def with_args(cls, instance: INode, t: typing.Type[T]) -> IsInstance[T]:
        return typing.cast(IsInstance[T], cls(instance, TypeNode(t)))

    @classmethod
    def assert_type(cls, instance: INode, t: typing.Type[T]) -> T:
        return typing.cast(T, cls(instance, TypeNode(t)).as_type_or_raise)


class Eq(InheritableNode, IBoolean, IInstantiable):

    idx_left = 1
    idx_right = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types((
            INode,
            INode,
        )))

    @property
    def left(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_left)

    @property
    def right(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_right)

    @property
    def as_bool(self) -> bool | None:
        left = self.left.apply()
        right = self.right.apply()
        return left == right

    @classmethod
    def from_ints(cls, left: int, right: int) -> Eq:
        return cls(Integer(left), Integer(right))

class IsInsideRange(InheritableNode, IBoolean, IInstantiable):

    idx_value = 1
    idx_min_value = 2
    idx_max_value = 3

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IInt,
            IInt,
            IInt,
        ]))

    @classmethod
    def from_raw(cls, value: int | IInt, min_value: int, max_value: int) -> typing.Self:
        v = Integer(value) if isinstance(value, int) else value
        return cls(v, Integer(min_value), Integer(max_value))

    @property
    def raw_value(self):
        return self.nested_arg(self.idx_value)

    @property
    def min_value(self):
        return self.nested_arg(self.idx_min_value)

    @property
    def max_value(self):
        return self.nested_arg(self.idx_max_value)

    @property
    def as_bool(self) -> bool | None:
        value = self.raw_value.apply()
        if not isinstance(value, IInt):
            return None
        min_value = self.min_value.apply()
        if not isinstance(min_value, IInt):
            return None
        max_value = self.max_value.apply()
        if not isinstance(max_value, IInt):
            return None
        return min_value.as_int <= value.as_int <= max_value.as_int

class IntBooleanNode(BaseInt, IBoolean, IInstantiable):

    @property
    def as_bool(self) -> bool | None:
        if self.as_int == 0:
            return False
        if self.as_int == 1:
            return True
        return None

class MultiArgBooleanNode(InheritableNode, IBoolean, ABC):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup.rest(IBoolean.as_type())

    @property
    def as_bool(self) -> bool | None:
        raise NotImplementedError

class DoubleIntBooleanNode(InheritableNode, IBoolean, ABC):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IInt,
            IInt,
        ]))

    @property
    def as_bool(self) -> bool | None:
        raise NotImplementedError

class AndNode(MultiArgBooleanNode, IInstantiable):

    @property
    def as_bool(self) -> bool | None:
        args = self.args
        has_none = False
        for arg in args:
            if not isinstance(arg, IBoolean):
                has_none = True
            elif arg.as_bool is None:
                has_none = True
            elif arg.as_bool is False:
                return False
        return None if has_none else True

class OrNode(MultiArgBooleanNode, IInstantiable):

    @property
    def as_bool(self) -> bool | None:
        args = self.args
        has_none = False
        for arg in args:
            if not isinstance(arg, IBoolean):
                has_none = True
            elif arg.as_bool is None:
                has_none = True
            elif arg.as_bool is True:
                return True
        return None if has_none else False

class GreaterThan(DoubleIntBooleanNode, IInstantiable):

    @property
    def as_bool(self) -> bool | None:
        args = self.args
        assert len(args) == 2
        a, b = args
        if not isinstance(a, IInt) or not isinstance(b, IInt):
            return None
        return a.as_int > b.as_int

class LessThan(DoubleIntBooleanNode, IInstantiable):

    @property
    def as_bool(self) -> bool | None:
        args = self.args
        assert len(args) == 2
        a, b = args
        if not isinstance(a, IInt) or not isinstance(b, IInt):
            return None
        return a.as_int < b.as_int
