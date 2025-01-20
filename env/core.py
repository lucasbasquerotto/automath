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

class IInheritableNode(INode, ABC):

    @classmethod
    def new(cls, *args: INode) -> typing.Self:
        raise NotImplementedError(cls)

class ISpecialValue(INode, ABC):

    @property
    def node_value(self) -> INode:
        raise NotImplementedError(self.__class__)

class IDefault(INode, ABC):

    @classmethod
    def create(cls) -> typing.Self:
        return cls()

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

class IFromSingleNode(IInheritableNode, typing.Generic[T], ABC):

    @classmethod
    def with_node(cls, node: T) -> typing.Self:
        return cls.new(node)

class ISingleChild(IFromSingleNode, typing.Generic[T], ABC):

    @property
    def child(self) -> T:
        raise NotImplementedError(self.__class__)

class IOptional(IDefault, IFromSingleNode[T], typing.Generic[T], ABC):

    @property
    def value(self) -> T | None:
        raise NotImplementedError()

    def value_or_else(self, default_value: T) -> T:
        value = self.value
        if value is None:
            value = default_value
        return value

    def is_empty(self) -> IsEmpty:
        return IsEmpty(self)

    @property
    def value_or_raise(self) -> T:
        value = IsEmpty(self).value_or_raise
        return typing.cast(T, value)

    def raise_if_empty(self):
        self.is_empty().raise_if_empty()

class ISingleOptionalChild(ISingleChild[IOptional[T]], typing.Generic[T], ABC):

    @classmethod
    def with_optional(cls, child: T | None) -> typing.Self:
        return cls.with_node(Optional(child) if child is not None else Optional())

    @property
    def child(self) -> IOptional[T]:
        raise NotImplementedError(self.__class__)

class IWrapper(INode, ABC):
    pass

class IGroup(IWrapper, typing.Generic[T], ABC):

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

    def with_arg_group(self, group: BaseGroup, info: RunInfo) -> INode:
        raise NotImplementedError(self.__class__)

class IRunnable(INode, ABC):

    def run(self, info: RunInfo) -> INode:
        raise NotImplementedError(self.__class__)

class IBoolean(INode):

    _true: IBoolean | None = None
    _false: IBoolean | None = None

    @property
    def as_bool(self) -> bool | None:
        raise NotImplementedError(self.__class__)

    @property
    def strict_bool(self) -> bool:
        value = self.as_bool
        if value is None:
            raise self.to_exception()
        return value

    def raise_on_false(self):
        if self.as_bool is False:
            raise self.to_exception()

    def raise_on_not_true(self):
        if self.as_bool is not True:
            raise self.to_exception()

    def raise_on_undefined(self):
        if self.as_bool is None:
            raise self.to_exception()

    def to_exception(self):
        return InvalidNodeException(BooleanExceptionInfo(self))

    @classmethod
    def true(cls) -> IBoolean:
        t = cls._true
        if t is None:
            t = IntBoolean.create_true()
            cls._true = t
        return t

    @classmethod
    def false(cls) -> IBoolean:
        f = cls._false
        if f is None:
            f = IntBoolean.create()
            cls._false = f
        return f

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

class BaseNode(IRunnable, ABC):

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
        assert isinstance(self, t), f'{type(self)} != {t}'
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

    def __hash__(self) -> int:
        if self._cached_hash is not None:
            return self._cached_hash
        full_type_name = self.type.__module__ + '.' + self.type.__name__
        hash_value = hash((self.func, full_type_name))
        self._cached_hash = hash_value
        return hash_value

    def with_arg_group(self, group: BaseGroup, info: RunInfo) -> INode:
        t = self.type
        assert issubclass(t, IInheritableNode)
        return t.new(*group.as_tuple).as_node.run(info)

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

    def run(self, info: RunInfo):
        self.validate()
        return self

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

    def run(self, info: RunInfo):
        self.validate()
        return self

class Integer(BaseInt, IInstantiable):

    _zero: Integer | None = None
    _one: Integer | None = None

    @classmethod
    def from_int(cls, value: int) -> Integer:
        if value == 0:
            z = cls._zero
            if z is None:
                z = cls(value)
                cls._zero = z
            return z
        if value == 1:
            o = cls._one
            if o is None:
                o = cls(value)
                cls._one = o
            return o
        return cls(value)


###########################################################
######################## MAIN NODE ########################
###########################################################

class InheritableNode(BaseNode, IInheritableNode, ABC):

    @classmethod
    def new(cls, *args: INode) -> typing.Self:
        return cls(*args)

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
        type_group = self.arg_type_group().group.apply()
        if isinstance(type_group, OptionalTypeGroup):
            assert len(args) <= 1, \
                f'{type(self)}: {len(args)} > 1'
        elif isinstance(type_group, CountableTypeGroup):
            assert len(args) == len(type_group.args), \
                f'{type(self)}: {len(args)} != {len(type_group.args)}'

    def run(self, info: RunInfo):
        self.validate()
        args = [arg.as_node.run(info) for arg in self.args]
        self.arg_type_group().validate_values(DefaultGroup(*args))
        return self.func(*args)

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

    @classmethod
    def with_value(cls, value: T | None) -> typing.Self:
        return cls(value) if value is not None else cls()

class Optional(OptionalBase[T], IInstantiable, typing.Generic[T]):

    @classmethod
    def with_int(cls, value: int | None) -> Optional[Integer]:
        return Optional(Integer(value)) if value is not None else Optional()

###########################################################
########################## SCOPE ##########################
###########################################################

class ParentScopeBase(BaseInt, IDefault, ABC):

    @classmethod
    def create(cls) -> typing.Self:
        return cls(1)

class NearParentScope(ParentScopeBase, IInstantiable):
    pass

class FarParentScope(ParentScopeBase, IInstantiable):
    pass

class IScope(INode, ABC):
    pass

class IOpaqueScope(IScope, ABC):
    pass

class IInnerScope(IScope, ABC):
    pass

class Placeholder(InheritableNode, IFromInt, typing.Generic[T], ABC):

    idx_parent_scope = 1
    idx_index = 2
    idx_type_node = 3

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            ParentScopeBase,
            BaseInt,
            IType,
        ]))

    @property
    def parent_scope(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_parent_scope)

    @property
    def index(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_index)

    @property
    def type_node(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_type_node)

    def run(self, info: RunInfo):
        node = super().run(info).cast(Placeholder[T])
        parent_scope = node.parent_scope.apply().cast(ParentScopeBase).run(info)
        scope_index = parent_scope.as_int
        index = node.index.apply().cast(BaseInt).run(info)
        scope_data_group = info.scope_data_group.apply().cast(ScopeDataGroup)
        if isinstance(parent_scope, NearParentScope):
            scope_index = len(scope_data_group.as_tuple) - scope_index + 1
        scope = NodeArgIndex(scope_index).find_in_node(scope_data_group).value_or_raise
        assert isinstance(scope, ScopeDataPlaceholderItem)
        assert isinstance(self, scope.item_inner_type())
        item = NodeArgIndex(index.as_int).find_in_node(scope).value_or_raise
        return item.as_node.run(info)

class Param(Placeholder[T], IInstantiable, typing.Generic[T]):

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls(
            FarParentScope.create(),
            Integer(value),
            UnknownType(),
        )

class Var(Placeholder[T], IInstantiable, typing.Generic[T]):

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls(
            NearParentScope.create(),
            Integer(value),
            UnknownType(),
        )

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

class BaseGroup(InheritableNode, IGroup[T], IDefault, typing.Generic[T], ABC):

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

    def amount(self) -> int:
        return len(self.args)

    def amount_node(self) -> Integer:
        return Integer(self.amount())

class DefaultGroup(BaseGroup[INode], IInstantiable):

    @classmethod
    def item_type(cls):
        return INode

class BaseIntGroup(BaseGroup[IInt], ABC):

    @classmethod
    def item_type(cls):
        return IInt

    @classmethod
    def from_ints(cls, indices: typing.Sequence[int]) -> typing.Self:
        return cls(*[Integer(i) for i in indices])

class IntGroup(BaseIntGroup, IInstantiable):
    pass

class NestedArgIndexGroup(BaseIntGroup, IInstantiable):

    def apply(self, node: BaseNode) -> BaseNode:
        args_indices = [arg.as_int for arg in self.args]
        return node.nested_args(tuple(args_indices)).apply()

class BaseOptionalValueGroup(BaseGroup[IOptional[T]], IFromInt, typing.Generic[T], ABC):

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls(*([Optional()]*value))

    @classmethod
    def item_type(cls):
        return Optional[T]

    @classmethod
    def from_optional_items(cls, items: typing.Sequence[T | None]) -> typing.Self:
        return cls(*[Optional.with_value(item) for item in items])

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

class OptionalValueGroup(BaseOptionalValueGroup[T], IInstantiable, typing.Generic[T]):
    pass

###########################################################
####################### TYPE GROUPS #######################
###########################################################

class IBaseTypeGroup(INode, ABC):
    pass

class CountableTypeGroup(
    BaseGroup[IType],
    IBaseTypeGroup,
    IFromInt,
    IInstantiable,
    typing.Generic[T],
):

    @classmethod
    def item_type(cls) -> type[IType]:
        return IType

    @classmethod
    def from_types(cls, types: typing.Sequence[type[INode]]) -> typing.Self:
        return cls(*[TypeNode(t) for t in types])

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls(*[UnknownType()] * value)

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
    IFromInt,
    IFromSingleNode[IOptional[IInt]],
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
    def from_int(cls, value: int) -> typing.Self:
        return cls(CountableTypeGroup.from_int(value))

    @classmethod
    def with_node(cls, node: IOptional[INT]) -> typing.Self:
        value = node.value.as_int if node.value is not None else None
        if value is None:
            return cls.rest()
        return cls.from_int(value)

    @classmethod
    def rest(cls, type_node: IType = UnknownType()) -> typing.Self:
        return cls(RestTypeGroup(type_node))

    def new_amount(self, amount: int) -> typing.Self:
        group = self.group.apply().cast(IBaseTypeGroup)
        if not isinstance(group, CountableTypeGroup):
            return self.with_node(Optional(Integer(amount)))
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

class ScopedFunctionBase(InheritableNode, IFunction, IScope, ABC):

    idx_param_type_group = 1
    idx_expr = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            ExtendedTypeGroup,
            INode,
        ]))

    @property
    def param_type_group(self) -> TmpNestedArg:
        return self.as_node.nested_arg(self.idx_param_type_group)

    @property
    def expr(self) -> TmpNestedArg:
        return self.as_node.nested_arg(self.idx_expr)

    def with_arg_group(self, group: BaseGroup, info: RunInfo) -> INode:
        new_group = group.run(info)
        param_type_group = self.param_type_group.apply().cast(ExtendedTypeGroup)
        param_type_group.validate_values(new_group)
        expr = self.expr.apply()
        scope_data = ScopeDataParamItem(*new_group.as_tuple)
        new_info = (
            info.clear_scopes()
            if isinstance(self, IOpaqueScope)
            else info
        ).add_scope(scope_data)
        return expr.run(new_info)

class FunctionExpr(
    ScopedFunctionBase,
    IFromSingleNode[T],
    IOpaqueScope,
    IInstantiable,
    typing.Generic[T],
):

    @classmethod
    def with_node(cls, node: T) -> typing.Self:
        return cls.new(ExtendedTypeGroup.rest(), node)

class FunctionWrapper(
    ScopedFunctionBase,
    IFromSingleNode[T],
    IInnerScope,
    IInstantiable,
    typing.Generic[T],
):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            ExtendedTypeGroup,
            FunctionCall,
        ]))

    @classmethod
    def with_node(cls, node: T) -> typing.Self:
        return cls.new(ExtendedTypeGroup.rest(), node)

###########################################################
################### BASIC BOOLEAN NODES ###################
###########################################################

class BaseIntBoolean(BaseInt, IBoolean, IDefault, ABC):

    @classmethod
    def create(cls) -> typing.Self:
        return cls(0)

    @classmethod
    def from_bool(cls, value: bool) -> typing.Self:
        assert value in (True, False)
        return cls.from_int(1 if value else 0)

    @classmethod
    def create_true(cls) -> typing.Self:
        return cls(1)

    @property
    def as_bool(self) -> bool | None:
        if self.as_int == 0:
            return False
        if self.as_int == 1:
            return True
        return None

class IntBoolean(BaseIntBoolean, IInstantiable):
    pass

class RunnableBoolean(InheritableNode, IBoolean, ABC):

    def run(self, info: RunInfo) -> INode:
        args = [arg.as_node.run(info) for arg in self.args]
        value = self.func(*args).strict_bool
        return IntBoolean.from_bool(value)

class BooleanWrapper(
    RunnableBoolean,
    ISingleChild[IBoolean],
    ABC,
):

    idx_value = 1

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IBoolean,
        ]))

    @property
    def raw_child(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_value)

    @property
    def child(self) -> IBoolean:
        return self.raw_child.apply().cast(IBoolean)

    @property
    def as_bool(self) -> bool | None:
        return self.child.as_bool

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

class SingleOptionalBooleanChildWrapper(
    RunnableBoolean,
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

class IsEmpty(SingleOptionalBooleanChildWrapper[INode], IInstantiable):

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

class IsInstance(RunnableBoolean, typing.Generic[T], IInstantiable):

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


class Eq(RunnableBoolean, IInstantiable):

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

class IsInsideRange(RunnableBoolean, IInstantiable):

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

class MultiArgBooleanNode(RunnableBoolean, ABC):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup.rest(IBoolean.as_type())

class DoubleIntBooleanNode(RunnableBoolean, ABC):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IInt,
            IInt,
        ]))

    @classmethod
    def with_args(cls, value_1: int, value_2: int) -> DoubleIntBooleanNode:
        return cls(Integer(value_1), Integer(value_2))

class And(MultiArgBooleanNode, IInstantiable):

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

    def run(self, info: RunInfo) -> INode:
        run_args: list[IBoolean] = []
        for arg in self.args:
            run_arg = arg.as_node.run(info)
            assert isinstance(run_arg, IBoolean)
            val_1 = run_arg.as_bool
            if val_1 is False:
                return IBoolean.false()
            run_args.append(run_arg)
        for run_arg in run_args:
            run_arg.raise_on_undefined()
        return IBoolean.true()

class Or(MultiArgBooleanNode, IInstantiable):

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

    def run(self, info: RunInfo) -> INode:
        run_args: list[IBoolean] = []
        for arg in self.args:
            run_arg = arg.as_node.run(info)
            assert isinstance(run_arg, IBoolean)
            val_1 = run_arg.as_bool
            if val_1 is True:
                return IBoolean.true()
            run_args.append(run_arg)
        for run_arg in run_args:
            run_arg.raise_on_undefined()
        return IBoolean.false()

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

###########################################################
################### CONTROL FLOW NODES ####################
###########################################################

class ScopeDataPlaceholderItem(BaseGroup[Optional], ABC):

    @classmethod
    def item_type(cls):
        return Optional

    @classmethod
    def item_inner_type(cls) -> type[Placeholder]:
        raise NotImplementedError

    @classmethod
    def is_dynamic(cls) -> IBoolean:
        raise NotImplementedError

    def add_item(self, index: Integer, value: INode) -> typing.Self:
        self.is_dynamic().raise_on_not_true()
        items = list(self.as_tuple)
        if index.as_int > len(items):
            items = items + [Optional()] * (index.as_int - len(items))
        current = items[index.as_int - 1]
        current.is_empty().raise_on_not_true()
        items[index.as_int - 1] = Optional(value)
        return self.func(*items)

class ScopeDataParamItem(ScopeDataPlaceholderItem, IInstantiable):

    @classmethod
    def item_inner_type(cls):
        return Param

    @classmethod
    def is_dynamic(cls) -> IBoolean:
        return IBoolean.false()

class ScopeDataVarItem(ScopeDataPlaceholderItem, IInstantiable):

    @classmethod
    def item_inner_type(cls):
        return Var

    @classmethod
    def is_dynamic(cls) -> IBoolean:
        return IBoolean.true()

class ScopeDataGroup(BaseGroup[ScopeDataPlaceholderItem], IInstantiable):

    @classmethod
    def item_type(cls):
        return ScopeDataPlaceholderItem

    def add_item(self, item: ScopeDataPlaceholderItem) -> typing.Self:
        items = list(self.as_tuple)
        items.append(item)
        return self.func(*items)

    def add_scope_item(
        self,
        scope_index: Integer,
        item_index: Integer,
        value: INode,
    ) -> typing.Self:
        item = NodeArgIndex(scope_index.as_int).find_in_node(self).value_or_raise
        assert isinstance(item, ScopeDataPlaceholderItem)
        new_item = item.add_item(item_index, value)
        result = NodeArgIndex(
            scope_index.as_int
        ).replace_in_target(self, new_item).value_or_raise
        return result

class RunInfo(InheritableNode, IDefault, IInstantiable):

    idx_scope_data_group = 1

    @classmethod
    def create(cls) -> typing.Self:
        return cls(ScopeDataGroup())

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            ScopeDataGroup,
        ]))

    @property
    def scope_data_group(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_scope_data_group)

    def add_scope(self, item: ScopeDataPlaceholderItem) -> typing.Self:
        group = self.scope_data_group.apply().cast(ScopeDataGroup)
        new_group = group.add_item(item)
        return self.func(new_group)

    def clear_scopes(self) -> typing.Self:
        return self.func(ScopeDataGroup())

    def add_scope_item(
        self,
        scope_index: Integer,
        item_index: Integer,
        value: INode,
    ) -> typing.Self:
        group = self.scope_data_group.apply().cast(ScopeDataGroup)
        new_group = group.add_scope_item(scope_index, item_index, value)
        return self.func(new_group)

class ControlFlowBaseNode(InheritableNode, ABC):

    def run(self, info: RunInfo):
        self.validate()
        return self._run(info)

    def _run(self, info: RunInfo):
        raise NotImplementedError(self.__class__)

class FunctionCall(ControlFlowBaseNode, IInstantiable):

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
        return cls(fn, args)

    def _run(self, info: RunInfo):
        fn = self.function.apply()
        assert isinstance(fn, IFunction)
        arg_group = self.arg_group.apply().run(info)
        assert isinstance(arg_group, BaseGroup)
        return fn.with_arg_group(group=arg_group, info=info)

class If(ControlFlowBaseNode, IInstantiable):

    idx_condition = 1
    idx_true_expr = 2
    idx_false_expr = 3

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IBoolean,
            INode,
            INode,
        ]))

    @property
    def condition(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_condition)

    @property
    def true_expr(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_true_expr)

    @property
    def false_expr(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_false_expr)

    def _run(self, info: RunInfo):
        condition = self.condition.apply().run(info)
        assert isinstance(condition, IBoolean)
        flag = condition.strict_bool
        if flag:
            return self.true_expr.apply().run(info)
        return self.false_expr.apply().run(info)

class LoopGuard(InheritableNode, IInstantiable):

    idx_condition = 1
    idx_result = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IBoolean,
            INode,
        ]))
    @property
    def condition(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_condition)

    @property
    def result(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_result)

    @classmethod
    def with_args(cls, condition: IBoolean, result: INode) -> LoopGuard:
        return cls(condition, result)

class Loop(ControlFlowBaseNode, IInstantiable):

    idx_callback = 1

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IFunction,
        ]))

    @property
    def callback(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_callback)

    def _run(self, info: RunInfo):
        condition = True
        data: Optional[INode] = Optional()
        while condition:
            fn = self.callback.apply()
            result = FunctionCall(fn, DefaultGroup(data)).as_node.run(info)
            assert isinstance(result, LoopGuard)
            cond_node = result.condition.apply().run(info)
            assert isinstance(cond_node, IBoolean)
            cond_node.raise_on_undefined()
            new_data = result.result.apply().run(info)
            data = Optional(new_data)
            cond_node.raise_on_undefined()
            condition = cond_node.strict_bool
        return data
